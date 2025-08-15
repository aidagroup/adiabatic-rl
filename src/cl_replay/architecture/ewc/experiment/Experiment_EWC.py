import sys
import numpy       as np
import tensorflow  as tf
import math 

from cl_replay.api.experiment    import Experiment, Experiment_Replay
from cl_replay.api.parsing       import Kwarg_Parser
from cl_replay.api.utils         import log

from cl_replay.architecture.ewc.model.EWC  import EWC

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_EWC(Experiment_Replay):


    def _init_parser(self, **kwargs):
        Experiment._init_parser(self, **kwargs)
        self.flags = kwargs

        self.mode              = self.parser.add_argument('--mode',                 type=str,   default='ewc',              choices=['ewc', 'mean_imm', 'mode_imm']   , help='what type of imm. [default="ewc", "mean_imm"]')
        self.imm_transfer_type = self.parser.add_argument('--imm_transfer_type',    type=str,   default='weight_transfer',  choices=['L2_transfer', 'weight_transfer'], help='what type of imm?')
        self.imm_alpha         = self.parser.add_argument('--imm_alpha',            type=float, default=0.5,                help='balancing parameter')
        self.model_type        = self.parser.add_argument('--model_type',           type=str,   default='EWC',              help='class to load form module "model"')

        self.extra_eval        = self.parser.add_argument('--extra_eval', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')
        self.forgetting_tasks  = self.parser.add_argument('--forgetting_tasks', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define forgetting tasks.')

        self.multi_head        = self.parser.add_argument('--multi_head',           type=str,   default='no',               choices=['yes', 'no'], help='one classification head per task?')

        self.prev_tasks = []


    def create_model(self):
        # TODO: see Experiment_DGR.py, allow layer creation via bash file...
        model_inputs  = tf.keras.Input(self.get_input_shape())
        if self.model_type == 'ewc_dnn':
            flat    = tf.keras.layers.Flatten()(model_inputs)
            dense_1 = tf.keras.layers.Dense(128, activation="relu", name='dense_1')(flat)
            # dense_2 = tf.keras.layers.Dense(512, activation="relu", name='dense_2)(dense_1)
            out     = tf.keras.layers.Dense(128, activation="relu", name='dense_3')(dense_1)

        if self.multi_head == 'yes':
            heads   = []            
            heads_cls_map = dict()
            for i, task_classes in enumerate(self.tasks[1:]):
                heads_cls_map.update({ i : {}})
                head_ = tf.keras.layers.Dense(len(task_classes), name=f'classification_head_{i}')(out)
                head_.cls_mapping = dict()
                for j, cls in enumerate(task_classes):
                    heads_cls_map[i].update({j : cls})                
                heads.append(head_)
            model_outputs = heads
            
        else:
            model_outputs  = tf.keras.layers.Dense(name="prediction", units=self.num_classes)(out)

        self.model = EWC(inputs=model_inputs, outputs=model_outputs, **self.flags)
        self.model.compile(run_eagerly=True, optimizer=None)
        self.model.summary()
        
        if self.multi_head == 'yes':
            self.model.multi_head = self.multi_head
            self.model.base_model_size = 3
            self.model.heads_cls_map = heads_cls_map

        self.model.set_parameters(**{'tasks' : self.tasks})

        return self.model


    def get_input_shape(self):
        return self.h, self.w, self.c
    
    
    def before_task(self, task, **kwargs):
        """ generates datasets of past tasks if necessary and resets model layers """
        if self.mode in ['mean_imm', 'mode_imm'] and self.imm_transfer_type == 'L2_transfer': # for weight-transfer, we keep previous weights
            self.model.randomize_weights()
        
        if task > 1:
            # forgetting: delete FIM & old params of task to forget -> !only task-wise deletion possible!
            for i, f_task in enumerate(self.forgetting_tasks):
                if f_task < task: # task to forget was learned in the past, so forget it ASAP
                    log.info(f'deleting FIM and stored params for T{f_task}.')
                    self.prev_tasks = [task for task in self.prev_tasks if task != f_task]

                    del self.model.ewc_storage[f_task]
                    del self.model.fims[f_task]
                    
                    del self.forgetting_tasks[i]
        
        current_classes = set(self.tasks[task])
        other_classes = set()
        for task_classes in self.tasks[task:]:
            other_classes = (other_classes ^ set(task_classes)) - current_classes            
        
        model_upd_kwargs = {
            'current_task' : task, 
            'prev_tasks': self.prev_tasks, 
            'current_classes' : list(current_classes),
            'other_classes': list(other_classes)
        }
        self.model.set_parameters(**model_upd_kwargs)

        current_train_set = self.training_sets[task]
        self.sampler.reset()
        self.feed_sampler(task, current_train_set)
    
    
    def feed_sampler(self, task, current_data):
        cur_xs, cur_ys = current_data
        self.sampler.add_subtask(xs=cur_xs, ys=cur_ys)
        self.sampler.set_proportions([1.])
        
        self.train_steps = cur_xs.shape[0] // self.batch_size
        
        log.info(f'setting up "steps_per_epoch"... iterations for current task T{task}: {self.train_steps},')
    
    
    def after_task(self, task, **kwargs):
        self.prev_tasks.append(task)
        
        if task == len(self.tasks)-1: return
        
        self.model.compute_fim(self.training_sets[task])
        
        if self.mode == 'ewc': return
        if self.mode in ['mean_imm', 'mode_imm']:
            self.model.apply_imm_after_task(self.mode, self.imm_transfer_type, self.imm_alpha, task)   # merge params from previous and current task
            # self._test(task)                                                                          # call tests after IMM update


    def _test(self, task):
        super()._test(task)
        # NOTE: extra evaluation is problematic for multi-headed classifier
        # it would be necessary to determine which head is in charge for each class and combine the results from multiple forward calls ...
        # -----
        if self.extra_eval != [] and self.multi_head == 'no':
            _, self.eeval_test, _, self.eeval_amount = self.dataset.get_dataset(self.extra_eval, task_info=None)
            self.model.test_task = f'EXTRA'  # test task identifier
            log.info(f'\t[TEST] -> {self.model.test_task}({np.unique(np.argmax(self.eeval_test[1], axis=-1))})')
            self.model.evaluate(x=self.eeval_test[0], y=self.eeval_test[1],
                                batch_size=self.test_batch_size,
                                steps=(self.eeval_amount//self.test_batch_size),
                                callbacks=self.eval_callbacks,
                                verbose=self.verbosity,
                                return_dict=True)


if __name__ == '__main__':
    Experiment_EWC().run_experiment()
