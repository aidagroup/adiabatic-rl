import copy
import sys
import numpy       as np
import tensorflow  as tf
import math 

from cl_replay.api.experiment    import Experiment, Experiment_Replay
from cl_replay.api.parsing       import Kwarg_Parser
from cl_replay.api.utils         import log

from cl_replay.architecture.ewc.model.LWF  import LWF

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_LWF(Experiment_Replay):


    def _init_parser(self, **kwargs):
        Experiment._init_parser(self, **kwargs)
        self.flags = kwargs

        self.model_type        = self.parser.add_argument('--model_type', type=str, default='lwf_dnn', help='determines model architecture.')

        self.extra_eval        = self.parser.add_argument('--extra_eval', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')
        self.forgetting_tasks  = self.parser.add_argument('--forgetting_tasks', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define forgetting tasks.')


    def create_model(self):
        # TODO: see Experiment_DGR.py, allow layer creation via bash file...
        model_inputs  = tf.keras.Input(self.get_input_shape())
        if self.model_type == 'lwf_dnn':
            flat    = tf.keras.layers.Flatten()(model_inputs)
            dense_1 = tf.keras.layers.Dense(128, activation="relu")(flat)
            #dense_2 = tf.keras.layers.Dense(512, activation="relu")(dense_1)
            out     = tf.keras.layers.Dense(128, activation="relu")(dense_1)
        model_outputs  = tf.keras.layers.Dense(name="prediction", units=self.num_classes)(out)

        self.model = LWF(inputs=model_inputs, outputs=model_outputs, **self.flags)
        self.model.compile(run_eagerly=True, optimizer=None)
        
        task_list = 
        
        self.model.set_parameters({'tasks' : self.task_list})
        
        return self.model


    def get_input_shape(self):
        return self.h, self.w, self.c


    def before_task(self, task, **kwargs):
        """ generates datasets of past tasks if necessary and resets model layers """
        model_upd_kwargs = {'current_task' : task}
        
        if task > 1:
            # prepare some helper data regarding tasks & classes!
            past_classes = set()
            for p_class in self.tasks[1:task]:
                past_classes.update(p_class) 
            log.debug(f"past classes so far: {self.past_classes}")
            # forgetting: simply delete prev_task entry!
            for i, f_task in enumerate(self.forgetting_tasks):
                if f_task < task: # task to forget was learned in the past, so forget it ASAP
                    log.info(f'deleting FIM and stored params for T{f_task}.')
                    self.prev_tasks = [task for task in self.model.prev_tasks if task not in f_task]
                    
                    del self.forgetting_tasks[i]

            # create a copy of the previous model for each replay task
            model_upd_kwargs.update({
                'copy' : copy.deepcopy(self.model), 'prev_tasks' : self.prev_tasks
            })
            
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
        self.model.prev_tasks.append(task)
        

    def train_on_task(self, task):
        if task > 1:
            # -------- LwF extra classifier training
            # 1) perform training on the new task with the classification layer/network, only optimize last linear layer
            for layer in self.model.layers[:-1]: layer.trainable = False  # freeze all layers except the classification layer
            # ----
            if self.train_method == 'batch':
                self.train_batch_mode(data=self.sampler(), epochs=self.epochs, train_steps=self.train_steps, current_task=task, use_callbacks=False)
                return
            
            log.info('{:20s}'.format(f' [START] training on task: {task} total epochs: {self.epochs} ').center(64, '~'))
            train_history = self.model.fit(self.sampler(),
                                        epochs=int(self.epochs[task-1]),
                                        batch_size=self.batch_size,
                                        steps_per_epoch=self.train_steps,
                                        # callbacks=self.train_callbacks,  # NOTE: callback is disabled to not log metrics during classifier traininsg
                                        verbose=self.verbosity)
            # ----
            # 2) unfreeze layers and continue normal training
            for layer in self.model.layers[:-1]: layer.trainable = True
            # --------

        super().train_on_task(task)
            
            
    def _test(self, task):
        super()._test(task)
        if self.extra_eval != []:
            _, self.eeval_test, _, self.eeval_amount = self.dataset.get_dataset(self.extra_eval, task_info=None)
            self.model.test_task = f'EXTRA'  # test task identifier
            log.info(f'\t[TEST] -> {self.model.test_task}')
            self.model.evaluate(x=self.eeval_test[0], y=self.eeval_test[1],
                                batch_size=self.test_batch_size,
                                steps=(self.eeval_amount//self.test_batch_size),
                                callbacks=self.eval_callbacks,
                                verbose=self.verbosity,
                                return_dict=True)


if __name__ == '__main__':
    Experiment_LWF().run_experiment()