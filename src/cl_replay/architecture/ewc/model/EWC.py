import time
import tensorflow      as tf
import numpy           as np

from collections       import defaultdict
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients

from cl_replay.api.model.DNN    import DNN
from cl_replay.api.parsing      import Kwarg_Parser
from cl_replay.api.utils        import log


class EWC(DNN):
    
    
    def __init__(self, inputs, outputs, **kwargs):
        super(EWC, self).__init__(inputs, outputs, **kwargs)
        self.parser        = Kwarg_Parser(**kwargs)
        self.lambda_       = self.parser.add_argument('--lambda', type=float, default=100., help='EWC lambda')
        self.init_tf_variables()
        
        self.tasks = {}
        self.prev_tasks = []
        self.current_task = -1
        self.other_classes = []
        self.current_classes = []


    def call(self, inputs, task):
        if self.multi_head:
            x = super(EWC, self).call(inputs)[task]
        else:
            x = super(EWC, self).call(inputs)    
        return x


    def randomize_weights(self): 
        for var in self.trainable_variables:
            var.assign(tf.random.truncated_normal(var.shape, 0, 0.1, dtype=self.dtype_tf))


    def init_tf_variables(self):
        self.ewc_storage = {}  # dict indexed by task (T0 - TN), has elements corr. to trainable_variables
        self.fims        = {}  # dict indexed by task (T0 - TN)

        # NOTE: SGD or ADAM?
        # self.opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, weight_decay=0.0005)


    def train_step(self, data, **kwargs):
        xs, ys, sw = data[0], data[1], data[2]
        
        if self.multi_head:
            ys = tf.gather(ys, indices=self.current_classes, axis=-1)
        
        t1 = time.time()

        gradients, trainable_vars = self.compute_gradients(xs, ys, sw)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        
        self.all_metrics[-1].update_state(delta)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data, **kwargs):
        xs, ys = data[0], data[1]
        
        if self.multi_head and self.current_task != 0:
            ys = tf.gather(ys, indices=self.current_classes, axis=-1)
        
        t1 = time.time()
                
        logits = self(inputs=xs, task=self.current_task-1)

        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms

        self.all_metrics[0].update_state(ys, logits)
        self.all_metrics[-1].update_state(delta)
        
        return {m.name: m.result() for m in self.metrics}


    def compute_loss(self, xs, ys, logits, sample_weight=None):
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))
         
        if self.current_task > 1: self.compute_ewc_penalty()
        
        self.all_metrics[0].update_state(ys, logits)
        self.all_metrics[1].update_state((self.loss))
        
        return self.loss


    def compute_gradients(self, xs, ys, sw):
        with tf.GradientTape() as g:
            out = self(xs, task=self.current_task-1)
            loss = self.compute_loss(xs, ys, out, sw)

        trainable_vars = self.trainable_variables
        return g.gradient(loss, trainable_vars), trainable_vars


    def compute_ewc_penalty(self):
        for task in self.prev_tasks:  # calculates the EWC loss penalty
            # print(f'computing EWC penalty term for task T{task}.')
            for var, var_prev, fim_var_prev in zip(self.trainable_variables, self.ewc_storage[task], self.fims[task]):
                # print(var.shape, var_prev.shape, fim_var_prev.shape)
                self.loss += tf.reduce_sum(fim_var_prev * (var - var_prev)**2)
        self.loss *= self.lambda_


    def compute_fim(self, data):
        log.debug(f'computing FIM for task T{self.current_task}.')
        
        xs, ys = data
        num_samples = data[0].shape[0]

        trainable_vars = self.trainable_variables
        if self.multi_head:
            trainable_vars = [var for var in trainable_vars if 'dense' or f'classification_head{self.current_task-1}' in var.name]
        
        variance = [tf.zeros_like(t_v) for t_v in trainable_vars]
        
        for x in xs:  # iterate sample for sample -> true ewc
            x = np.expand_dims(x, axis=0)
            with tf.GradientTape() as g:
                model_out = self.call(inputs=x, task=self.current_task-2)
                log_likelihood = tf.nn.log_softmax(model_out)          

            gradients = g.gradient(log_likelihood, trainable_vars)
            gradients = [g for g in gradients if type(g) is not type(None)] 
            # for g in gradients:
            #     print(g.shape)
            #     print(tf.reduce_min(g), tf.reduce_max(g))
            
            # We accumulate the second-order partial derivates, squaring the first-order partial derivate is sufficient.
            # Using a common alternative to the fisher matrix def.: evaluate it from the negative expected value of the Hessian of the log-likelihood:
            # $$ \mathcal{F}(\theta) = -\mathbb{E} \left[ \frac{\partial^2}{\partial \theta^2} \log L(X; \theta) \right $$
            # This relates to the curvature of the log-likelihood function around the maximum likelihood estimate.
            variance = [var + (grad**2) for var, grad in zip(variance, gradients)]
        
        fisher_diagonal = [tensor / num_samples for tensor in variance]  # NOTE: has to be done at the end

        for f_var in fisher_diagonal:
            print(f_var.shape, tf.reduce_min(f_var), tf.reduce_max(f_var))

        # copy variables
        self.fims[self.current_task]        = [tf.constant(variances + 0.) for variances in fisher_diagonal]
        self.ewc_storage[self.current_task] = [tf.constant(var + 0.) for var in trainable_vars]
  

    def set_parameters(self, **kwargs):
        self.current_task = kwargs.get('current_task', None)
        self.prev_tasks = kwargs.get('prev_tasks', None)
        self.all_classes = kwargs.get('all_classes', None)
        self.current_classes = kwargs.get('current_classes', None)
        self.other_classes = kwargs.get('other_classes', None)


    def apply_imm_after_task(self, mode, imm_transfer_type, imm_alpha, current_task):
        if mode == 'ewc'    : return
        if current_task == 0: return

        prev_task_weight = 1. - imm_alpha
        cur_task_weight  = imm_alpha
        if mode == 'mean_imm':
            for var, prev_var in zip(self.trainable_variables, self.ewc_storage[current_task - 1]):
                var.assign(prev_task_weight * prev_var + cur_task_weight * var)

        if mode == 'mode_imm':
            for var, prev_var, oldfim_var, fim_var, fim_b in zip(
                self.trainable_variables, self.ewc_storage[current_task - 1], self.fims[current_task - 1], self.fims[current_task]):
                common_var = prev_task_weight * oldfim_var + cur_task_weight * fim_var + 1e-30

        new_var = (prev_task_weight * prev_var * oldfim_var + cur_task_weight * var * fim_var) / common_var
        var.assign(new_var)
