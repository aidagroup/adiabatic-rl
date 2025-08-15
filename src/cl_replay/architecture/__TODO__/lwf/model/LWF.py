import time
import tensorflow      as tf
import numpy           as np

from collections       import defaultdict

from cl_replay.api.model.DNN    import DNN
from cl_replay.api.parsing      import Kwarg_Parser
from cl_replay.api.utils        import log


class LWF(DNN):
    
    def __init__(self, inputs, outputs, **kwargs):
        
        
        super(LWF, self).__init__(inputs, outputs, **kwargs)
        self.parser        = Kwarg_Parser(**kwargs)
        self.lwf_alpha     = self.parser.add_argument('--lwf_alpha', type=float, default=1.0, help='LwF alpha')
        self.lwf_temp      = self.parser.add_argument('--lwf_temp', type=float, default=2.0, help='LwF temperature')
        self.sgd_eps       = self.parser.add_argument('--sgd_eps', type=float, default=0.01, help='SGD learning rate')
        self.sgd_wdecay    = self.parser.add_argument('--sgd_wdecay', type=float, default=5e-4, help='SGD weight decay')

        self.tasks = {}
        self.all_classes = []
        self.prev_tasks = []
        self.current_task = -1
        self.copy = None 
        
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.sgd_eps, momentum=0.0, weight_decay=self.sgd_wdecay)


    def mod_kl_div(self, old, new):
        ''' KD loss, see Hinton et al. (NIPS 2015). '''
        return -tf.mean(tf.reduce_sum(tf.multiply(old, tf.math.log(new)), axis=1))


    def smooth(self, logits, temp, dim):
        ''' Calculate modified version of probabilities. 
            LwF uses T=2 according to a grid-search. 
        '''
        log = tf.square(logits, (1/temp))  

        return log / tf.expand_dims(tf.reduce_sum(log, axis=dim), axis=1)


    def init_tf_variables(self):
        return


    def train_step(self, data, **kwargs):
        xs, ys, sw = data[0], data[1], data[2]
        
        t1 = time.time()

        with tf.GradientTape() as g:
            # ---- LwF applied
            if self.current_task > 1:
                for prev_task in self.prev_tasks: # iterate through previous tasks
                    # TODO: use prev_task information to identify which classes to filter
                    prev_classes = self.tasks.get(prev_task)
                    print(prev_classes)
                    # --> we only need logits & targets for previous classes contained in exactly that task!
                    logits = self(xs)[...,prev_classes] # slice everything except indices of available classes for the current task
                    print(logits)
                    with g.stop_recording():
                        targets = self.copy(xs)[...,prev_classes]
                        print(targets)                    
                    self.loss += self.lwf_alpha * self.mod_kl_div(
                        self.smooth(tf.nn.softmax(targets, axis=-1), self.lwf_temp, 1),
                        self.smooth(tf.nn.softmax(logits, axis=-1), self.lwf_temp, 1)
                    ) / len(self.prev_tasks)
            else:
                logits = self(xs)
                self.loss = self.compute_loss(xs, ys, logits, sw)
                
        # ---- continue opt -> backwards pass & optimization step
        gradients = g.gradient(self.loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        
        self.all_metrics[-1].update_state(delta)

        return {m.name: m.result() for m in self.metrics}


    def call(self, inputs, training=None, *args, **kwargs):
        self.fwd = self.forward(input_tensor=inputs)
        
        return self.fwd
    

    def forward(self, input_tensor):
        x = self(input_tensor)
        
        return x

    def compute_loss(self, xs, ys, logits, sample_weight=None):

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys))
        
        # self.all_metrics[0].update_state(ys, logits)
        # self.all_metrics[1].update_state((self.dnn_loss + self.lambda_ * self.ewc_loss))
        
        return self.loss


    def set_parameters(self, **kwargs):
        self.current_task = kwargs['current_task']
        self.prev_tasks = kwargs['prev_tasks']
        self.tasks = kwargs['tasks']
        self.copy = kwargs['copy']
