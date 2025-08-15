import math
import numpy                as np
import tensorflow           as tf

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log



class NearestMean_Layer(Custom_Layer):


    def __init__(self, **kwargs):
        super(NearestMean_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs

        self.input_layer            = self.parser.add_argument('--input_layer',         type=int,   default=[None],             help='a list of prefixes of this layer inputs')
        #-------------------------- SAMPLING
        self.num_classes            = self.parser.add_argument('--num_classes',         type=int,   default=10,                 help='number of output classes')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,                help='sampling batch size')
        #-------------------------- LEARNING
        self.batch_size             = self.parser.add_argument('--batch_size',          type=int,   default=100,                help='bs')
        self.K                      = self.parser.add_argument('--K')
        
        # K: take top K measured distances.
        # each K is a vote, class with minimum distance to the evaluated sample is the forecast for the instance.
        # -> "classification by majority vote"
        

    def build(self, input_shape):
        self.input_shape    = input_shape
        self.channels_in    = np.prod(input_shape[1:])
        self.channels_out   = self.num_classes

        self.fwd, self.return_loss, self.raw_return_loss    = None, None, None
        self.resp_mask                                      = None

        self.build_layer_metrics()


    def call(self, inputs, training=None, *args, **kwargs):
        self.fwd = self.forward(input_tensor=inputs)

        return self.fwd


    #@tf.function(autograph=False)
    def forward(self, input_tensor):
        
        # do we need to compare with all samples from current train set?
        # later on we need these data samples since classes will vanish from train set -> build a reservoir with N samples for each class?
        
        # calc distance between input_tensor and K elements
        # sort distances
        # chose first k distances
        
        # distance measure:
            # 1. euclidean: d(A,B) = \sqrt{\sum (A_i - B_i)**2}
            # 2. L1 distance: d(A,B) = \sum |A_i - B_i|
            # 3. cosine similarity: d(A,B) = 1 - \frac{A \cdot B}{||A|| ||B||}

        # prediction: tf.argmin(distance)

        # GMM prototype(s) with highest resp(s), maybe pick top 3 activations?
        #  -> for each sample we calc the distance to our reservoir of samples?

        # Layer is not trained! Only fwd(), however samples need to be selected (rebalanced?) accordingly

        # return self.logits
        return


    # def get_fwd_result(self):       return self.fwd
    # def get_output_result(self):    return self.logits


    def pre_train_step(self):
        return


    def reset_layer(self, **kwargs):
        return


    def backwards(self, topdown=None, **kwargs):
        ''' 
        Performs a sampling operation.
        This should select prototypes according to the classes to generate.
        '''
        return 


    def compute_output_shape(self, input_shape):
        ''' Returns a tuple containing the output shape of this layers computation. '''
        return self.batch_size, self.channels_out


    def set_parameters(self, **kwargs):
        return


    def get_layer_opt(self):
        ''' Returns the optimizer instance attached to this layer. '''
        return None


    def build_layer_metrics(self):
        self.layer_metrics = [
            # tf.keras.metrics.CategoricalAccuracy(name=f'{self.prefix}acc') # uses one-hot
        ]


    def get_layer_metrics(self):
        return self.layer_metrics


    def get_logging_params(self):
        return {}
