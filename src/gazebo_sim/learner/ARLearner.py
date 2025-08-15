import math
import sys, os
import numpy as np
import tensorflow as tf ;

import time ;

import matplotlib
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

import argparse ;


from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models

from gazebo_sim.learner.DQNLearner import DQNLearner ;
from gazebo_sim.model.GMM import build_model ;

from cl_replay.architecture.ar.model import DCGMM
from cl_replay.api.utils import log, change_loglevel ;
import logging ;


class ARLearner(DQNLearner):
    def __init__(self, n_actions, obs_space, config):
        self.config,_ = self.parse_args() ;
        
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = self.config.train_batch_size
        #self.task = self.config.start_as_task

        self.epsilon = self.epsilon_init = self.config.initial_epsilon
        self.epsilon_delta = self.config.epsilon_delta
        self.eps_replay_factor = self.config.eps_replay_factor
        self.eps_min = self.config.final_epsilon
        self.last_dice = -1
        self.gamma = self.config.gamma ;
        self.target_network_update_freq = self.config.dqn_target_network_update_freq ;

        self.input_dims = obs_space
        
        # TODO should be called from RLAgent
        self.before_experiment()

        self.exploration = True ;
        change_loglevel("INFO") ;


    # -----------------------------------> PRE/POST ROUTINES


    def build_models(self):
        #print("!!", self.config.dqn_target_network) ;
        self.n_actions = len(self.action_space) ;
        # vanilla, model and target_model are the same. DQN and DDQN are not supported for da moment
        if True:
            self.model = build_model("AR", self.input_dims, len(self.action_space), self.config) ;
            self.target_model = self.model ;
            self.gmm_opt = self.model.opt_and_layers[-2][0] ;
            self.ro_opt = self.model.opt_and_layers[-1][0] ;
            self.gmm_layer = self.model.layers[-2] ;
            self.ro_layer = self.model.layers[-1] ;
     
            self.frozen_model = build_model("AR-frozen", self.input_dims, len(self.action_space), self.config) ;


# ---------------------------------------> BEFORE/AFTER
    def before_experiment(self):
        if self.config.replay_buffer == 'prioritized':
            self.replay_buffer = PrioritizedReplayBuffer(self.config.capacity, self.input_dims, self.config.per_alpha, self.config.per_beta, self.config.per_eps, self.config.per_delta_beta)
        else:
            self.replay_buffer = ReplayBuffer(self.config.capacity, self.input_dims)

        self.build_models() ;


    def before_task(self, task):
        DQNLearner.before_task(self, task) ;
        print("------------------------BEFORE!!", self.config.start_task_ar) ;
        self.update_frozen_model(task) ;
        self.train_step = 0 ;

    def update_frozen_model(self, task):
        if task >= self.config.start_task_ar:
          print("--------------------COPYING!!") ;
          self.model.reset() ;
          self.copy_model_weights(self.model, self.frozen_model) ;
          self.frozen_model.layers[-2].tf_somSigma.assign (1.0) ; # TODO: param


    def after_task(self, task):
        DQNLearner.after_task(self, task) ;



    def learn(self, task):

        self.train_step += 1 ;
        if self.train_step < self.batch_size:
            return  # if buffer is not full yet -> pass
        
        states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)

        td_error = self.learn_vanilla(task, states, actions, rewards, states_, terminal)

        if self.train_step % 100 == 0:
          print("VIS/PROTOS") ;
          self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'gmm_T{task}_{self.train_step}.png'), "current")
          self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'frozen_T{task}_{self.train_step}.png'), "frozen")



        self.update_epsilon()


    # simple Q LEARN, no ddqn for the moment!
    def learn_vanilla(self, task, states, actions, rewards, nextStates, dones, weights=1.0, learn_gmm = True, learn_ro = True):
        #print("F", self.ro_layer.get_grad_factors())
        t_start = time.time_ns()
        #print("TASK!!!", task, self.train_step) ;
        ratio = self.config.ar_replay_ratio ;

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        nextStates = tf.convert_to_tensor(nextStates, dtype=tf.float32)

        # generate samples and targets from frozen model
        if task >= self.config.start_task_ar:
            states_gen = tf.concat([self.frozen_model.do_variant_generation(
                states, selection_layer_index=-2) for i in range(0,ratio) ], axis=0) ;
            #self.vis_samples(states_gen, ratio, 32,"gen") ;
            #self.vis_samples(states, 1, 32,"real") ; 
            #np.save(f"{self.config.root_dir}/results/{self.config.exp_id}/gen{self.train_step}.npz", states_gen.numpy())

            outputs_gen = self.frozen_model(states_gen) ;
            actions_gen =  tf.concat([actions for i in range(0,ratio)],axis=0) ;
            q_gen = tf.gather(outputs_gen, actions_gen, batch_dims=1) ;


        # GENERATE real TARGETS use the Bellman eqn
        pred_real_next = self.invoke_target_model(nextStates) # 32,6
        q_real = self.bellman(pred_real_next, rewards, dones) # N,1
        #print(q_real.numpy()) ;

        gmm_vars = self.gmm_layer.trainable_variables
        ro_vars = self.ro_layer.trainable_variables

        if learn_gmm == True: 
          self.gmm_layer.active = True ;
          self.gmm_layer.pre_train_step()
        if learn_ro == True: 
          self.ro_layer.active = True ;
          self.ro_layer.pre_train_step()

        with tf.GradientTape(persistent=True) as g:
            pred_real = tf.gather(self.model(states), actions, batch_dims=1) ;
            # normal gmm loss
            gmm_loss_real = -1. * tf.reduce_mean(self.gmm_layer.loss_fn(y_pred=self.gmm_layer.get_fwd_result())) ;
            #print(self.gmm_layer.get_fwd_result().numpy()[:,0,0,:].argmax(axis=1))
            # normal td error
            ro_loss_real = tf.reduce_mean(tf.square(pred_real-q_real)) ;
            gmm_loss = gmm_loss_real ;
            ro_loss = ro_loss_real ;

            if task >= self.config.start_task_ar:
                pred_gen = tf.gather(self.model(states_gen),actions_gen,batch_dims=1) ;
                # gmm loss on gen samples
                gmm_loss_gen = -1. * tf.reduce_mean(self.gmm_layer.loss_fn(y_pred=self.gmm_layer.get_fwd_result())) ;
                # distillation td loss
                ro_loss_gen = tf.reduce_mean(tf.square(pred_gen-q_gen)) ;
                gmm_loss = (1./(ratio+1.)) * gmm_loss_real + ((ratio)/(ratio+1.)) * gmm_loss_gen ;
                ro_loss = ro_loss_real + ro_loss_gen ;
                print("gen loss", gmm_loss_gen, "real loss", gmm_loss_real, "somSigma", self.gmm_layer.tf_somSigma) ;

            self.gmm_layer.set_layer_loss(gmm_loss)

        gmm_grads = 0.0 ; ro_grads = 0.0 ;
        if learn_gmm == True:
          gmm_grads = g.gradient(gmm_loss, gmm_vars)
          gmm_grads_vars = DCGMM.factor_gradients(
            zip(gmm_grads, gmm_vars), self.gmm_layer.get_grad_factors())
          self.gmm_opt.apply_gradients(gmm_grads_vars)
          self.gmm_layer.post_train_step()
        if learn_ro == True:
          ro_grads = g.gradient(ro_loss, ro_vars)
          #print([(g.numpy().max(), g.numpy().min()) for g in ro_grads]) ;
          # access weight gradients
          ro_grads_vars = DCGMM.factor_gradients(
            zip(ro_grads, ro_vars), self.ro_layer.get_grad_factors())
          #print([(g.numpy().min(), g.numpy().max()) for (g,v) in ro_grads_vars])  
          self.ro_opt.apply_gradients(ro_grads_vars)

        del g

        t_end = time.time_ns()
        self.train_step_duration = np.divide(np.abs(np.diff([t_start, t_end])), 1e9)

        return ro_loss ;


    def get_current_status(self):
      return (self.epsilon,) ;



    def vis_protos(self, name, which_model):
        K = self.gmm_layer.K
        c = self.gmm_layer.c_in
        mus = self.gmm_layer.mus
        if which_model == "frozen":
          mus = self.frozen_model.layers[-2].mus ;
        mus = mus.numpy()
        mus = mus.reshape(K, c)
        f, axes = plt.subplots(int(math.sqrt(K)), int(math.sqrt(K)))
        f.set_figwidth(int(math.sqrt(K))) ;
        f.set_figheight(int(math.sqrt(K))) ;

        for i, ax in enumerate(axes.ravel()):
            ax.imshow(mus[i].reshape(self.input_dims))
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
        #plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99,top=0.99,bottom=0.01)
        plt.savefig(name, bbox_inches='tight')
        plt.close('all')

    def vis_samples(self, samples, rows, cols, name):

        f,ax = plt.subplots(rows, cols) ;
        for _ax,sample in zip(ax.ravel(), samples.numpy()):
          #print("minmax=", sample.min(), sample.max()) ;
          _ax.imshow(sample) ;
          _ax.set_xticklabels([])
          _ax.set_yticklabels([])
          _ax.set_axis_off() ;
          #_ax.set_aspect('equal')
        plt.subplots_adjust(wspace=0.1, hspace=0.1) ;
        plt.axis('off') ;
        #plt.tight_layout() ;
        #print(f"{self.config.root_dir}/results/{self.config.exp_id}/{name}{self.train_step}.png") ;
        plt.savefig(f"{self.config.root_dir}/results/{self.config.exp_id}/{name}{self.train_step}.png") ;



    def parse_args(self):
        parser = argparse.ArgumentParser()
        self.define_base_args(parser) ;

        parser.add_argument('--start_task_ar',       type=int,   default=1,      help='at what task should AR start? Duh!')

        qgmm_group = parser.add_argument_group('qgmm')
        qgmm_group.add_argument('--ar_replay_ratio',                type=int,   default=2,                                     help='Ratio between generated and real samples')
        qgmm_group.add_argument('--qgmm_K',                         type=int,   default=100,                                     help='Number of K components to use for the GMM layer.')
        qgmm_group.add_argument('--qgmm_eps_0',                     type=float, default=0.011,                                  help='Start epsilon value (initial learning rate).')
        qgmm_group.add_argument('--qgmm_eps_inf',                   type=float, default=0.01,                                   help='Smallest epsilon value (learning rate) for regularization.')
        qgmm_group.add_argument('--qgmm_lambda_sigma',              type=float, default=0.,                                     help='Sigma factor.')
        qgmm_group.add_argument('--qgmm_lambda_pi',                 type=float, default=0.,                                     help='Pis factor.')
        qgmm_group.add_argument('--qgmm_alpha',                     type=float, default=0.01,                                   help='Regularizer alpha.')
        qgmm_group.add_argument('--qgmm_gamma',                     type=float, default=0.9,                                    help='Regularizer gamma.')
        qgmm_group.add_argument('--qgmm_regEps',                    type=float, default=0.01,                                   help='Learning rate for the readout layer (SGD epsilon).')
        qgmm_group.add_argument('--qgmm_lambda_W',                  type=float, default=1.,                                     help='Weight factor.')
        qgmm_group.add_argument('--qgmm_lambda_b',                  type=float, default=0.,                                     help='Bias factor.')
        qgmm_group.add_argument('--qgmm_reset_somSigma',            type=float, default=0.5,                      help='Resetting annealing radius which is set with each new sub-task.')      
        qgmm_group.add_argument('--qgmm_somSigma_sampling',         type=str,   default='yes',        choices=['yes', 'no'],    help='Activate to uniformly sample from a radius around the BMU.')
        qgmm_group.add_argument('--qgmm_log_protos_each_n',         type=int,   default=0,                                      help='Saves protos (as an image) each N steps.')
        qgmm_group.add_argument('--qgmm_load_ckpt',                 type=str,   default="no",choices=["yes","no"],                                            help='Provide a path to a checkpoint file for a warm-start of the GMM.')
        qgmm_group.add_argument('--qgmm_init_forward',              type=str,   default='yes',        choices=['yes', 'no'],    help='Init weights to favor driving forward.')

        config, unparsed = parser.parse_known_args()
        return config, unparsed ;
  


