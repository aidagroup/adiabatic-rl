"""
Central class for orchestrating a CRL experiment
Runs tasks, episodes for training and eval!
All governed by a cmd line arguments
"""

import socket ;
from argparse import ArgumentParser;

import os
import time
from datetime import datetime
import itertools
import numpy as np
from pprint import pformat


from threading import Thread, Event;
from queue import Queue ;

import imageio ;


# function for reading socket input in a non-blocking way for debugging
# start another terminal and execute trigger.py
# !!! IF you do not start trigger.py, this function will just block and do nothing
# !!! This does not matter since it is on a separate thread
# WE LISTEN ON localhost:config.debugPort over ipv4
def thread_fn(q, stop_event, port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ;
  s.bind(("127.0.0.1", port)) ;
  print("Listening...") ;
  s.listen(3) ;

  # waiting for incoming connections and rechicking avaery 1.0 seconds
  # important to do it non-blockingly since otherwise the function/thread would never 
  # terminate if no connection is accepted. We would be stuck in the blocking accept() method
  s.settimeout(1.0) ;
  conn_accepted = False ;
  while conn_accepted == False:
    if stop_event.is_set() == True: return ;
    conn_accepted = True ;
    try:
      conn, addr = s.accept() ; 
    except:
      conn_accepted = False ;

  print("conn accepted!") ;
  conn.settimeout(1.0)

  while True:

    s = "" ;
    while s=="":
      if stop_event.is_set() == True: return ;
      try:
        msg = conn.recv(1024) ;
        s = msg.decode() ;
      except:
        s = "" ;
    conn.send(b"confirmed!")
    q.put(s) ;
    #print("Put a new cmd", s) ;



class RLAgent(object):
    def __init__(self, env, learner):
        self.config, unparsed =  self.parse_args() ;

        # debugging via tcp/ip messages that are observed in a separate thread (thread_fn)
        self.q = Queue() ;
        self.stop_event = Event() ; 
        self.kbd_thread = Thread(target=thread_fn,args=(self.q,self.stop_event, self.config.debug_port)) ;
        self.kbd_thread.start() ;
        self.debug_flag = False ;
        self.wait_flag = False ;
        self.next_cmd = "" ;
        self.debug_action = None ;

        self.environment = env ;
        self.learner = learner ;
        

        self.results_folder = self.config.root_dir + '/results/'
        self.results_folder += self.config.exp_id + "/" ;
        os.makedirs(self.results_folder, exist_ok=True)
        # ensure checkpoint subfolder exists
        self.ckpt_folder = os.path.join(self.results_folder, 'ckpt')
        os.makedirs(self.ckpt_folder, exist_ok=True)
        print("Created results dir:", self.results_folder) ;
        args_path = os.path.join(self.results_folder,'args.txt')
        with open(args_path, 'w') as file:
          file.write('TIME: '+ str(datetime.now()))
          file.write('\n--- RL Agent ---\n')
          file.write(pformat(vars(self.config)))
          file.write('\n--- Environment ---\n')
          file.write(pformat(vars(self.environment.config)))
          file.write('\n--- Learner ---\n')
          file.write(pformat(vars(self.learner.config)))
        self.save_counter = 0
        self.task = 0 ;

    # potentially clean up simulator etc., delete temp files, ...
    # here, we just terminate the debug thread so the whole program can exit
    def mop_up(self):
      self.stop_event.set() ;
      self.kbd_thread.join() ;


    # check whether keyboard thread has new input
    def get_command(self):
      if self.q.empty() == False:
        return self.q.get() ;
      else:
        tmp = self.next_cmd ;
        self.next_cmd = "" ;
        return tmp ;


    # just for inspecting what the DNN/AR instance does! 
    # Governed by enter-terminated kbd inputs running on another thread (see above)
    def debug(self,obs_tm1, obs_t,a_t,r_t, term, trunc, info):        
        key = self.get_command() ;

        save_flag = False ;
        save_key = "1" ;
        plus_minus = 0.0 ;
        if key != "":
          #print(key)
          if key == "d": self.debug_flag = (not self.debug_flag) ;
          if key == 'e': self.eval_flag = (not self.eval_flag) ;
          if key == 'w': self.wait_flag = (not self.wait_flag) ;
          if key == '+': plus_minus = 0.1 ;
          if key == '-': plus_minus =  -0.1 ;
          if key in ['1','2']: save_flag = True ; save_key = key ;
          self.learner.epsilon += plus_minus ;

        if self.debug_flag:
          #print("shape", obs_t.shape, obs_t.dtype, obs_t.max()) ;
          imageio.imwrite("camera.png",(obs_tm1*255).astype("uint8")) ;
          imageio.imwrite("camera2.png",(obs_t*255).astype("uint8")) ;

          print("DEBUG: Action", a_t, r_t, self.learner.epsilon) ;

          if self.wait_flag:
            x = self.get_command() ;
            while x =="": 
              x = self.get_command() ;
            self.next_cmd = x ;
            if x[0] == "a": self.debug_action = int(x[1:]) ;

    def debug_before_step(self, exploration, action_t):
        if self.debug_flag==True:
          print("Exploration" if exploration==True else "Policy","-- Action ", action_t) ;
        if self.debug_action != None:
          print("External action!!") ;
          tmp = self.debug_action 
          self.debug_action = None;
          return tmp ;
        else:
          return action_t ;


    def save2csv(self,fname,data):
      fname = f'{self.save_counter}_' + fname
      self.save_counter += 1
      path = os.path.join(self.results_folder, fname)
      f = open(path, "w") ;
      for line in data:
        tmp = "" ;
        for entry in line:
          tmp += " " + str(entry) ;
        f.write(tmp+"\n") ;
      f.close() ;

    # train OR eval, depends on train flag
    def do_episode(self, task, i, train=True):
        truncated = False
        terminated = False
        score = 0
        counter = 0
        print("---> start episode", i, " TRAIN" if train==True else "EVAL") ;
        if train==True:
          self.learner.enable_exploration() ;

        observation_tm1, _ = self.environment.reset()

        # debug here as well so we can see initial states and actions
        self.debug(observation_tm1, observation_tm1, 0, 0, False, False, {});
        reward_t = 0.0
        while not truncated and not terminated: # while not done

            # let learner choose action, or else chose action by external debugging
            action_t, exp = self.learner.choose_action(observation_tm1)            
            action_t = self.debug_before_step(exp, action_t) ;

            # perform one env step
            observation_t, reward_t, terminated, truncated, info = self.environment.step(action_t)

            # debug listening for socket commands
            self.debug(observation_tm1, observation_t, action_t, reward_t, terminated,truncated, info);

            # adapt learner
            if train==True:
              self.learner.store_transition(observation_tm1,action_t,reward_t,observation_t,terminated or truncated) # better: to POLearner!
              self.learner.learn(task)	
            observation_tm1 = observation_t

            # update stats
            score += reward_t
            counter+=1 
        print(f" task {task}, episode {i}, length {counter}") #, cube {self.environment.current_name}, mass {self.environment.current_mass}, score {score}")
        return (score, reward_t, counter) ;


    # train or eval, depends on flag "train"
    def do_task(self, task, train=True):
      train_stats = [] ;
      task_total_iterations = 0 ;

      self.environment.switch(task) ;
      self.learner.set_task(task) ;
      self.learner.disable_exploration() ;

      if train==True:
        self.learner.enable_exploration() ;
        self.learner.before_task(task) ;
      
      print("----start task ", task) ;
      task_episode_counter = -1 ;
      curr_stat_learner = tuple('learner_stats' for _ in self.learner.get_current_status())
      curr_stat_env = tuple('env_stats' for _ in self.environment.get_current_status())
      train_stats.append(('task_episode_counter',) + ('score', 'episode_length') + ('last_reward',) + curr_stat_learner + curr_stat_env) ;
      while True:
        # execute episode
        task_episode_counter += 1 ;
        score, last_reward, episode_length = self.do_episode(task, task_episode_counter, train=train) ;
        train_stats.append((task_episode_counter,) +  (score, episode_length) + (last_reward,) + self.learner.get_current_status() + self.environment.get_current_status()) ;
        task_total_iterations += episode_length ;

        # check whether training duration has been exceeded, whether mrasures in episodes or timesteps
        current_duration_train = task_episode_counter if self.config.training_duration_unit == "episodes" else task_total_iterations ;
        current_duration_eval = task_episode_counter if self.config.evaluation_duration_unit == "episodes" else task_total_iterations ;
        current_duration = current_duration_train if train==True else current_duration_eval ;
        duration_limit = self.config.training_duration if train==True else self.config.evaluation_duration ;
        if task == 0 and self.config.training_duration_task_0 != -1 and train==True: duration_limit = self.config.training_duration_task_0 ;
        #print(current_duration, duration_limit) ;
        if  current_duration > duration_limit: break ;

      if train==True: self.learner.after_task(task) ;

      return train_stats ;


    ## assumes that the last task is eval only
    def go(self):
        self.learner.before_experiment() ;

        if self.config.start_task > 0:
          print("Loading task", self.config.start_task) ;
          self.learner.load(os.path.join(self.results_folder, 'ckpt', self.config.exp_id + str(self.config.start_task-1)+".weights.h5")) ;

        nr_tasks = self.environment.get_nr_of_tasks() ;
        for task in range(self.config.start_task,nr_tasks):          
          print("GO task", task) ;

          # evaluation of previous tasks: 0,..., task-1
          for eval_task in range(self.config.eval_start_task,task):
            eval_stats = self.do_task(eval_task, train=False) ;
            self.save2csv(f"eval_resultsT{eval_task}_before_T{task}.csv", np.array(eval_stats)) ;

          # by convention: last task is eval only!
          if task == nr_tasks-1:
            eval_stats = self.do_task(task, train=False)  ;
            self.save2csv(f"eval_resultsT{task}.csv", np.array(eval_stats)) ;
          # otherwise train normally
          else:
            train_stats = self.do_task(task, train=True) ;
            self.save2csv(f"train_resultsT{task}.csv", np.array(train_stats)) ;
          # Keras expects HDF5 extension for save_weights
          self.learner.save(os.path.join(self.results_folder, 'ckpt', self.config.exp_id + str(task)+".weights.h5")) ;



    def parse_args(self):
        parser = ArgumentParser() ;

        parser.add_argument('--debug_port',                  type=int, default=11000        ,          help='port for TCP debug connections')
        parser.add_argument('--start_task',                  type=int, default=0        ,          help='task > 0 start and load')
        parser.add_argument('--eval_start_task',             type=int, default=0        ,          help='use tasks >= eval_start_task for evaluation')
        parser.add_argument('--seed',     type=int, default=42,                          help='The random seed for the experiment run.')
        parser.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated results') ;
        parser.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')
        

        parser.add_argument('--debug',                   type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output') ;

        parser.add_argument('--training_duration_task_0',        type=int,  default=-1, help='Defines the duration of task 0, -1 means as the others.')
        parser.add_argument('--training_duration',        type=int,  default=10000, help='Defines the number of iterations á training_duration_unit.')
        parser.add_argument('--evaluation_duration',      type=int,  default=10,    help='Defines the number of iterations á evaluation_duration_unit.')
        parser.add_argument('--training_duration_unit',   type=str, default='timesteps', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an training iteration.')
        parser.add_argument('--evaluation_duration_unit', type=str, default='episodes',  choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an evaluation iteration.')
         

        parser.add_argument('--env',  type=str, default='po', help='Sets the RL env to use.')

        parser.add_argument('--step_duration_nsec',     type=float, default=5e+8,                               help='frequency in simulation time is 1/step_duration_nsec')

        parser.add_argument('--max_steps_per_episode',  type=int, default=10000, help='Sets the number of steps after which an episode gets terminated.')
        # obsolete
        parser.add_argument('--load_ckpt',  nargs='?', type=str, default='',                   help='Provide a path to a checkpoint file to initialize a model.')

        config, unparsed = parser.parse_known_args() ;

        return config, unparsed;






