""" Experimental experiment using the new DQN learner
    TODO: create own argparser for RLAgent and other global params
"""

import time
from datetime import datetime
import numpy as np
from argparse import ArgumentParser ;


from gazebo_sim.learner import ARLearner
from gazebo_sim.learner import DQNLearner
from gazebo_sim.learner import XLearner
from gazebo_sim.simulation import LineFollowingWrapper
from gazebo_sim.simulation import PushingObjectsWrapper
from gazebo_sim.simulation import CatchingRobotWrapper
from gazebo_sim.agent import RLAgent ;

if __name__== "__main__":

    print(f'Begin execution at: {datetime.now()}')

    p = ArgumentParser() ;
    p.add_argument("--obs_per_sec_sim_time", type=int, required=True, default=15) ;
    p.add_argument("--benchmark", type=str, required=True) ;
    p.add_argument("--algorithm", type=str, required=True) ;
    config, unparsed = p.parse_known_args() ;
    
    # instantiate environment
    # compute nsec delay between two observations
    # complicated by the fact that gazebo computes step durations only to msec precision
    # so a frame rate of 30 means that the delay between two frames is 33msec, but not 33.3333 msec
    # so we have to round down if we want to work with nsec delays     
    hz = 30. ; # we have to know this, definedi n the robot sdf file, camera sensor plugin
    nsec_per_frame = int(1000./hz) * 1000000. ;
    nsec = nsec_per_frame * (hz / config.obs_per_sec_sim_time) ;
    print("Assumed time per frame: ", nsec) ; 

    # TODO: None argument raus!
    if config.benchmark in [ "mono-lf", "colored-lf"]:
      env = LineFollowingWrapper(step_duration_nsec = nsec) ;
    elif config.benchmark == "po":
      env = PushingObjectsWrapper(step_duration_nsec = nsec) ;
    elif config.benchmark == "cr":
      env = CatchingRobotWrapper(step_duration_nsec=nsec) ;


    # instantiate learner
    if config.algorithm == "DQN":
      learner = DQNLearner(n_actions=len(env.action_entries),
                                 obs_space=env.get_input_dims(),
                                 config=None) ; 

    if config.algorithm == "AR":
      learner = ARLearner(n_actions=len(env.action_entries),
                                 obs_space=env.get_input_dims(),
                                 config=None) ; 

    if config.algorithm == "X":
      learner = XLearner(n_actions=len(env.action_entries),
                                 obs_space=env.get_input_dims(),
                                 config=None) ; 


    # instantiate agent
    agent = RLAgent(env, learner)

    # execute experiment
    agent.go()
    print(f'Finish execution at: {datetime.now()}')
    agent.mop_up(); # Terminates debug thread so program can exit

