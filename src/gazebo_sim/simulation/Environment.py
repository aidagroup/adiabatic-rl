"""
Abstract Environment Class for any 3-pi Robot Scenario
"""
import time ;
import math ;
import argparse ;
import numpy as np ;

from abc import ABC, abstractmethod
from typing import * ;
from threading import Lock ;
from scipy.spatial.transform import Rotation

from gz.transport13 import Node ;
from gz.msgs10.empty_pb2 import Empty ;
from gz.msgs10.scene_pb2 import Scene ;
from gz.msgs10.image_pb2 import Image ;
from gz.msgs10.twist_pb2 import Twist ;
from gz.msgs10.world_control_pb2 import WorldControl ;
from gz.msgs10.boolean_pb2 import Boolean ;
from gz.msgs10.pose_pb2 import Pose ;
from gz.msgs10.pose_v_pb2 import Pose_V ;

class TwistAction():
    def __init__(self, name, wheel_speeds, separation=.1):
        self.name = name
        self.wheel_speeds = wheel_speeds

        self.action = Twist()
        if wheel_speeds[0]+wheel_speeds[0] != 0:
            self.action.linear.x = (wheel_speeds[0] + wheel_speeds[1]) / 2
            self.action.angular.z = (wheel_speeds[0] - wheel_speeds[1]) / separation

    def return_instruction(self):
        return self.action

    def to_string(self):
        return f'{self.name}: {self.wheel_speeds}'

    def __str__(self):
      return f"WheelSpeeds={self.wheel_speeds[0]}/{self.wheel_speeds[1]}" ;

class Task():
    class Transform():
        def __init__(self,position:Tuple[float,float,float],euler_rotation:Tuple[int,int,int]):
            self.position = position
            orientation = Rotation.from_euler('xyz',euler_rotation,degrees=True).as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;
        def add_rotation(self, euler_rotation:Tuple[int,int,int]):
            rot_modifier = Rotation.from_euler('xyz',euler_rotation,degrees=True)
            current_orientation = Rotation.from_quat(self.orientation)
            orientation = current_orientation * rot_modifier
            orientation = orientation.as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;

    def __init__(self, task_name:str, start_points:List[Transform], **kwargs) -> None:
        self.task_name = task_name
        self.starting_points = start_points
        self.settings = kwargs
    def get_random_start(self)->Transform:
        indices = np.arange(len(self.starting_points))
        random_index = np.random.choice(indices)
        return self.starting_points[random_index]
    def get(self,key:str):
        return self.settings.get(key,None)

class EnvironmentConfig():
    def __init__(self,observation_shape:Tuple[int,int,int],tasks:Dict[str,Task],actions:List[TwistAction],robot_name:str,vehicle_prefix:str,world_name:str,camera_topic:str, debug = False) -> None:
        self.debug = debug

        ### ENVIRONMENT WRAPPER
        self.observation_shape = observation_shape
        self.tasks = tasks
        self.actions = actions
        
        ### ENVIRONMENT MANAGER
        self.robot_name = robot_name
        self.vehicle_prefix = vehicle_prefix
        self.world_name = world_name
        self.camera_topic = camera_topic

class EnvironmentWrapper(ABC):
    def __init__(self,env_config:EnvironmentConfig,step_duration_nsec=100*1000*1000)->None:
        assert env_config != None, "The EnvironmentWrapper needs environment configurations!"
        self.config = self.parse_args()
        env_config.debug = self.config.debug
        self.step_duration_nsec = step_duration_nsec

        self.training_duration = self.config.training_duration
        self.evaluation_duration = self.config.evaluation_duration
        self.max_steps_per_episode = self.config.max_steps_per_episode
        self.task_list = self.config.task_list

        self.observation_shape = env_config.observation_shape
        self.tasks = env_config.tasks
        self.action_entries = env_config.actions
        self.nr_actions = len(self.action_entries)

        self.step_count = 0
        self.task_index = 0

    #@abstractmethod
    def set_manager(self,env_config:EnvironmentConfig):
        self.manager = EnvironmentManager(env_config)

    def get_nr_of_tasks(self):
        return len(self.task_list)

    def get_input_dims(self):
        return self.observation_shape

    @abstractmethod
    def get_current_status(self):
        return (0.,)

    def perform_action(self, action_index:int)->None:
        """ high level action execution """
        if self.config.debug=="yes": print(f'action request at tick [{self.step_count}]')

        action = self.action_entries[action_index] # select action to publish to GZ

        if self.config.debug=="yes": 
            print(f'action i={action_index} ({action.wheel_speeds[0]:2.2f}/{action.wheel_speeds[1]:2.2f}) published at tick [{self.step_count}]')

        self.manager.gz_perform_action(action)

    def get_observation(self, nsec=None):
        if nsec is None:
            nsec = self.step_duration_nsec
        t0 = self.manager.get_last_obs_time()
        #print(s)
        i = 0 ;
        last = t0 ;
        while ((self.manager.get_last_obs_time() - t0) < nsec):
            #time.sleep(0.001) ;
            if self.manager.get_last_obs_time() != last: 
              last = self.manager.get_last_obs_time() ;
              i += 1 ;
              #print("frame while waiting: i, t0,last,delta,nsec = ", i, t0, last, last-t0, nsec) ;

            pass ;
        response = self.manager.get_data()
        return response ;

    @abstractmethod
    def switch(self,task_index:int)->None:
        pass

    @abstractmethod
    def reset(self):
        self.step_count = 0

    @abstractmethod
    def step(self, action_index:int):
        pass

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.', exit_on_error=False)

        # ------------------------------------ LEARNER
        default_group = parser.add_argument_group('default')
        parser.add_argument('--debug', type=str,default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        default_group.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated result.') ;
        default_group.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')
        default_group.add_argument('--task_list', type=str, required=True, nargs="*", help='tasks to execute')

        duration_group = parser.add_argument_group('duration')
        duration_group.add_argument('--training_duration',        type=int, default=200, help='Defines the number of iterations á training_duration_unit.')
        duration_group.add_argument('--evaluation_duration',      type=int, default=5,    help='Defines the number of iterations á evaluation_duration_unit.')
        duration_group.add_argument('--training_duration_unit',   type=str, default='episodes', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) t') ;
        duration_group.add_argument('--evaluation_duration_unit', type=str, default='episodes',  choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level)') ;
        duration_group.add_argument('--max_steps_per_episode',    type=int, default=40, help='Sets the number of steps after which an episode gets terminated.')
        duration_group.add_argument('--start_task',               type=int, default=0        ,          help='')
        
        cfg,unparsed = parser.parse_known_args() ; 
        return cfg ;

    def close(self):
        self.manager.destroy_node()

class EnvironmentManager(Node):
    def __init__(self,env_config):
        self.init_node()
        self.mutex = Lock()

        self.env_config = env_config

        self.robot_name = self.env_config.robot_name
        self.vehicle_prefix = self.env_config.vehicle_prefix
        self.world_name = self.env_config.world_name

        self.step = 0
        self.last_obs_time = 0

        if self.subscribe(Image,f'{self.vehicle_prefix}/camera',self.gz_handle_observation_callback):
            print("subscribed to Camera!")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info!")

        self.gz_action = self.advertise(f'{self.vehicle_prefix}/motor',Twist)

        self.wait_for_simulation()

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()
        self.set_pose_service = f'{self.world_name}/set_pose'
        self.pos_req = Pose()

    def init_node(self):
        super().__init__()

    def wait_for_simulation(self):
        response = self.request_scene()
        for m in response.model:
          print("Model in scene", m.name) ;

    def request_scene(self):
        result = False;
        start_time = time.time()
        while result is False:
            # Request the scene information
            result, response = self.request(f'{self.world_name}/scene/info', Empty(), Empty, Scene, 1)
            print(f'\rWaiting for simulator... {(time.time() - start_time):.2f} sec', end='')
            time.sleep(0.1)
        print('\nScene received!')
        return response

    def get_step(self):
        return self.step

    def get_data(self):
        return self.data

    def get_position(self):
        return self.position
    
    def get_orientation(self):
        return self.orientation
    
    def get_orientation_euler(self):
        return Rotation.from_quat(self.orientation)

    def get_last_obs_time(self):
        return self.last_obs_time
    
    def convert_image_msg(self, msg):
        return np.frombuffer(msg.data,dtype=np.uint8).astype(np.float32).reshape(msg.height,msg.width,3) / 255. ;

    def gz_handle_observation_callback(self,msg):
        with self.mutex:
            print("OBS") ;
            self.data = msg
            self.last_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec

    def gz_handle_dynamic_pose_callback(self,msg):
        with self.mutex:
            for p in msg.pose:
                if p.name == self.robot_name:
                    self.position = [p.position.x,p.position.y,p.position.z];
                    self.orientation = [p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]
                    return;
            print(f"THERE WAS NO\033[92m {self.robot_name}\033[0m IN THE SIMULATION!")

    def gz_perform_action(self, action:TwistAction):
        self.step += 1
        self.gz_action.publish(action.return_instruction())
        if self.env_config.debug=="yes": print(f'action published: ', action.to_string())

    def gz_perform_action_stop(self):
        action = TwistAction('stop', [0.0, 0.0], 0.1)
        self.gz_action.publish(action.return_instruction())

    def gz_publish_new_scene(self):
        self.scene

    def perform_switch(self, task_index:str):
        pass

    def perform_reset(self, position, orientation):
        self.position = position
        self.orientation = orientation
        self.set_entity_pose_request(self.robot_name,self.position,orientation)

    def set_entity_pose_request(self, name, position, orientation):
        self.pos_req.name = name
        self.pos_req.position.x = position[0]
        self.pos_req.position.y = position[1]
        self.pos_req.position.z = position[2]
        self.pos_req.orientation.x = orientation[0]
        self.pos_req.orientation.y = orientation[1]
        self.pos_req.orientation.z = orientation[2]
        self.pos_req.orientation.w = orientation[3]

        result = False ;
        while result == False:
          result, response = self.request(self.set_pose_service, self.pos_req, Pose, Boolean, 1) ;
          time.sleep(0.01)
          if self.env_config.debug=="yes": print(result, response.data)
          if response.data == True: break ;

    def trigger_pause(self, pause):
        self.res_req.pause = pause
        if self.env_config.debug=="yes": print(f'pause={pause} request !')

        result = False ;
        while result == False:
          result, response = self.request(self.world_control_service, self.res_req, WorldControl, Boolean, 1) ;
          time.sleep(0.01)
          if response.data == True: break ;
        if self.env_config.debug=="yes": print(f'pause={pause} request done!')
