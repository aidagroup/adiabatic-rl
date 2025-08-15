"""
$-task catching robots scenario
"""
from typing import Dict, List, Tuple
import matplotlib
import copy

from gazebo_sim.simulation.Environment import *

class CatchingRobotConfig(EnvironmentConfig):
    def __init__(self, observation_shape: Tuple[int, int, int], tasks: Dict[str, Task], actions: List[TwistAction], robot_name: str, vehicle_prefix: str, world_name: str, camera_topic: str, runner_action:TwistAction,runner_start_positions:Task, debug = False) -> None:
        super().__init__(observation_shape, tasks, actions, robot_name, vehicle_prefix, world_name, camera_topic, debug)

        ### RUNNER SPECIFIC
        self.runner_action = runner_action
        self.runner_start_positions = runner_start_positions

class CatchingRobotWrapper(EnvironmentWrapper):
    def __init__(self, step_duration_nsec=100 * 1000 * 1000) -> None:
        self.debug = False

        ## Possible Tasks (With either 15 or -15 rotation as starting point)
        tasks = {}
        tasks['red_cube'] = Task('red_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],parking=Task.Transform([-30,-1,-0.25],[0,0,0]))
        tasks['green_capsule'] = Task('green_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],parking=Task.Transform([-30,-2,-0.25],[0,0,0]))
        tasks['blue_sphere'] = Task('blue_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],parking=Task.Transform([-30,-3,-0.25],[0,0,0]))
        tasks['yellow_cylinder'] = Task('yellow_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],parking=Task.Transform([-30,-4,-0.25],[0,0,0]))
        
        ## Action space of the Robot
        raw_action_space = 4.*np.array([[0.,0.],[0.5,0.1],[0.1,0.5],[0.3,0.3]]) ;
        actions = []
        for e in raw_action_space:
          label = "" ;
          if e[0]+e[1] < 0.0001: label = "stop" ;
          elif e[0] == e[1]: label = "straight" ;
          elif e[0] >= e[1]: label = "right" ;
          elif e[0] <= e[1]: label = "left" ;
          actions.append(TwistAction(label, e))
        print("Nr actions is ", len(actions)) ;

        self.cardinal_directions = {
            'north' : Task.Transform([0.0,0.0,0.0],[0,0,0]),
            'east' : Task.Transform([0.0,0.0,0.0],[0,0,90]),
            'south' : Task.Transform([0.0,0.0,0.0],[0,0,180]),
            'west' : Task.Transform([0.0,0.0,0.0],[0,0,270])
        }
        ### Depenging on what png the ground_plane uses
        tiny_arena = [-2.5,2.5,-2.5,2.5]
        small_arena = [-5,5,-5,5]
        big_arena = [-10,10,-10,10]

        self.arena_bounds = tiny_arena # TODO: Argparse!
        self.runner_collision_thickness = 0.42 # TODO: Argparse!

        runner_start_positions = Task('runner_start',[Task.Transform([0.0,0.0,-0.25],[0.0,0.0,60.0]),Task.Transform([0.0,0.0,-0.25],[0.0,0.0,-60.0])])

        env_config = CatchingRobotConfig(observation_shape=[20,20,3],tasks=tasks,actions=actions,robot_name='catcher_robot',vehicle_prefix='/vehicle',world_name='/world/catching_robot_world',camera_topic='/vehicle/camera',runner_action=TwistAction('forward',[0.25,0.25]),runner_start_positions=runner_start_positions,debug=self.debug)

        super().__init__(env_config,step_duration_nsec)

        self.set_manager(env_config=env_config)

        self.info = {
           'input_dims': self.observation_shape,
           'number_actions': self.nr_actions,
           'arena_bounds': self.arena_bounds,
           'terminate_cond': 'unassigned',
        }

        # for reward computation
        c2 = self.observation_shape[0]
        self.reward_comp = {}
        self.reward_comp['x'] = np.arange(0,c2).reshape(1,c2) * np.ones([c2,1]) ;
        self.reward_comp['y'] = np.ones([1,c2]) * np.arange(0,c2).reshape(c2,1) ;

    def set_manager(self, env_config:CatchingRobotConfig):
        self.manager = CatchingRobotManager(env_config)
    
    def get_current_status(self):
        # this will get printed into the results. If this ever is an issue just make it a number for each condition
        return (self.info['terminate_cond'],)

    def switch(self, task_index: int) -> None:
        self.task_id = self.task_list[task_index]
        print("Switching to task", self.task_id, task_index)

        self.manager.perform_switch(self.task_id)
        
        super().switch(task_index)

        self.reset()

    def reset(self):
        super().reset()
        current_task = self.tasks[self.task_id]
        self.starting_transform = current_task.get_random_start()

        self.manager.trigger_pause(False)
        # stop and wait until robots have stopped
        self.manager.gz_stop_runner()
        self.manager.gz_perform_action_stop()
        response = self.get_observation(nsec=200.*1000*1000)
        # re-place robot
        self.manager.perform_reset(self.starting_transform.position,self.starting_transform.orientation) ;
        response = self.get_observation(nsec=500*1000*1000) ;
        
        self.manager.gz_start_runner()

        self.manager.trigger_pause(True)

        state = self.manager.convert_image_msg(response)
        state = state[::4,::4,:]

        _, _, _ = self.compute_reward(state,-1,self.manager.get_position(),self.manager.get_runner_position())

        return (state, self.info)

    def step(self, action_index: int):
        super().step(action_index)

        self.manager.trigger_pause(False)
        self.runner_out_of_bounds(self.manager.get_runner_position())

        self.perform_action(action_index=action_index)
        self.step_count += 1

        response = self.get_observation(nsec=self.step_duration_nsec)


        self.manager.trigger_pause(True)

        # make a decent np array out of the received observation
        state = self.manager.convert_image_msg(response) ;
        # downsample for speed
        state=state[::4,::4,:] ;

        reward, terminated, truncated = self.compute_reward(state,action_index, self.manager.get_position(), self.manager.get_runner_position()) ;
        tmp = f'--- {self.get_current_status()[0]}' if terminated or truncated else ''
        print(f'[{self.step_count}] {self.action_entries[action_index].name:<9} @ pos[{self.manager.get_position()[0]:.3f},{self.manager.get_position()[1]:.3f}] -> runner @ pos[{self.manager.get_runner_position()[0]:.3f},{self.manager.get_runner_position()[1]:.3f}] = reward: {reward:.3f} {tmp}')

        return state,reward,terminated,truncated, self.info ;

    def runner_out_of_bounds(self, runner_position:Tuple[float,float,float]):
        if runner_position[1] < self.arena_bounds[0]: # West of the arena
            self.manager.rotate_runner(self.cardinal_directions['east'])
        elif runner_position[1] > self.arena_bounds[1]: # East of the arena
            self.manager.rotate_runner(self.cardinal_directions['west'])
        if runner_position[0] < self.arena_bounds[2]: # South of the arena
            self.manager.rotate_runner(self.cardinal_directions['north'])
        elif runner_position[0] > self.arena_bounds[3]: # North of the arena
            self.manager.rotate_runner(self.cardinal_directions['south'])

    def compute_reward(self, state, action_index,catcher_pos, runner_pos):
        super().compute_reward(state)
        truncated = False
        terminated = False
        c = (state.shape[0] // 2) ;
        hsv = matplotlib.colors.rgb_to_hsv(state) ;
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        colored_pixels = (s > 0.3).astype(np.int32) ; 
        csum = colored_pixels.sum() ;
        print(hsv.shape, "VMINMAX", "H", h.min(), h.max(), "S", s.min(), s.max(), "V", v.min(), v.max(), csum)
        cog_x = (self.reward_comp['x']*colored_pixels).sum()  / csum if csum > 0 else 0. ;
 
        reward = 1. - math.fabs(cog_x - c) / c;

        collider = self.runner_collision_thickness / 2.0

        if catcher_pos[0] <= runner_pos[0] + collider and catcher_pos[0] >= runner_pos[0] - collider and catcher_pos[1] <= runner_pos[1] + collider and catcher_pos[1] >= runner_pos[1] - collider:
            truncated = True
            reward = 10
            self.info['terminate_cond'] = 'Cought_Runner'
        elif catcher_pos[0] < self.arena_bounds[0] or catcher_pos[0] > self.arena_bounds[1] or catcher_pos[1] < self.arena_bounds[2] or catcher_pos[1] > self.arena_bounds[3]:
            truncated = True
            reward = -10
            self.info['terminate_cond'] = 'Left_Arena'
        elif csum < 1:  # LOST SIGHT OF OBJECT
            truncated = True
            self.info['terminate_cond'] = 'Lost_Runner'
            reward = -10
        elif self.step_count >= self.max_steps_per_episode:
            terminated = True
            self.info['terminate_cond'] = 'Max_Steps_Reached'
            reward = -10
        else:
            if self.debug: print("COND: Normal, REWARD: ",reward) ;
            self.info['terminate_cond'] = f"COND: Normal, REWARD: {reward}"
            modifier = 0.1 if action_index == 0 else 1.0 ;
            reward = reward*modifier
        
        return reward , truncated, terminated ;

class CatchingRobotManager(EnvironmentManager):
    def __init__(self,env_config:CatchingRobotConfig):
        self.init_node()
        self.mutex = Lock()

        self.env_config = env_config

        self.robot_name = self.env_config.robot_name
        self.vehicle_prefix = self.env_config.vehicle_prefix
        self.world_name = self.env_config.world_name

        self.runner_action = env_config.runner_action
        self.runner_start_positions = env_config.runner_start_positions
        self.runner_name = next(iter(self.env_config.tasks.values())).task_name

        self.step = 0
        self.last_obs_time = 0

        if self.subscribe(Image,f'{self.vehicle_prefix}/camera',self.gz_handle_observation_callback):
            print("Subscribed to Camera!")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info! for catching robot")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_another_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info! for running robot")

        self.gz_action = self.advertise(f'{self.vehicle_prefix}/motor',Twist)

        self.gz_runner_actions = {}
        for task in env_config.tasks.values():
            self.gz_runner_actions[task.task_name] = self.advertise(f'/{task.task_name}/motor',Twist)

        #self.scene_publisher = self.advertise(f'{self.world_name}/default/scene', Scene) # possibly .../default/scene

        self.wait_for_simulation()

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()
        self.set_pose_service = f'{self.world_name}/set_pose'
        self.pos_req = Pose()

    def get_runner_position(self):
        return self.runner_position
    
    def perform_reset(self, position, orientation):
        self.runner_position = self.runner_start_positions.get_random_start()
        self.set_entity_pose_request(self.runner_name,self.runner_position.position,self.runner_position.orientation)
        return super().perform_reset(position, orientation)
    
    def park_runner(self, task_id):
        parking_transform = self.env_config.tasks[task_id].get('parking')
        self.gz_stop_runner()
        self.set_entity_pose_request(self.runner_name,parking_transform.position,parking_transform.orientation)
        print(f"parked runner {self.runner_name} at {parking_transform.position}")
    
    def perform_switch(self,task_id:str):
        ### task_name has to coincide with the name of the robot in question
        if hasattr(self,'task_id'):
            self.park_runner(self.task_id)
        self.task_id = task_id
        self.runner_name = self.env_config.tasks[task_id].task_name
    
    def rotate_runner(self,transform:Task.Transform):
        tmp = copy.deepcopy(transform)
        angle = np.random.uniform(-60,60)
        tmp.add_rotation([0,0,int(angle)])
        self.set_entity_pose_request(self.runner_name,self.runner_position,tmp.orientation)
        print(f"redirected {self.runner_name} at {self.runner_position} by {tmp.orientation}")

    def gz_start_runner(self):
        self.gz_runner_actions[self.runner_name].publish(self.runner_action.return_instruction())
        if self.env_config.debug =="yes": print(f'runner_action published: ', self.runner_action.to_string())

    def gz_stop_runner(self):
        action = TwistAction('stop', [0.0, 0.0], 0.1)
        self.gz_runner_actions[self.runner_name].publish(action.return_instruction())
        print(f'Stopping {self.runner_name}!')

    def gz_handle_another_dynamic_pose_callback(self, msg):
        with self.mutex:
            for p in msg.pose:
                if p.name == self.runner_name:
                    self.runner_position = [p.position.x,p.position.y,p.position.z];
                    return;
            print(f"THERE WAS NO\033[92m {self.runner_name}\033[0m IN THE SIMULATION!")
