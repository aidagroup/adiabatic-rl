"""
$-task pushing objects scenario
"""
import matplotlib

OBJ_ID_LOOKUP = {"red":0, "green":1, "blue":2, "yellow":3, "pink":4, "cyan":5, "other_red":6}
from gazebo_sim.simulation.Environment import *

class PushingObjectsWrapper(EnvironmentWrapper):
    def __init__(self, step_duration_nsec=100 * 1000 * 1000) -> None:
        ## Possible Tasks (With either 15 or -15 rotation as starting point)
        tasks = {}
        tasks['red'] = Task('red',[Task.Transform(position=[0.4,0.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,0.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=20)
        tasks['other_red'] = Task('other_red',[Task.Transform(position=[0.4,0.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,0.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=0)
        tasks['green'] = Task('green',[Task.Transform(position=[0.4,4.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,4.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=20)
        tasks['blue'] = Task('blue',[Task.Transform(position=[0.4,8.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,8.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=0)
        tasks['yellow'] = Task('yellow',[Task.Transform(position=[0.4,-4.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,-4.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=0)
        tasks['pink'] = Task('pink',[Task.Transform(position=[0.4,-8.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,-8.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=0)
        tasks['cyan'] = Task('cyan',[Task.Transform(position=[0.4,-12.0,0.05],euler_rotation=[0.0,0.0,15.0]),Task.Transform(position=[0.4,-12.0,0.05],euler_rotation=[0.0,0.0,-15.0])],mass=20)
        
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

        env_config = EnvironmentConfig(observation_shape=[20,20,3],tasks=tasks,actions=actions,robot_name='3pi_front_cam_robot',vehicle_prefix='/vehicle',world_name='/world/pushing_objects_world',camera_topic='/vehicle/camera')
        super().__init__(env_config,step_duration_nsec)
        self.set_manager(env_config=env_config)

        self.info = {
           'input_dims': self.observation_shape,
           'number_actions': self.nr_actions,
           'closeup_positions': [],
           'terminate_cond': 'unassigned',
        }

        # for reward computation
        c2 = self.observation_shape[0]
        self.x = np.arange(0,c2).reshape(1,c2) * np.ones([c2,1]) ;
        self.y = np.ones([1,c2]) * np.arange(0,c2).reshape(c2,1) ;
    
    def get_current_status(self):
        obj_name = self.info['object'][0]
        return (self.info['object'][1], OBJ_ID_LOOKUP[obj_name],self.info['terminate_cond'])

    def switch(self, task_index: int) -> None:
        super().switch(task_index)
        self.task_id = self.task_list[task_index]
        print("switching to task", self.task_id, task_index)

        self.manager.perform_switch(task_index)
        self.reset()

    def reset(self):
        super().reset()
        current_task = self.tasks[self.task_id]
        self.current_name = current_task.task_name
        self.current_mass = current_task.get('mass')
        self.starting_transform = current_task.get_random_start()
        self.info['object'] = (self.current_name,self.current_mass)
        self.info['closeup_positions'] = []

        self.manager.trigger_pause(False)
        # stop and wait until robot has stopped
        self.manager.gz_perform_action_stop()
        response = self.get_observation(nsec=200.*1000*1000)
        # re-place robot
        self.manager.perform_reset(self.starting_transform.position,self.starting_transform.orientation) ;
        response = self.get_observation(nsec=500*1000*1000) ;

        self.manager.trigger_pause(True)

        state = self.manager.convert_image_msg(response)
        state = state[::4,::4,:]

        _, _, _ = self.compute_reward(state,self.manager.get_position(),-1)

        return (state, self.info)

    def step(self, action_index: int):
        super().step(action_index)

        self.manager.trigger_pause(False)

        self.perform_action(action_index=action_index)
        self.step_count += 1

        response = self.get_observation(nsec=self.step_duration_nsec)

        self.manager.trigger_pause(True)

        # make a decent np array out of the received observation
        state = self.manager.convert_image_msg(response) ;
        # downsample for speed
        state=state[::4,::4,:] ;
        pos = self.manager.get_position()
        reward, terminated, truncated = self.compute_reward(state, pos, action_index) ;
        distance_to_cube = f'{(0.9 - pos[0]):.3f}'
        tmp = f'--- {self.get_current_status()[2]}' if terminated or truncated else ''
        print(f'Step: {self.step_count:<4}, Action: {self.action_entries[action_index].name:<8}, Cube: {self.current_name:<8}, Mass: {self.current_mass:<3}, Dist: {distance_to_cube:<6}, Reward: {reward:<5} {tmp}')
        if (0.9 - pos[0]) <= 0.2:
            self.info['closeup_positions'].append([round(pos[0],2),round(pos[1],2)])

        return state,reward,terminated,truncated, self.info ;

    def compute_reward(self, state, position, action_index):
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
        cog_x = (self.x*colored_pixels).sum()  / csum if csum > 0 else 0. ;
 
        reward = 1. - math.fabs(cog_x - c) / c;

        if csum < 5:  # LOST SIGHT OF OBJECT
            truncated = True
            self.info['terminate_cond'] = "COND: LOST"
            return 0.0, truncated, terminated ; # TODO 0 better than -1 to avoid high td errors for irrelevant samples...
        if self.step_count >= self.max_steps_per_episode:
            terminated = True
            self.info['terminate_cond'] = "COND: MAX STEPS REACHED"
            return reward, truncated, terminated ;

        #if position[0] >= 0.85 and math.fabs(position[1]-cube_pos) < 0.15 and reward > 0.5: # x coord of the robot (closeness to the cube) 0.9 ) collision with cube
        if position[0] >= 0.85 and reward > 0.6: # x coord of the robot (closeness to the cube) 0.9 ) collision with cube
            truncated = True
            if self.current_mass > 1: # we mean > 0 but safer this way
                reward = -10
            else:
                reward = 10
            self.info['terminate_cond'] = "COND: Pushed object"
            return reward, truncated, terminated ;
        #if position[0] >= 0.85 and math.fabs(position[1]-cube_pos) > 0.15:
        if position[0] >= 0.85 and reward < 0.6: # cube missed
          truncated=True;
          terminated=False;
          self.info['terminate_cond'] = "COND: drove past object"
          return reward,truncated,terminated ; 

        self.info['terminate_cond'] = "COND: Normal" ;
        modifier = 0.1 if action_index == 0 else 1.0 ; # punish stop action
        return reward * modifier, truncated, terminated ;
