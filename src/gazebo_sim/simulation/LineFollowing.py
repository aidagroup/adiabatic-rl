"""
$-task line following scenario
"""
from gazebo_sim.simulation.Environment import *

class LineFollowingWrapper(EnvironmentWrapper):
    def __init__(self, step_duration_nsec=100 * 1000 * 1000) -> None:
        tasks = {}
        tasks['circle_red'] = Task('circle_red',[Task.Transform(position=[-40.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),Task.Transform(position=[-40.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_green'] = Task('circle_green',[Task.Transform(position=[-20.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),Task.Transform(position=[-20.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_blue'] =  Task('circle_blue',[Task.Transform(position=[0.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),Task.Transform(position=[0.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_yellow'] =Task('circle_yellow',[Task.Transform(position=[20.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),Task.Transform(position=[20.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_white'] = Task('circle_white',[Task.Transform(position=[40.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),Task.Transform(position=[40.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])

        actions = []
        raw_action_space = 1.5*np.array([
            [0.05, 0.35], [0.15, 0.25], 
            [0.2, 0.2], 
            [0.25, 0.15], [0.35, 0.05]
            ]) ;
        
        for wheel_speeds in raw_action_space:
            diff = np.diff(wheel_speeds)
            if diff == 0: actions.append(TwistAction('forward', wheel_speeds))
            elif diff < 0: actions.append(TwistAction('right', wheel_speeds))
            elif diff > 0: actions.append(TwistAction('left', wheel_speeds))

        # observation space
        channels = 3 ;
        observation_shape = [3*2, 50, channels]

        env_config = EnvironmentConfig(observation_shape=observation_shape,tasks=tasks,actions=actions,robot_name='3pi_robot',vehicle_prefix='/vehicle',world_name='/world/race_tracks_world',camera_topic='/vehicle/camera')

        super().__init__(env_config,step_duration_nsec)
        self.set_manager(env_config=env_config)

        self.info = {
            "sequence_length":3, 
            "input_dims": self.observation_shape,
            "number_actions": self.nr_actions
        }

    # returns a tuple of floats for logging purposes
    def get_current_status(self):
        return (self.info['track'],) ;

    def switch(self, task_index):
        super().switch(task_index)
        self.task_id = self.task_list[task_index]
        print("switching to task", self.task_id, task_index)
        self.manager.perform_switch(task_index)
        self.reset()

    def reset(self):
        super().reset()
       
        current_task = self.tasks[self.task_id]
        self.current_name = current_task.task_name
        self.starting_transform = current_task.get_random_start()
        self.info['track'] = self.current_name
        
        self.manager.trigger_pause(False) # resume sim
        
        self.manager.gz_perform_action_stop()  # send stop action
        self.get_observation(nsec=500*1000*1000.)  # extend get_obs spinning loop to 0.5s (5e8) or 1s (1e9)

        self.manager.perform_reset(self.starting_transform.position,self.starting_transform.orientation)

        response = self.get_observation(nsec=(500. * 1000. * 1000.)) # wait again
        
        self.manager.trigger_pause(True) # block sim
        state = self.manager.convert_image_msg(response)[1:3,::2,:] ;

        return (self.glue_images(state, self.step_count), self.info);

    def glue_images(self,img, step_count:int):
        if step_count == 0:
          self.img0 = self.img1 = img ;
        elif step_count == 1:
          self.img0 = self.img1 ;
          self.img1 = img ;
        ret = np.concatenate([self.img0,self.img1,img]) ;
        self.img0 = self.img1 ; self.img1 = img ;
        return ret ;

    def step(self, action_index:int):
        super().step(action_index)

        self.manager.trigger_pause(False) # resume sim        

        self.perform_action(action_index=action_index) # perform action
        self.step_count +=1

        # get resulting observation
        response = self.get_observation(nsec=self.step_duration_nsec)

        self.manager.trigger_pause(True)    # pause sim

        state = self.manager.convert_image_msg(response)[1:3,::2,:]

        reward = self.compute_reward(state)
        terminated = reward < 0.
        truncated = self.step_count>=self.max_steps_per_episode

        obs = self.glue_images(state, self.step_count) # returns observation for learner model

        return obs, reward, terminated, truncated, self.info
    
    # --------------------------------> REWARD

    def compute_reward(self, img):
        gray_img = img.mean(axis=2) ;
        c = img.shape[1] ;
        black_mask = (gray_img[0,:] < 0.1) ;
        if black_mask.astype(np.int32).sum() < 1: # no black pixels visible
          return -1. ;

        x_indices = np.linspace(0.,c,c) ;
        black_indices = x_indices[black_mask] ;
        left_line_index = np.min(black_indices) ;
        
        ret = 1.- math.fabs(left_line_index-c//2) / (c//2) ;
        return ret ;



