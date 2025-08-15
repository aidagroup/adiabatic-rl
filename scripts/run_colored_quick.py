import sys

# Configure command-line flags expected by EnvironmentWrapper, RLAgent, and DQNLearner
sys.argv = [
    sys.argv[0],
    # EnvironmentWrapper config
    '--task_list', 'circle_red', 'circle_green', 'circle_blue', 'circle_yellow', 'circle_white',
    '--training_duration_unit', 'timesteps',
    '--evaluation_duration_unit', 'episodes',
    '--training_duration', '200',
    '--evaluation_duration', '2',
    '--max_steps_per_episode', '30',
    '--start_task', '0',
    # RLAgent config
    '--exp_id', 'quick_colored',
    '--root_dir', '/home/danya/cvpr2025',
    '--debug', 'no',
    # Learner config
    '--train_batch_size', '32',
    '--gamma', '0.8',
    '--dqn_dueling', 'no',
    '--dqn_target_network', 'no',
    '--dqn_adam_lr', '0.001',
]

from gazebo_sim.simulation.LineFollowing import LineFollowingWrapper
from gazebo_sim.learner.DQNLearner import DQNLearner
from gazebo_sim.agent.RLAgent import RLAgent


def main() -> None:
    env = LineFollowingWrapper()
    learner = DQNLearner(n_actions=len(env.action_entries), obs_space=env.get_input_dims(), config=None)
    agent = RLAgent(env, learner)
    agent.go()


if __name__ == '__main__':
    main()



