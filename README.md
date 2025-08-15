# Intentional/Incremental/Inferential Continual Reinforcement Learning - ICRL

Reinforcement learning with Gazebo and Tensorflow, no other dependencies. 
Start PO benchmark by sourcing color-lf.bash, mono-lf.bash or po-4-task.bash! 
These bash files are a good example how to set up your own experiments.

## Quickstart (Ubuntu 24.04 Noble)

### 1) Install Gazebo Harmonic and Python bindings
```
sudo rm -f /etc/apt/sources.list.d/gazebo-stable.list
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt update
sudo apt install -y gz-harmonic python3-gz-transport13 python3-gz-msgs10
```

### 2) Python environment
Create a venv that can import the apt-installed `gz.*` Python modules:
```
sudo apt install -y python3.12-venv
python3.12 -m venv .venv312 --system-site-packages
source .venv312/bin/activate
pip install -U pip
pip install numpy scipy matplotlib imageio tensorflow pandas
```

### 3) Environment variables (Gazebo resources and Python path)
```
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export GZ_VERSION=8 GZ_DISTRO=harmonic GZ_IP=127.0.0.1 GZ_PARTITION=$(hostname)
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}$(pwd)/models
```

## Training (headless)

There are two supported ways to start training. The bash files are the canonical way. A minimal Python runner is also provided.

### Option A: Bash launchers (recommended)
These kill old processes, start Gazebo headless, and run the experiment with full CLI flags. Pass your root and project directories as args (example uses the current path):
```
bash colored-lf.bash "$(dirname "$(pwd)")" "$(basename "$(pwd)")"
# or
bash mono-lf.bash   "$(dirname "$(pwd)")" "$(basename "$(pwd)")"
bash po-4-task.bash "$(dirname "$(pwd)")" "$(basename "$(pwd)")"
```

### Option B: Minimal runner (colored LF)
This starts training using `gazebo_sim` classes directly (useful if your Python can’t import `ExperimentDQN`).
```
python scripts/run_colored_quick.py
```

Training artifacts are written to `results/<exp_id>/`:
- Metrics: `*_train_resultsT*.csv`, `*_eval_resultsT*.csv`
- Checkpoints: `results/<exp_id>/ckpt/<exp_id><task>.weights.h5`

## Evaluation / Inference (no training)

Assuming you trained colored-LF and have `results/quick_colored/ckpt/quick_colored{0..4}.weights.h5`:
```
source .venv312/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export GZ_VERSION=8 GZ_DISTRO=harmonic GZ_IP=127.0.0.1 GZ_PARTITION=$(hostname)
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}$(pwd)/models
python - <<'PY'
import sys
sys.argv=[sys.argv[0],
  '--task_list','circle_red','circle_green','circle_blue','circle_yellow','circle_white',
  '--start_task','4',                     # load last checkpoint and eval final task
  '--training_duration','0','--training_duration_unit','timesteps',
  '--evaluation_duration','5','--evaluation_duration_unit','episodes',
  '--max_steps_per_episode','30',
  '--exp_id','quick_colored','--root_dir','$(pwd)']
from gazebo_sim.simulation.LineFollowing import LineFollowingWrapper
from gazebo_sim.learner.DQNLearner import DQNLearner
from gazebo_sim.agent.RLAgent import RLAgent
env = LineFollowingWrapper()
learner = DQNLearner(n_actions=len(env.action_entries), obs_space=env.get_input_dims(), config=None)
agent = RLAgent(env, learner)
agent.go()
PY
```

## GUI mode

To see the robot move in the Gazebo window:
1) Start Gazebo with GUI:
```
gz sim $(pwd)/colored_line_following.sdf
```
2) In another terminal, run the Evaluation/Inference snippet above (with the same env vars). The Python process will connect via transport to the running GUI sim.

## “See what the agent sees” (debug images)

You can toggle debug and view camera images in a browser:
```
source .venv312/bin/activate
python trigger.py 11000
# In the trigger console: press 'd' + Enter to enable debug image saving
```
Then open `image.html` in your browser to auto-refresh the saved images.

## Results & where to look
- Checkpoints: `results/<exp_id>/ckpt/*.weights.h5`
- Metrics per task/phase: `results/<exp_id>/*_{train,eval}_resultsT*.csv`
- Run config dump: `results/<exp_id>/args.txt`

## Philosophy
There is a rather loose coupling between an Agent, a Learner and an Environment, each modeled by a separate instance. Environment instances follow the OpenAI Gym 
protocol except that they have an additional switch(task) method. A learner is expected to have load, save, store_transition, before_experiment, before_task, after_task
 and learn methods.
The agent executes the entire experiment and takes care of evaluation and logging to text files.

## Tasks and babbling
Each CRL experiment is partitioned into tasks. These are assumed to be numbered consecutively, starting at 0. 
Task 0 can be used for motor babbling in experiments using DQNLearner and subclasses by manipulating the exploration_start_task parameter.
Each experiment can be re-started at a task boundary by loading 
saved model weights files. The Agent takes care of this using the start_task parameter.

## Important Python executable files
* src/gazebo_sim/ExperimentDQN.py: This file reads its own command line arguments, instantiates an Agent, an Environment and a Learner class.
  Then, execution is passed to agent.go() to start the experiment.
* trigger.py: Takes a single port argument (int) and connects to a running RLAgent instance via TCP/IP. You can send debug commands to this instance, read from terminal.
* condense.py: Each RLAgent instance will create log files for an experiment. These can be converted to a format that slurm-exp understands, to create nice pivot result tables

## Important Python classes
* gazebo_sim.agent.RLAgent: Simple Q-Learning Agent that takes care of a CRL experiment. Manages tasks, episodes, results logging, etc. Takes params from cmd line, see bash files or the parse_args() method 
* gazebo_sim.learner.DQNLearner, ARLearner, ...: each learner class should inherit from DQNLearner. Ideally, we will create a common learner superclass, but for now this will do.
* gazebo_sim.simulation.PushingObjectsWrapper, gazebo_sim.simulation.LinefollowingWrapper: 

## Important sdf files
* pushing_objects.sdf  Standard pushign objects world. Disjoint from other sdf files in the models repo
* line_following.sdf Line following mono version
* colored_line_following.sdf Color line following. Reuses models from mono line following in the models directory

## The models subdirectory
Here, all gazebo models are stored that are needed for PO, mono-LF  and colored-LF benchmarks. For own experiments, just set the GAZEBO_MODEL_DIR variable in your bash files appropriately 
if you define own models outside this dir.

## Bash file
Bash files for starting an experiment should be self-contained, export all necessary environment variables and contain the invocation of the experiment main file with all parameters.
The invocation should be enclosed in # +++ and # --- to be recognized by the slurm-exp tool.
For your own experiment, you can copy an existing bash file and just replace the part between # +++ and # ---. Of course, all necessary environment
variables will have to be set prior to this call in the bash file. All of the sections are annotated with comments so you should have an easy time
doing this. The existing bash files allow a root directory and project subdirectory to be set from the command line as ${1} and ${2}. 
The absolute path to the project is then ${1}/${2}. If no parameters are specified, entries from the bash files itself are used. Useful if you want
to test locally and deploy on a cluster.


## Debugging
A running RLAgent instance can be debugged via TCP/IP. Debugging is designed to work even if we are only connected via ssh by saving pictures and displaying debug info.
If you can create an SSH tunnel, pictures can be seen in real-time in your browser by opening image.html, which reloads certain image files periodically.

Note: Gazebo may print warnings about `vertical_fov`, `gravity`, or `kinematic` elements in SDF. These are expected with SDF 1.10 and do not affect training/evaluation.

## Results logging
RL agent takes a root_dir and an exp_id parameter. It creates a directory root_dir/results/exp_id/ and stores all eval logs there.
These are simple text files... Since logs are uniquely linked t oa particular exp_id, evaluation of many different runs is possible, e.g., on a cluster.
