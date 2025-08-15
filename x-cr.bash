#!/bin/bash

#
# PO benchmark, 3000 iter/task, 2 waske phases per task
#

PROCESSES=(
    "gz.*sim"
    "colored_line_following.sdf"
    "pushing_objects.sdf"
    "gazebo_simulator"
    "ExperimentPO.py"
    "ruby"
    "gz"
)



function print_message {
    echo ${3}
}

function print_info { print_message "BLUE"   "INFO" "${*}" ; }
function print_warn { print_message "YELLOW" "WARN" "${*}" ; }
function print_ok   { print_message "GREEN"  "OK"   "${*}" ; }
function print_err  { print_message "RED"    "ERR"  "${*}" ; }
function print_part { print_message "CYAN"   "PART" "${*}" ; }
function print_unk  { print_message "PURPLE" "UNK"  "${*}" ; }


function check_process {
    pgrep -f "${1}"
}



function eval_state {
    local state=$?

    if (( $state == 0 ))
        then print_ok "success ${1}"
        else print_err "failed ${1}"
    fi

    return $state
}


function kill_process {
    pkill -9 -f "${1}"
}


function execute_check {
    print_info "check process ${entry}"
    eval_state $(check_process "${entry}")
}

function execute_kill {
    print_info "try to kill ${entry}"
    eval_state $(kill_process "${entry}")
}

function execute_watchout {
    print_info "watchout for possible zombies"
    for entry in ${PROCESSES[@]}
    do
        execute_check &&
        execute_kill
    done
}

function execute_state {
    state=$?
    if (( $state == 0 ))
        then print_ok "success (${1})"
        else print_err "failed (${1})"
    fi
    return $state
}


# *------------ COMMON DEFINITIONS ----------------------
SRC_PATH="/home/ydenker/git"
PROJECT_DIR="icrl"
echo ARGS $#
if [ "$#" == "2" ] ; then
SRC_PATH=${1} ;
PROJECT_DIR=${2}
fi 
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
# *-------------------------------------------------------

# PYTHONPATH - PYTHONPATH - PYTHONPATH --------------------------------
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/sccl/src
# *--------------------------------------------------------------------

# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ
export GZ_VERSION="8"
export GZ_DISTRO="harmonic"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="$(hostname)"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${ROOT_PATH}/models"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ


# kill zombies
execute_watchout

# debug
#ps -ef | grep gz


# start gazebo
sim_options=" -r -s --headless-rendering --render-engine ogre2"
#sim_options=" -r --render-engine ogre"
gz sim ${sim_options} "${ROOT_PATH}/catching_robot.sdf"  &

gz service -l

# debug
#ps -ef | grep gz

# start RL
echo  Executing gazebo_sim.ExperimentDQN


# +++
python3 -m gazebo_sim.ExperimentDQN                                                                                          \
        --benchmark                                         po \
        --debug_port                                        11002                                                           \
        --seed                                              42                                                              \
        --exp_id                                            3xpo4                                                          \
        --root_dir                                          "${ROOT_PATH}"                                                  \
        --obs_per_sec_sim_time                              15                                                              \
        --training_duration                                 5000                                                             \
        --training_duration_task_0                          10000 \
        --evaluation_duration                               10                                                               \
        --training_duration_unit                            timesteps                                                        \
        --evaluation_duration_unit                          episodes                                                        \
        --max_steps_per_episode                             30                                                              \
        --task_list                                         red_cube red_cube blue_sphere green_capsule yellow_cylinder yellow_cylinder \
        --start_task                                        0                                                               \
        --eval_start_task                                   1                                                               \
        --exploration_start_task                            1 \
        --gamma                                             0.1                                                             \
        --train_batch_size                                  64                                                              \
        --algorithm                                         X                                                             \
        --ARL_PARAMS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --cycle_length                                      2500                                                          \
        --cycles_per_task                                   2 \
        --sleep_phase_train_steps                           2500                                                          \
        --draw_percentage                                   0.1                                                          \
        --DQN_PARAMS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --dqn_fc1_dims                                      128                                                             \
        --dqn_fc2_dims                                      64                                                              \
        --dqn_adam_lr                                       1e-4                                                            \
        --dqn_dueling                                       no                                                              \
        --dqn_target_network                                no                                                              \
        --dqn_target_network_update_freq                    200                                                             \
        --AR_PARAMS  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --start_task_ar                                     1                                                               \
        --ar_replay_ratio                                   3 \
        --qgmm_K                                            324                                                             \
        --qgmm_eps_0                                        0.011                                                           \
        --qgmm_eps_inf                                      0.01                                                            \
        --qgmm_lambda_sigma                                 0.                                                              \
        --qgmm_lambda_pi                                    0.                                                              \
        --qgmm_alpha                                        0.011                                                           \
        --qgmm_gamma                                        0.85                                                            \
        --qgmm_regEps                                       0.1                                                             \
        --qgmm_lambda_W                                     1.0                                                             \
        --qgmm_lambda_b                                     0.0                                                             \
        --qgmm_reset_somSigma                               1.0                                                             \
        --qgmm_somSigma_sampling                            yes                                                             \
        --qgmm_log_protos_each_n                            1000                                                            \
        --qgmm_init_forward                                 no                                                              \
        --EXPLORATION_PARAMS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --exploration                                       eps-greedy                                                      \
        --initial_epsilon                                   1.0                                                             \
        --final_epsilon                                     0.4                                                           \
        --epsilon_delta                                     0.0002                                                          \
        --eps_replay_factor                                 1.0                                                          \
        --BUFFER_PARAMS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --replay_buffer                                     with_td                                                         \
        --capacity                                          2500                                                            \
        --per_alpha                                         0.6                                                             \
        --per_beta                                          0.6                                                             \
        --per_eps                                           1e-6                                                            \
        --per_delta_beta                                    0.00005                                                         \
        ; execute_state "ExperimentPO"
# ---
#        --training_duration_task_0                          10000                                                             \

echo DONE

# kill zombies
execute_watchout

# debug
# ps -ef | grep gz

