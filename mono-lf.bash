#!/bin/bash
PROCESSES=(
    "gz.*sim"
    "colored_line_following.sdf"
    "gazebo_simulator"
    "ExperimentLF.py"
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
SRC_PATH="/home/gepperth/research/programming/python"
PROJECT_DIR="new-icrl"
echo ARGS $#
if [ "$#" == "2" ] ; then
SRC_PATH=${1} ;
PROJECT_DIR=${2}
fi 
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
# *-------------------------------------------------------

# PYTHONPATH - PYTHONPATh - PYTHONPATH --------------------------------
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
sim_options=" -r -s --headless-rendering --render-engine ogre2 "
#sim_options=" -r --render-engine ogre"
gz sim ${sim_options} "${ROOT_PATH}/line_following.sdf"  &

# debug
#ps -ef | grep gz

# start RL
echo  Executing gazebo_sim.ExperimentLF

# +++
python3 -m gazebo_sim.ExperimentDQN                          \
        --benchmark                                          mono-lf \
        --debug_port 11001 \                                                   \
        --exp_id                                            iccp0                                                     \
        --root_dir                                          "${ROOT_PATH}"                                                   \
        --start_task 0 \
        --exploration_start_task 1 \
        --eval_start_task 1 \
        --obs_per_sec_sim_time                              15 \
        --max_steps_per_episode                             100                                                           \
        --task_list                                         circle_3_l circle_3_l circle_3_r straight circle_3_l                                     \
        --training_duration_unit                            timesteps                                                       \
        --evaluation_duration_unit                          episodes                                                       \
        --training_duration                                 6000                                                           \
        --training_duration_task_0                          80                                                               \
        --evaluation_duration                               20                                                             \
        --train_batch_size                                  32                                                              \
        --gamma                                             0.8                                                             \
        --algorithm                                         AR                                                            \
        --dqn_dueling                                       no                                                             \
        --dqn_target_network                                no                                                             \
        --dqn_target_network_update_freq                    200                                                            \
        --dqn_adam_lr                                       1e-4                                                            \
        --qgmm_K                                            100                                                             \
        --qgmm_eps_0                                        0.011                                                           \
        --qgmm_eps_inf                                      0.01                                                            \
        --qgmm_lambda_sigma                                 0.                                                              \
        --qgmm_lambda_pi                                    0.                                                              \
        --qgmm_alpha                                        0.011                                                           \
        --qgmm_gamma                                        0.90                                                            \
        --qgmm_regEps                                       0.1                                                             \
        --qgmm_lambda_W                                     1.0                                                             \
        --qgmm_lambda_b                                     0.0                                                             \
        --qgmm_reset_somSigma                               0.4                                                             \
        --qgmm_somSigma_sampling                            yes                                                             \
        --qgmm_log_protos_each_n                            1000                                                            \
        --qgmm_init_forward                                 no                                                              \
        --qgmm_load_ckpt                                    checkpoints/checkpoints/OFFLINE-dcgmm-1.ckpt \
        --exploration                                       eps-greedy                                                      \
        --initial_epsilon                                   1.0                                                             \
        --epsilon_delta                                     0.00013                                                            \
        --final_epsilon                                     0.001                                                            \
        --replay_buffer                                     default                                                         \
        --eps_replay_factor                                 0.5                                                             \
        --eps_reset                                 0.5                                                             \
        --capacity                                          4000                                                              \
        --per_alpha                                         0.6                                                             \
        --per_beta                                          0.6                                                             \
        --per_eps                                           1e-6                                                            \
        --per_delta_beta                                    0.00006        \
        --sequence_length                                   3                                                               \
        --seed                                              42                                                              \
        --debug                                             no                                                             \
        --verbose                                           yes \
        ; execute_state "ExperimentLF"
# ---

echo DONE

# kill zombies
execute_watchout

# debug
# ps -ef | grep gz

