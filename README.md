# DRL_based_SFC_Embedding
 DRL-based algorithms for solving SFC Embedding Problems

# Implemented Algorithms
 ## A2C-based SFC embedding
    - Apply Advantage Actor-Critic DRL algorithm to solve the SFC embedding problem
	- Run "AC_SFC_Main_01.05.py" for training/testing the A2C RL-agent
    - Usage: AC_SFC_Main_01.05.py [-h] [--net_arch NET_ARCH] [--epochs EPOCHS] [--n_workers N_WORKERS] [--sfc_spec SFC_SPEC]                                             [--data_folder DATA_FOLDER] [--model_dir MODEL_DIR] [--model_name MODEL_NAME] [--train_freq TRAIN_FREQ]                                [--adv_style ADV_STYLE] [--lr LR] [--gamma GAMMA] [--tau TAU] [--critic_factor CRITIC_FACTOR]                                          [--actor_factor ACTOR_FACTOR] [--entropy_factor ENTROPY_FACTOR] [--entropy_decay_val ENTROPY_DECAY_VAL]                                [--entropy_decay_freq ENTROPY_DECAY_FREQ] [--entropy_min ENTROPY_MIN] [--betas BETAS [BETAS ...]]                                      [--big_rwd BIG_RWD] [--n_steps N_STEPS] [--max_moves MAX_MOVES] [--rsc_scaler RSC_SCALER]                                              [--is_binary_state IS_BINARY_STATE] [--state_noise_scale STATE_NOISE_SCALE] [--opt_lr OPT_LR]                                          [--opt_epsilon OPT_EPSILON] [--opt_weight_decay OPT_WEIGHT_DECAY] [--opt_alpha OPT_ALPHA]                                              [--opt_momentum OPT_MOMENTUM] [--opt_centered OPT_CENTERED] [--test_size TEST_SIZE]                                                    [--traffic_type TRAFFIC_TYPE] [--en_log EN_LOG]                                                                                        mode 

 ## DQN-based SFC embedding
    - Apply DQN RL algorithm to solve the SFC embedding problem
	- Run "DQ_SFC_Main_01.08.py" for training the DQN RL-agent
	- Run "test_model_random_req_order_v01.08.py" for testing the DQN RL-agent

# Dataset
 - Using the Dataset created by Huy Duong in the Ciena SOF-WP1 Project

 - Notes
    - The "reordered_traffic_500000_slots_1_con.tra" file cannot be committed to GitHub due to larger than 100 MB of size. This file will be uploaded to another cloud repo.
    
# TODO
 - Re-organize each algorithm's source code into a separated folder


    

 


