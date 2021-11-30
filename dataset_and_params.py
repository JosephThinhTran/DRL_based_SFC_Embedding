# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:42:10 2021

@author: Onlyrich-Ryzen

List of datasets for training and testing
List of parameters

"""
def get_sfc_spec_file():
    return 'sfc_file.sfc'

def get_train_datasets():
    train_dataset_list = [
        {'net_topo' : "ibm_200000_slots_1_con.net",
            'traffic' : "reordered_traffic_200000_slots_1_con.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-06-04.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-04.tra"},
        \
        {'net_topo': "ibm_200000_slots_1_con_2021-September-21__16-06-36.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-36.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-06-10.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-06-10.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-08-42.net",
            'traffic': "reordered_traffic_200000_slots_1_con_2021-September-21__16-08-42.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-September-21__16-09-30.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-September-21__16-09-30.tra"},
            \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-24-34.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-24-34.tra"},
            \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-24-55.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-24-55.tra"},
            \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-25-28.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-25-28.tra"},
            \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-25-55.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-25-55.tra"},
            \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-27-07.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-27-07.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-06__00-27-36.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-06__00-27-36.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-35-32.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-35-32.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-36-24.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-36-24.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-37-00.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-37-00.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-37-33.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-37-33.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-38-11.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-38-11.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-16__22-38-48.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-16__22-38-48.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-22-26.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-22-26.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-23-12.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-23-12.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-23-23.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-23-23.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-23-33.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-23-33.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-23-47.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-23-47.tra"},
        \
        {'net_topo' : "ibm_200000_slots_1_con_2021-November-17__11-24-06.net",
            'traffic' : "reordered_traffic_200000_slots_1_con_2021-November-17__11-24-06.tra"}
        ]
    return train_dataset_list

def get_test_datasets(choice):
    if choice == '15k':
        test_dataset_list = [
            {"net_topo": "ibm_15000_slots_1_con.net",
             "traffic": "reordered_traffic_15000_slots_1_con.tra"}
            ]
    else:
        test_dataset_list = [
            {"net_topo": "ibm_100000_slots_1_con.net",
             "traffic": "reordered_traffic_100000_slots_1_con.tra"}
        ]
    return test_dataset_list

def get_train_test_params(max_epochs, sfc_spec_file, 
                    train_dataset_list, train_dir, 
                    test_dataset_list, test_dir, 
                    fc_hidden_layers, state_dim, 
                    actor_ouput_dim, args):
    OPERATION_MODE = {'train_mode': 1, 'test_mode': 0}
    NET_ARCH = {'shared_net': 1, 'shared_net_w_RNN':2, 'separated_net':3}
    ADV_STYLE = {'n_step_return': 1, 'gae': 2}# n-step return or GAE

    params = {'epochs': max_epochs,
            'n_workers': args.n_workers,
            'sfc_spec': sfc_spec_file,
            'datasets': train_dataset_list,
            'traffic_type': args.traffic_type,
            'data_folder': args.data_folder,
            'model_dir': args.model_dir,
            'train_dir': train_dir,
            'test_dir': test_dir,
            'train_freq': args.train_freq,
            'adv_style': ADV_STYLE[args.adv_style],
            'hidden_layers': fc_hidden_layers,
            'input_dims': state_dim,
            'actor_dims': actor_ouput_dim,
            'learning_rate': args.lr,
            'gamma': args.gamma,
            'tau': args.tau,
            'critic_factor': args.critic_factor,
            'actor_factor': args.actor_factor,
            'entropy_factor': args.entropy_factor,
            'entropy_decay_val': args.entropy_decay_val,
            'entropy_decay_freq': args.entropy_decay_freq,
            'entropy_min': args.entropy_min,
            'betas': args.betas,
            'big_rwd': args.big_rwd,
            'N_steps': args.n_steps,
            'max_moves': args.max_moves,
            'resource_scaler': args.rsc_scaler,
            'is_binary_state': args.is_binary_state,
            'state_noise_scale': args.state_noise_scale,
            'opt_lr': args.opt_lr,
            'opt_epsilon': args.opt_epsilon,
            'opt_weight_decay': args.opt_weight_decay,
            'opt_alpha': args.opt_alpha,
            'opt_momentum': args.opt_momentum,
            'opt_centered': args.opt_centered,
            'en_log': args.en_log
            }
    
    return params
