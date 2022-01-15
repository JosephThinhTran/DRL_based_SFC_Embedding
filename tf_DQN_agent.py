from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import keras.losses
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import copy
import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import os
from collections import deque


#####################################
class TFDQNet:
    ''' Class for Tensorflow Q-Network'''

    def __init__(self, name, model_dir, fc_hid_layers, input_dims, output_dims,
                 gamma, learning_rate, using_gpu=None):
        '''
        Parameters
        ----------
        name : STRING
            NAME OF THE Q-NET
        model_dir : STRING
            FOLDER NAME STORING THE MODEL
        fc_hid_layers : np.array int32
            # OF NEURONS PER FULLY-CONNECTED HIDDEN LAYER
            # DEFAULT: 2 HIDDEN LAYERS
        input_dims : int32
            INPUT DIMENSION
        output_dims : int32
            OUTPUT DIMENSION
        gamma : float32
            DISCOUNT FACTOR
        learning_rate : float32
            LEARNING RATE

        Returns
        -------
        None.

        '''
        # init
        # super(DQNet, self).__init__()

        # load params to the class
        self.model = None
        self.name = name
        self.model_dir = model_dir
        abs_path = os.getcwd()
        full_path = os.path.join(abs_path, self.model_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        self.checkpoint_file = os.path.join(full_path, self.name + '.pt')

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.fc_hid_layers = fc_hid_layers

        self.setup_dqn_model()

        # # config neural net
        # self.fc1 = nn.Linear(self.input_dims, self.fc_hid_layers[0])
        # self.fc2 = nn.Linear(self.fc_hid_layers[0], self.fc_hid_layers[1])
        # if len(self.fc_hid_layers) == 3:
        #     self.fc3 = nn.Linear(self.fc_hid_layers[1], self.fc_hid_layers[-1])
        # self.qvals = nn.Linear(self.fc_hid_layers[-1], self.output_dims)
        #
        # # optimizer and loss function
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # self.loss_fn = torch.nn.MSELoss()

        # # Using GPU if available
        # if (using_gpu is not None) and torch.cuda.is_available():
        #     gpu_id = "cuda:0"
        #     self.device = gpu_id
        #     self.to(self.device)
        #     # print(f'Using GPU {torch.cuda.get_device_name(gpu_id)}')
        # else:
        #     self.device = 'cpu'
        #     self.to(self.device)
        #     # print('Using CPU')
    #########################################

    #########################################
    def setup_dqn_model(self):
        # define the model
        self.model = model = Sequential()
        model.add(Dense(self.fc_hid_layers[0], activation='relu', input_shape=(self.input_dims,)))
        model.add(Dense(self.fc_hid_layers[1], activation='relu'))
        model.add(Dense(self.output_dims))

        # compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.MeanSquaredError(), metrics=['accuracy'])

        # # config neural net
        # self.fc1 = nn.Linear(self.input_dims, self.fc_hid_layers[0])
        # self.fc2 = nn.Linear(self.fc_hid_layers[0], self.fc_hid_layers[1])
        # if len(self.fc_hid_layers) == 3:
        #     self.fc3 = nn.Linear(self.fc_hid_layers[1], self.fc_hid_layers[-1])
        # self.qvals = nn.Linear(self.fc_hid_layers[-1], self.output_dims)

        # optimizer and loss function
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # self.loss_fn = torch.nn.MSELoss()

        # # Using GPU if available
        # if (using_gpu is not None) and torch.cuda.is_available():
        #     gpu_id = "cuda:0"
        #     self.device = gpu_id
        #     self.to(self.device)
        #     # print(f'Using GPU {torch.cuda.get_device_name(gpu_id)}')
        # else:
        #     self.device = 'cpu'
        #     self.to(self.device)
        #     # print('Using CPU')
    #########################################

    #########################################
    def forward(self, state):
        """
         Calculate the Q-values
        """
        return self.model.predict([state])

        # x = self.fc1(state)
        # # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.selu(x)
        # x = self.fc2(x)
        # if len(self.fc_hid_layers) == 3:
        #     x = self.fc3(x)
        # # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.selu(x)
        # qvals = self.qvals(x)
        # return qvals

    #####################################
    def save_params(self):
        """ Save network parameters to file """
        print("Saving " + self.name + "'s network parameters to file")
        # torch.save(self.state_dict(), self.checkpoint_file)
        self.model.save_weights(self.checkpoint_file)
    #####################################

    # def load_params(self):
    #     ''' Load network parameters from file'''
    #     print("Loading " + self.name + "'s network parameters from file")
    #     # torch.load(self.state_dict(), self.checkpoint_file)
    #     self.load_state_dict(torch.load(self.checkpoint_file))
#####################################


#####################################
class TFDQAgent(object):
    ''' Wrapper for the Tensorflow DQNet agent'''

    def __init__(self, model_dir, fc_hid_layers,
                 input_dims, output_dims,
                 gamma, learning_rate,
                 epsilon, buffer_size,
                 batch_size, using_gpu=False):
        ''' Deep-Q RL Agent

        Parameters
        ----------
        name : STRING
            NAME OF THE Q-NET
        model_dir : STRING
            FOLDER NAME STORING THE MODEL
        fc_hid_layers : np.array int32
            # OF NEURONS PER FULLY-CONNECTED HIDDEN LAYER
            # DEFAULT: 2 HIDDEN LAYERS
        input_dims : int32
            INPUT DIMENSION
        output_dims : int32
            OUTPUT DIMENSION
        gamma : float32
            DISCOUNT FACTOR
        learning_rate : float32
            LEARNING RATE
        epsilon : float32
            EXPLORATION RATE FOR THE E-GREEDY
        buffer_size : int32
            SIZE OF THE REPLAY BUFFER
        batch_size : int32
            SIZE OF A MINI BATCH

        Returns
        -------
        None.
        '''
        # Using GPU if available
        # if using_gpu and torch.cuda.is_available():
        #     gpu_id = "cuda:0"
        #     self.device = gpu_id
        #     print(f'Using GPU {torch.cuda.get_device_name(gpu_id)}')
        # else:
        #     self.device = 'cpu'
        #     print('Using CPU')

        # network params
        self.model_dir = model_dir
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc_hid_layers = fc_hid_layers

        # training params
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.batch_size = batch_size

        # create neural networks
        self.q_net = TFDQNet(name='TFQNet', model_dir=self.model_dir,
                             fc_hid_layers=self.fc_hid_layers,
                             input_dims=self.input_dims, output_dims=self.output_dims,
                             gamma=self.gamma, learning_rate=self.learning_rate, using_gpu=using_gpu)
        self.target_q_net = TFDQNet(name='TargetTFQNet', model_dir=self.model_dir,
                                    fc_hid_layers=self.fc_hid_layers,
                                    input_dims=self.input_dims, output_dims=self.output_dims,
                                    gamma=self.gamma, learning_rate=self.learning_rate, using_gpu=using_gpu)

        # synchronize q_net.params and target_q_net.params
        self.hard_sync()

        # init the replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def act(self, state, neighbor_filter=None):
        """
        Return action with connected neighbor filter

        Parameters
        ----------
        state : tensor
            SYSTEM STATE
        neighbor_filter : np.array
            A FILTER OF CONNECTED NEIGHBORS OF THE CURRENT NODE

        Returns
        -------
        action
        """
        # state = state.to(self.device)
        # self.q_net.eval()
        # qval = self.q_net.forward(state)
        # qval_ = qval.detach().cpu().numpy()
        # qval_ = qval.data.numpy()
        qval_ = self.q_net.model.predict([state])
        if neighbor_filter is not None:
            qval_ += neighbor_filter
        action = np.argmax(qval_)
        return action

    def batch_update(self):
        """ Update the network parameters using loss minimization"""
        # random batch sampling
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # convert to tensors
        state1_batch = tf.concat([s1 for (s1, a, r, s2, d) in minibatch], axis=0)
        action_batch = tf.convert_to_tensor([a for (s1, a, r, s2, d) in minibatch])
        reward_batch = tf.convert_to_tensor([r for (s1, a, r, s2, d) in minibatch], dtype=np.float32)
        state2_batch = tf.concat([s2 for (s1, a, r, s2, d) in minibatch], axis=0)
        done_batch = tf.convert_to_tensor([d for (s1, a, r, s2, d) in minibatch], dtype=np.float32)
        # state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
        # action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
        # reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
        # state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
        # done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

        # transfer tensors to GPU
        # state1_batch = state1_batch.to(self.device)
        # action_batch = action_batch.to(self.device)
        # reward_batch = reward_batch.to(self.device)
        # state2_batch = state2_batch.to(self.device)
        # done_batch = done_batch.to(self.device)

        # Update network parameters
        Q1 = self.q_net.model.predict(state1_batch)
        Q2 = self.target_q_net.model.predict(state2_batch)
        # with torch.no_grad():
        #    Q2 = self.target_q_net.forward(state2_batch)

        # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        Y = reward_batch + self.gamma * ((1 - done_batch) * tf.reduce_max(Q2, axis=1)[0])
        # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        X = tf.convert_to_tensor([Q1[i][action_batch[i]] for i in range(len(Q1))])
        loss_val = tf.keras.losses.mean_squared_error(X, Y)
        # loss_val = self.q_net.loss_fn(X, Y.detach())

        # update network params
        self.q_net.model.train_on_batch(state1_batch, Y)
        # self.q_net.optimizer.zero_grad()
        # loss_val.backward()
        # self.q_net.optimizer.step()

        # return loss_val.item()

        return loss_val

    #####################################
    def hard_sync(self):
        ''' Synchronize params of source_q_net and target_q_net '''
        # self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.model.set_weights(self.q_net.model.get_weights())
    #####################################

    #####################################
    def save_checkpoint(self):
        """ Save the models' parameters """
        self.q_net.save_params()
        self.target_q_net.save_params()

    def load_checkpoint(self, is_train=False):
        """
            Load neural networks' parameters from the checkpoint files
            Parameters:
                is_train : bool
                    TRAINING[True]/INFERENCE[False] MODE

        """
        self.q_net.load_params()
        self.target_q_net.load_params()

        # set training/inference mode
        if is_train:
            self.q_net.train()
            self.target_q_net.train()
        else:
            self.q_net.eval()
            self.target_q_net.eval()
#####################################


#####################################
class TFEDFEnv(py_environment.PyEnvironment):

    def __init__(self, graph):
        self._graph = copy.deepcopy(graph)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32, minimum=0, maximum=2 * self._graph.number_of_nodes(),
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        print(action)
        if action == -1:
            self._episode_ended = True
        elif action >= 2*self._graph.number_of_nodes():
            raise ValueError('`action` should be 0 to number of nodes-1.')
        else:
            self._state += 1

        if self._episode_ended or self._state >= 2:
            reward = self._state
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

