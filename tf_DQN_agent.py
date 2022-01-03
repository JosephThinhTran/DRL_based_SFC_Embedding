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
        self._model = None
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
        #
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
        self._model = model = Sequential()
        model.add(Dense(self.fc_hid_layers[0], activation='relu', input_shape=(self.input_dims,)))
        model.add(Dense(self.fc_hid_layers[1], activation='relu'))
        model.add(Dense(self.output_dims))

        # compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.MeanSquaredError, metrics=['accuracy'])

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
        ''' Calculate the Q-values'''
        return self._model.predict([state])

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

    # def save_params(self):
    #     ''' Save network parameters to file'''
    #     print("Saving " + self.name + "'s network parameters to file")
    #     torch.save(self.state_dict(), self.checkpoint_file)
    #
    # def load_params(self):
    #     ''' Load network parameters from file'''
    #     print("Loading " + self.name + "'s network parameters from file")
    #     # torch.load(self.state_dict(), self.checkpoint_file)
    #     self.load_state_dict(torch.load(self.checkpoint_file))
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

