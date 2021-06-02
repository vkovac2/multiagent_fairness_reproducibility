import random
import numpy as np
from collections import deque
import torch


# basic replay buffer (single agent)
class ReplayBuffer:
    def __init__(self, max_size, comm=False):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.comm = comm

    def push(self, state, action, reward, next_state, done):
        if self.comm:
            experience = (state, action[0], action[1], np.array([reward]), next_state, done)
        else:
            experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        if self.comm:
            comm_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            if self.comm:
                state, action, comm, reward, next_state, done = experience
                comm_batch.append(comm)
            else:
                state, action, reward, next_state, done = experience

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        if self.comm:
            return (state_batch, action_batch, comm_batch, reward_batch, next_state_batch, done_batch)
        else:
            return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    # def sample_sequence(self, batch_size):
    #     state_batch = []
    #     action_batch = []
    #     reward_batch = []
    #     next_state_batch = []
    #     done_batch = []

    #     min_start = len(self.buffer) - batch_size
    #     start = np.random.randint(0, min_start)

    #     for sample in range(start, start + batch_size):
    #         state, action, reward, next_state, done = self.buffer[start]
    #         state, action, reward, next_state, done = experience
    #         state_batch.append(state)
    #         action_batch.append(action)
    #         reward_batch.append(reward)
    #         next_state_batch.append(next_state)
    #         done_batch.append(done)

    #     return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)



# Modified from: https://github.com/cyoon1729/Multi-agent-reinforcement-learning/tree/1c59291d2a65906e15f1dc2a6113d1fd18592506
class MultiAgentReplayBuffer:
    
    def __init__(self, num_agents, max_size):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=max_size)
    
    def push(self, obs, action, reward, next_obs, done):
        experience = (obs, action, np.array(reward), next_obs, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        obs_batch = [[] for _ in range(self.num_agents)]  # [ [states of agent 1], ... ,[states of agent n] ]    ]
        action_batch = [[] for _ in range(self.num_agents)] # [ [actions of agent 1], ... , [actions of agent n]]
        reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]
        done_batch = [[] for _ in range(self.num_agents)]

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            obs_n, action_n, reward_n, next_obs_n, done_n = experience

            for i in range(self.num_agents):
                obs_batch[i].append(obs_n[i])
                action_batch[i].append(action_n[i])
                reward_batch[i].append(reward_n[i])
                next_obs_batch[i].append(next_obs_n[i])
                done_batch[i].append(done_n[i])

        return (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
