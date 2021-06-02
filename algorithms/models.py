import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Policy(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(MLP_Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class TwoHead_ActorCritic(nn.Module):
    def __init__(self, input_size, actor_hidden, critic_hidden, num_actions):
        super(TwoHead_ActorCritic, self).__init__()
        self.num_actions = num_actions

        self.critic = nn.Sequential(
            nn.Linear(input_size, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_size, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, num_actions)
        )


    def forward(self, x):
        # actor
        x_logits = self.actor(x)
        x_probs = F.softmax(x_logits, dim=0)

        # critic
        x_val = self.critic(x)

        return x_probs, x_val


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0] + num_actions, hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], 1)

        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        xa = torch.cat([x,a], 1)
        xa = F.relu(self.fc2(xa))
        xa = F.relu(self.fc3(xa))
        qval = self.fc4(xa)

        return qval

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_actions)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

#---------------------------------------------------
# Communication Policies
#---------------------------------------------------
class CommActor(nn.Module):
    def __init__(self, input_size, hidden_size, actions, init_w=3e-3):
        super(CommActor, self).__init__()

        # TODO: for initial test, ensure comm action dim = 1
        
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])

        # environment head
        self.fc_env = nn.Linear(hidden_size[1], actions[0].shape[0])

        # communication head
        self.fc_comm = nn.Linear(hidden_size[1], actions[1].shape[0])
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc_env.weight.data.uniform_(-init_w, init_w)
        self.fc_comm.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # env action
        x_env = torch.tanh(self.fc_env(x))

        # comm action
        x_comm = self.fc_comm(x)

        return x_env, x_comm


#--------------------------------------
# Initialization
#--------------------------------------
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)





