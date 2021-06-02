import pickle
import os
import re
import shutil
import sys
import random
import numpy as np
from scipy.spatial import distance
from collections import deque
import gym
import pickle
import torch
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from utils.misc import parse_obs, toroidal_distance

if os.name == 'posix':
    import matplotlib
    # matplotlib.use("macOSX")
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# Networks
#---------------------------------------------------
def param_update_hard(target, source):
    # replace source params with those of target
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def param_update_soft(target, source, tau):
    # update source params polyak averaging
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# # https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
# def average_gradients(model):
#     """ Gradient averaging. """
#     size = float(dist.get_world_size())
#     print('size = {}'.format(size))
#     print(a)
#     for param in model.parameters():
#         dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
#         param.grad.data /= size

# # https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
# def init_processes(rank, size, fn, backend='gloo'):
#     """ Initialize the distributed environment. """   
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)

def save_checkpoint(models, fp,  name, epoch):
    path = os.path.join(fp, 'checkpoints', name)
    mkdir_p(path)
    torch.save(models, os.path.join(path, 'epoch_{}.pth'.format(epoch)))

def load_checkpoint(fp, name, epoch=None):
    map_loc = None if torch.cuda.is_available() else {'cuda:0': 'cpu'}
    if epoch:
        # load specific epoch
        return torch.load(os.path.join(fp, 'checkpoints', name, 'epoch_{}.pth'.format(epoch)), map_location=map_loc)
    else:
        # load latest epoch
        path = os.path.join(fp, 'checkpoints', name)
        files = os.listdir(path)
        f_name = natural_sort(files)[-1]
        print('loading latest model: {}'.format(f_name))
        return torch.load(os.path.join(fp, 'checkpoints', name, f_name), map_location=map_loc)



#---------------------------------------------------
# Noise processes
#---------------------------------------------------
# Ornstein-Ulhenbeck Noise
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.02, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


#---------------------------------------------------
# Gumbel-Softmax
#---------------------------------------------------
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, -1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

#---------------------------------------------------
# Running Mean/Std
#---------------------------------------------------
class RunningMeanStdModel(torch.nn.Module):

    """Taken from rlpyt (hich is adapted from OpenAI baselines).
    Maintains a running estimate of mean and variance of data along each dimension,
    accessible in the `mean` and `var` attributes. """

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape).to(device))
        self.register_buffer("var", torch.ones(shape).to(device))
        self.register_buffer("count", torch.zeros(()).to(device))
        self.shape = shape

    def update(self, x):
        # _, T, B, _ = infer_leading_dims(x, len(self.shape))
        # x = x.view(T * B, *self.shape)
        x = torch.FloatTensor(x).to(device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        if self.count == 0:
            self.mean[:] = batch_mean
            self.var[:] = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean[:] = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var[:] = M2 / total
        self.count += batch_count


#---------------------------------------------------
# Plotting
#---------------------------------------------------
def plot_performance(fp, steps, avg_steps, rewards, train=True):
    print('plotting')
    plt.style.use('seaborn')
    palette = plt.get_cmap('Paired')

    # plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(rewards, color=palette(0))
    plt.plot(smoothed_rewards, color=palette(1))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if train:
        plt.savefig(os.path.join(fp, 'figures', 'tr_rewards.png'))
    else:
        plt.savefig(os.path.join(fp, 'figures', 'val_rewards.png'))
    plt.clf()

    plt.plot(steps, color=palette(2))
    plt.plot(avg_steps, color=palette(3))
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    if train:
        plt.savefig(os.path.join(fp, 'figures', 'tr_steps.png'))
    else:
        plt.savefig(os.path.join(fp, 'figures', 'val_steps.png'))
    plt.clf()

def plot_multiagent_performance_old(fp, steps, mean_steps, median_steps, rewards, mean_rewards, median_rewards, s_ratios):
    print('plotting')
    plt.style.use('seaborn')
    palette = plt.get_cmap('Paired')

    # plot results for each agent
    for i in range(len(steps)):
        plt.plot(rewards[i], color=palette(0), label='Raw')
        plt.plot(mean_rewards[i], color=palette(1), label='Mean')
        plt.plot(median_rewards[i], color=palette(1), linestyle='dashed', label='Median')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.ylim(-60.0, 60.0)
        plt.legend()
        plt.savefig(os.path.join(fp, 'figures', 'agent{}_val_rewards.png'.format(i)))
        plt.clf()

        plt.plot(steps[i], color=palette(2), label='Raw')
        plt.plot(mean_steps[i], color=palette(3), label='Mean')
        plt.plot(median_steps[i], color=palette(3), linestyle='dashed', label='Median')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.ylim(-5.0, 505.0)
        plt.legend()
        plt.savefig(os.path.join(fp, 'figures', 'agent{}_val_steps.png'.format(i)))
        plt.clf()

    plt.plot(s_ratios, color=palette(1))
    plt.xlabel('Episode')
    plt.ylabel('Success (%)')
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(fp, 'figures', 'predator_s_ratio.png'))
    plt.clf()


def plot_multiagent_performance(fp, results):
    total_epochs = len(results['success_ratios'])

    # tally preds in sensing range
    first_sensing_ratio = results['total_sensing_firsts'] / total_epochs
    second_sensing_ratio = results['total_sensing_seconds'] / total_epochs
    third_sensing_ratio = results['total_sensing_thirds'] / total_epochs

    # tally preds in comm range
    first_comm_ratio = results['total_comm_firsts'] / total_epochs
    second_comm_ratio = results['total_comm_seconds'] / total_epochs
    third_comm_ratio = results['total_comm_thirds'] / total_epochs

    print('plotting')
    plt.style.use('seaborn')
    palette = plt.get_cmap('Paired')

    # plot results for each agent
    for i in range(len(results['steps'])):
        plt.plot(results['all_rewards'][i], color=palette(0), label='Raw')
        plt.plot(results['mean_rewards'][i], color=palette(1), label='Mean')
        plt.plot(results['median_rewards'][i], color=palette(1), linestyle='dashed', label='Median')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.ylim(-60.0, 60.0)
        plt.legend()
        plt.savefig(os.path.join(fp, 'figures', 'agent{}_val_rewards.png'.format(i)))
        plt.clf()

        plt.plot(results['steps'][i], color=palette(2), label='Raw')
        plt.plot(results['mean_steps'][i], color=palette(3), label='Mean')
        plt.plot(results['median_steps'][i], color=palette(3), linestyle='dashed', label='Median')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.ylim(-5.0, 505.0)
        plt.legend()
        plt.savefig(os.path.join(fp, 'figures', 'agent{}_val_steps.png'.format(i)))
        plt.clf()

    plt.plot(results['success_ratios'], color=palette(1))
    plt.xlabel('Episode')
    plt.ylabel('Success (%)')
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(fp, 'figures', 'predator_s_ratio.png'))
    plt.clf()


    #############################################
    # Sensing range
    #############################################
    xs = np.arange(0, len(results['mean_in_sensing_range']))
    plt.plot(xs, results['mean_in_sensing_range'], color=palette(0), label='Raw')
    plt.plot(xs, [np.mean(results['mean_in_sensing_range'])]*len(xs), color=palette(1), label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Avg. Predators In Range (per time-step)')
    plt.ylim(-0.1, 3.1)
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_sensing_range_avg.png'))
    plt.clf()

    xs = np.arange(0, len(results['sensing_firsts']))
    firsts = np.array(results['sensing_firsts'], dtype=np.float)
    mask = np.isfinite(firsts)
    mean = np.mean([f for f in firsts if not np.isnan(f)])
    plt.plot(xs[mask], firsts[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, firsts, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time First Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_sensing_range_firsts.png'))
    plt.clf()

    seconds = np.array(results['sensing_seconds'], dtype=np.float)
    mean = np.mean([t for t in seconds if not np.isnan(t)])
    xs = np.arange(0, len(seconds))
    mask = np.isfinite(seconds)
    # get first value in list
    first_idx, last_idx = None, None
    for i in range(len(mask)):
        if mask[i]:
            first_idx = i
            break
    first_idx = 0 if first_idx is None else first_idx

    # get last vaue in list
    for i in range(len(mask)-1, -1, -1):
        if mask[i]:
            last_idx = i
            break
    last_idx = len(mask)-1 if last_idx is None else last_idx

    # fill NaN with near known value on the edges
    seconds_copy = np.copy(seconds)
    seconds_copy[:first_idx] = seconds_copy[first_idx]
    seconds_copy[last_idx + 1:] = seconds_copy[last_idx]
    mask = np.isfinite(seconds_copy)
    plt.plot(xs[mask], seconds_copy[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, seconds, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time Second Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_sensing_range_seconds.png'))
    plt.clf()

    thirds = np.array(results['sensing_thirds'], dtype=np.float)
    mean = np.mean([t for t in thirds if not np.isnan(t)])
    xs = np.arange(0, len(thirds))
    mask = np.isfinite(thirds)
    # get first value in list
    first_idx, last_idx = None, None
    for i in range(len(mask)):
        if mask[i]:
            first_idx = i
            break
    first_idx = 0 if first_idx is None else first_idx

    # get last vaue in list
    for i in range(len(mask)-1, -1, -1):
        if mask[i]:
            last_idx = i
            break
    last_idx = len(mask)-1 if last_idx is None else last_idx
    # fill NaN with near known value on the edges
    thirds_copy = np.copy(thirds)
    thirds_copy[:first_idx] = thirds_copy[first_idx]
    thirds_copy[last_idx + 1:] = thirds_copy[last_idx]
    mask = np.isfinite(thirds_copy)
    plt.plot(xs[mask], thirds_copy[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, thirds, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time Third Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_sensing_range_thirds.png'))
    plt.clf()

    bars = ('First', 'Second', 'Third')
    x_pos = np.arange(3)
    plt.bar(x_pos, [first_sensing_ratio, second_sensing_ratio, third_sensing_ratio], color=(palette(0), palette(0), palette(0)), edgecolor=palette(1), linewidth=2)
    plt.xticks(x_pos, bars)
    plt.savefig(os.path.join(fp, 'figures', 'predator_in_sensing_range_ratios.png'))
    plt.clf()

    #############################################
    # Comm range
    #############################################
    xs = np.arange(0, len(results['mean_in_comm_range']))
    plt.plot(xs, results['mean_in_comm_range'], color=palette(0), label='Raw')
    plt.plot(xs, [np.mean(results['mean_in_comm_range'])]*len(xs), color=palette(1), label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Avg. Predators In Range (per time-step)')
    plt.ylim(-0.1, 3.1)
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_comm_range_avg.png'))
    plt.clf()

    xs = np.arange(0, len(results['comm_firsts']))
    firsts = np.array(results['comm_firsts'], dtype=np.float)
    mask = np.isfinite(firsts)
    mean = np.mean([f for f in firsts if not np.isnan(f)])
    plt.plot(xs[mask], firsts[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, firsts, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time First Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_comm_range_firsts.png'))
    plt.clf()

    seconds = np.array(results['comm_seconds'], dtype=np.float)
    mean = np.mean([t for t in seconds if not np.isnan(t)])
    xs = np.arange(0, len(seconds))
    mask = np.isfinite(seconds)
    # get first value in list
    first_idx, last_idx = None, None
    for i in range(len(mask)):
        if mask[i]:
            first_idx = i
            break
    first_idx = 0 if first_idx is None else first_idx

    # get last vaue in list
    for i in range(len(mask)-1, -1, -1):
        if mask[i]:
            last_idx = i
            break
    last_idx = len(mask)-1 if last_idx is None else last_idx

    # fill NaN with near known value on the edges
    seconds_copy = np.copy(seconds)
    seconds_copy[:first_idx] = seconds_copy[first_idx]
    seconds_copy[last_idx + 1:] = seconds_copy[last_idx]
    mask = np.isfinite(seconds_copy)
    plt.plot(xs[mask], seconds_copy[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, seconds, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time Second Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_comm_range_seconds.png'))
    plt.clf()

    thirds = np.array(results['comm_thirds'], dtype=np.float)
    mean = np.mean([t for t in thirds if not np.isnan(t)])
    xs = np.arange(0, len(thirds))
    mask = np.isfinite(thirds)
    # get first value in list
    first_idx, last_idx = None, None
    for i in range(len(mask)):
        if mask[i]:
            first_idx = i
            break
    first_idx = 0 if first_idx is None else first_idx

    # get last vaue in list
    for i in range(len(mask)-1, -1, -1):
        if mask[i]:
            last_idx = i
            break
    last_idx = len(mask)-1 if last_idx is None else last_idx
    # fill NaN with near known value on the edges
    thirds_copy = np.copy(thirds)
    thirds_copy[:first_idx] = thirds_copy[first_idx]
    thirds_copy[last_idx + 1:] = thirds_copy[last_idx]
    mask = np.isfinite(thirds_copy)
    plt.plot(xs[mask], thirds_copy[mask], linestyle='--', dashes=(4,4), lw=2, color=palette(0))
    plt.plot(xs, thirds, color=palette(0), lw=2.5, label='Raw')
    plt.plot(xs, [mean]*len(xs), color=palette(1), lw=2.5, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Time Third Predator Enters')
    plt.ylim(0, 500)
    plt.legend()
    plt.savefig(os.path.join(fp, 'figures', 'predators_in_comm_range_thirds.png'))
    plt.clf()

    bars = ('First', 'Second', 'Third')
    x_pos = np.arange(3)
    plt.bar(x_pos, [first_comm_ratio, second_comm_ratio, third_comm_ratio], color=(palette(0), palette(0), palette(0)), edgecolor=palette(1), linewidth=2)
    plt.xticks(x_pos, bars)
    plt.savefig(os.path.join(fp, 'figures', 'predator_in_comm_range_ratios.png'))
    plt.clf()

    # save results dict
    with open(os.path.join(fp, 'results', 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)




#---------------------------------------------------
# Communication for training
#---------------------------------------------------
# def generate_perfect_comm(obs, env, is_prey=False):
#     obs = parse_obs(obs, env, is_prey)

#     # parse obs
#     dist = toroidal_distance(obs['pos'], obs['prey_pos'][0], env.world.size)
#     print('pred pos comm = {}'.format(obs['pos']))
#     print('prey pos comm = {}'.format(obs['prey_pos'][0]))
#     print('dist comm = {}'.format(dist))

#     if dist < env.world.sensor_range:
#         return np.array([1])
#     else:
#         print('BLAH')
#         return np.array([0])


#---------------------------------------------------
# Miscellaneous
#---------------------------------------------------
def mkdir_p(path):
    """
    Creates directory recursively if it does not already exist
    """
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def one_hot_encode(num, size):
    return np.eye(size)[num]

def thread_to_agent(vec, n_agents):
    # rearrange vector from per-thread to per-agent 
    return [np.vstack(vec[:, i]) for i in range(n_agents)]

def agent_to_thread(vec, n_threads):
    # rearrange vector from per-agent to per-thread
    return [[v[i] for v in vec] for i in range(n_threads)]

def count_in_range(sensing_range, obs_n, n_preds, w_size):
    prey_pos = obs_n[-1][0:2]
    total = 0
    for o in obs_n[:-1]:
        pos = o[0:2]
        dist = toroidal_distance(pos, prey_pos, w_size)
        if  dist < sensing_range:
            total += 1

    return total





