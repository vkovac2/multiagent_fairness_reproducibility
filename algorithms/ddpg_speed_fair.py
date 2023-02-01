import argparse
import os
import numpy as np  
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from multiagent.utils import VideoRecorder
from tqdm import tqdm

import re

from algorithms.models  import Actor, Critic
from algorithms.replay import ReplayBuffer
from algorithms.algo_utils import *
from baselines.bot_policies import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_Agent(object):
    def __init__(self, env, config, writer, index):
        self.env = env
        self.index = index
        self.learning_agent = True
        # action space depends on problem
        if config.comm_env:
            self.action_space = self.env.action_space[self.index].spaces[0]
        else:
            self.action_space = self.env.action_space[self.index]
        self.observation_space = self.env.observation_space[self.index].shape[0]
        self.normalize = config.normalize
        self.batch_size = config.batch_size
        self.clip_norm = config.clip_norm
        self.gamma = config.gamma
        self.tau = config.tau
        self.buffer_length = config.buffer_length
        self.writer = writer
        self.warmup_episodes = config.warmup_episodes
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.decay = config.decay
        self.max_speed = self.env.world.agents[0].max_speed
        self.log_interval = config.log_interval
        self.lambda_coeff = config.lambda_coeff

        # actor and target actor
        self.actor = Actor(self.observation_space, config.actor_hidden, self.action_space.shape[0]).to(device)
        self.actor_target = Actor(self.observation_space, config.actor_hidden, self.action_space.shape[0]).to(device)

        # critic and target critic
        self.critic = Critic(self.observation_space, config.critic_hidden, self.action_space.shape[0]).to(device)
        self.critic_target = Critic(self.observation_space, config.critic_hidden, self.action_space.shape[0]).to(device)

        # copy params between source and target networks
        param_update_hard(self.actor_target, self.actor)
        param_update_hard(self.critic_target, self.critic)

        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # running stats, replay buffer and random process
        if self.normalize:
            self.obs_rms = RunningMeanStdModel(self.observation_space)
            self.norm_obs_clip = self.env.world.size / 2.0
            self.norm_obs_var_clip = config.norm_obs_var_clip
        self.replay_buffer = ReplayBuffer(self.buffer_length)
        self.noise = OUNoise(self.action_space, decay_period=self.decay)

        # init exploration policy
        self.exp_actor = decentralized_predator(env, self.index, config.test_predator, False)


    def sample_action(self, obs, epoch, is_training=True):
        # e-greedy exploration
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max((self.decay - epoch)/self.decay, 0.0)

        if self.index == 0:
            self.writer.add_scalar('misc/predator_epsilon', epsilon, epoch)

        if is_training and np.random.uniform(0, 1) > (1.0 - epsilon):
            # exploration policy
            a_i_vec = self.exp_actor.sample_action(obs)
            action = a_i_vec[0]
        else:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                action = self.actor(obs)
                if is_training:
                    # noise process
                    action = self.noise.get_action(action.cpu(), t=epoch)

                action = (action / torch.max(torch.abs(action))) * self.max_speed
                action = action.squeeze(0).cpu().detach().numpy()

        return action


    def update_policy(self, epoch, other_policies):
        if epoch <= self.warmup_episodes:
            return

        # unpack replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        if self.normalize:
            state_batch = self.normalize_obs(torch.FloatTensor(state_batch).to(device))
            next_state_batch = self.normalize_obs(torch.FloatTensor(next_state_batch).to(device))
        else:
            state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
            next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)

        action_batch = torch.FloatTensor(np.array(action_batch)).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(device)
   
        # Q(s, a)
        curr_Q = self.critic(state_batch, action_batch)

        # mu'(a'|s') of next state from target policy network
        next_actions = self.actor_target(next_state_batch)

        # Q'(s', a') of next state/action from target Q network
        next_Q = self.critic_target(next_state_batch, next_actions.detach())

        # target Q values
        done_batch = done_batch.unsqueeze(1)
        expected_Q = reward_batch + (1.0 - done_batch) * self.gamma * next_Q

        # update critic
        critic_loss = F.mse_loss(curr_Q, expected_Q)
        self.critic_opt.zero_grad()
        critic_loss.backward() 
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
        self.critic_opt.step()

        # standard policy gradient actor loss
        new_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, new_actions).mean()                                      

        #  fairness loss: \Sum_j 1 - cos(\theta_i - \theta_j)
        xs, ys = torch.split(new_actions, 1, dim=1)
        thetas = torch.atan2(ys, xs)

        diffs = []
        for other in other_policies:
            other_xs, other_ys = torch.split(other(state_batch), 1, dim=1)
            other_thetas = torch.atan2(other_ys, other_xs)
            diffs.append(1 - torch.cos(thetas - other_thetas))
        fairness_loss = torch.cat(diffs).mean()

        total_actor_loss = actor_loss + self.lambda_coeff * fairness_loss

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)
        self.actor_opt.step()

        # soft update params of target networks
        param_update_soft(self.actor_target, self.actor, self.tau)
        param_update_soft(self.critic_target, self.critic, self.tau)

        # book-keeping
        if epoch % self.log_interval == 0:
            self.writer.add_scalar('loss/agent{}_actor_loss'.format(self.index), actor_loss, epoch)
            self.writer.add_scalar('loss/agent{}_fairness_loss'.format(self.index), fairness_loss, epoch)
            self.writer.add_scalar('loss/agent{}_total_actor_loss'.format(self.index), total_actor_loss, epoch)
            self.writer.add_scalar('loss/agent{}_critic_loss'.format(self.index), critic_loss, epoch)
            self.writer.add_scalar('norm/agent{}_actor_norm'.format(self.index), actor_norm, epoch)
            self.writer.add_scalar('norm/agent{}_critic_norm'.format(self.index), critic_norm, epoch)

                                   

    def normalize_obs(self, x):                         
        obs_var = self.obs_rms.var
        obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
        x = torch.clamp((x - self.obs_rms.mean) /
                        obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        return x

    def update_stats(self, obs):
        if self.normalize:
            self.obs_rms.update(obs)


    def get_params(self):
        return {
            'actor' : self.actor.state_dict(),
            'critic' : self.critic.state_dict(),
            'actor_target' : self.actor_target.state_dict(),
            'critic_target' : self.critic_target.state_dict(),
            'actor_opt' : self.actor_opt.state_dict(),
            'critic_opt' : self.critic_opt.state_dict()
        }

    def load_params(self, params):
        print('loading params for agent {}!'.format(self.index))
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor_opt.load_state_dict(params['actor_opt'])
        self.critic_opt.load_state_dict(params['critic_opt'])

        # double check in train mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()


class DDPG_Runner():
    def __init__(self, env, config):
        # training properties
        self.env = env
        self.config = config
        self.num_agents = env.n
        self.n_epochs = config.n_epochs
        self.n_steps = self.env.world.n_steps
        self.n_epochs_test = config.n_epochs_test
        self.update_steps = config.update_steps
        self.discrete = config.discrete
        self.use_curriculum = config.use_curriculum
        self.warmup_episodes = config.warmup_episodes
        self.pred_vel_start = config.pred_vel_start
        self.pred_vel_end = config.pred_vel_end
        self.decay = config.decay
        self.pred_test_vel = config.pred_test_vel

        self.epoch_start = 1

        print('curriculum = {}'.format(self.use_curriculum))

        # environment properties
        self.directory = config.directory
        self.log_interval = config.log_interval
        self.checkpoint_interval = config.checkpoint_interval
        self.checkpoint_path = config.checkpoint_path

        # tensorboard
        split_dir = self.directory.split('/')
        self.writer = SummaryWriter(log_dir=os.path.join('/'.join(split_dir[:-1]), 
                                'runs', split_dir[-1]))

        # init predators as DDPG
        self.predators = [DDPG_Agent(env, config, self.writer, i) for i in range(self.env.num_preds)]
        self.num_preds = len(self.predators)

        if config.mode is 'train' and self.checkpoint_path:
            print('loading warm-up model!')
            # init predators from checkpoint
            for i, a in enumerate(self.predators):
                params = load_checkpoint(self.checkpoint_path, 'agent_{}'.format(i))
                self.predators[i].load_params(params)

                #starting epoch
                path = os.path.join(self.checkpoint_path, 'checkpoints', 'agent_{}'.format(i))
                files = os.listdir(path)
                f_name = natural_sort(files)[-1]
                epochs = re.findall(r'\d+', f_name)
                self.epoch_start = int(epochs[0]) + 1




        # init prey as bots
        self.prey = [decentralized_prey(env, i+len(self.predators), config.test_prey, False) for i in range(self.env.num_prey)]
        self.num_prey = len(self.prey)
        self.agents = self.predators + self.prey
        self.is_training = True

        # set start speed
        for i, a in enumerate(self.env.world.agents):
            if i < self.num_preds:
                a.max_speed = self.pred_vel_start if config.mode is 'train' else self.pred_test_vel

    def sample_actions(self, obs_n, epoch):
        actions, action_vecs = [], []
        for i in range(self.num_agents):
            if self.agents[i].learning_agent:
                a_i = self.agents[i].sample_action(obs_n[i], epoch, self.is_training)
                a_i_vec = np.array([a_i, np.zeros(self.env.world.dim_c)])  # NOTE: PLACEHOLDER FOR COMMUNICATION
            else:
                a_i_vec = self.agents[i].sample_action(obs_n[i])
                a_i = a_i_vec[0]

            actions.append(a_i)
            action_vecs.append(a_i_vec)
        
        return actions, action_vecs

    def step_curriculum(self, epoch):
        # calculate updated max speed
        new_max = self.pred_vel_end + (self.pred_vel_start - self.pred_vel_end) * max((self.decay - epoch)/self.decay, 0.0)

        # apply to predators
        for i, a in enumerate(self.env.world.agents):
            if i < self.num_preds:
                a.max_speed = new_max

    def train(self):
        self.is_training = True
        train_steps = 0
        total_caps = 0
        successes = 0

        #if starting from checkpoint, set the initial speed
        if self.epoch_start > 1 and self.use_curriculum:
            c = self.step_curriculum(self.epoch_start)

        for epoch in tqdm(range(self.epoch_start, self.n_epochs+1)):
            rewards = [0.0 for _ in range(self.num_agents)]
            obs_n = self.env.reset()
            for step in range(1, self.n_steps+1):
                train_steps += 1

                # update running stats
                for i in range(self.num_preds):
                    self.predators[i].update_stats(obs_n[i])

                # sample action from policy and step env
                act_n, act_n_vec = self.sample_actions(obs_n, epoch)
                next_obs_n, reward_n, done_n, info_n = self.env.step(act_n_vec)

                # collect rewards
                for i in range(self.num_agents):
                    rewards[i] += reward_n[i]


                if all(done_n) or step == self.n_steps:
                    done_n = [1 for _ in range(self.num_agents)]
                    for j in range(self.num_agents):
                        if info_n[j]['active'] and self.agents[j].learning_agent:
                            # update buffer for each agent
                            self.agents[j].replay_buffer.push(obs_n[j], act_n[j], reward_n[j], next_obs_n[j], done_n[j])

                    # update success ratio
                    if step < self.n_steps:
                        # predators succeeded
                        successes += 1
                    s_ratio = successes / epoch

                    # check for curriculum update
                    if self.use_curriculum:
                        c = self.step_curriculum(epoch)

                    # checkpoints
                    if epoch > self.warmup_episodes and epoch % self.checkpoint_interval == 0:
                        for k in range(self.num_agents):
                            if self.agents[k].learning_agent:
                                save_checkpoint(self.agents[k].get_params(), self.directory, 'agent_{}'.format(k), epoch)
                    # logging
                    if epoch % self.log_interval == 0:
                        self.writer.add_scalar('epoch/steps', step, epoch)
                        self.writer.add_scalar('epoch/pursuer_rewards', rewards[0], epoch)
                        self.writer.add_scalar('misc/success_ratio', s_ratio, epoch)
                        self.writer.add_scalar('misc/total_captures', successes, epoch)
                        self.writer.add_scalar('misc/predator_max_speed', self.env.world.agents[0].max_speed, epoch)

                    break

                else:
                    # update buffer for each agent
                    done_n = [0 for _ in range(self.num_agents)]
                    for j in range(self.num_agents):
                        if self.agents[j].learning_agent:
                            if info_n[j]['active']:
                                self.agents[j].replay_buffer.push(obs_n[j], act_n[j], reward_n[j], next_obs_n[j], done_n[j])

                            if len(self.agents[j].replay_buffer) > self.agents[j].batch_size and train_steps % self.update_steps == 0:
                                other_policies = [ag.actor for i, ag in enumerate(self.agents) if i is not j and ag.learning_agent]
                                self.agents[j].update_policy(epoch, other_policies)

                    # update observation                    
                    obs_n = next_obs_n

        print('success ratio = {}'.format(s_ratio))
        self.env.close()
        return s_ratio


    def test(self, render=True):
        self.is_training = False

        # load checkpoints
        print('loading checkpoints for test!')
        for i, agent in enumerate(self.agents):
            if agent.learning_agent:
                params = load_checkpoint(self.checkpoint_path, 'agent_{}'.format(i))
                agent.load_params(params)
            
        # evaluate
        steps = [[] for _ in range(self.num_agents)]
        mean_steps = [[] for _ in range(self.num_agents)]
        median_steps = [[] for _ in range(self.num_agents)]
        all_rewards = [[] for _ in range(self.num_agents)]
        mean_rewards = [[] for _ in range(self.num_agents)]
        median_rewards = [[] for _ in range(self.num_agents)]
        successes = 0
        s_ratios = []
        for epoch in tqdm(range(1, self.n_epochs_test+1)):
            # to record rollout
            rec = VideoRecorder(self.env, path=os.path.join(self.checkpoint_path, 'videos', 'run_{}.mp4'.format(epoch)))

            # reset
            obs_n = self.env.reset()
            step = 0
            rewards = [0.0 for _ in range(self.num_agents)]
            done = False

            # hit it!
            while not done:
                # update running stats
                for i in range(self.num_preds):
                    self.predators[i].update_stats(obs_n[i])
                    
                act_n, act_n_vec = self.sample_actions(obs_n, epoch)
                next_obs_n, reward_n, done_n, _ = self.env.step(act_n_vec)

                for i in range(self.num_agents):
                    rewards[i] += reward_n[i]

                # update
                step += 1
                if all(done_n) or step == self.n_steps:
                    if step < self.n_steps:
                        successes += 1
                    done = True

                # update observation                    
                obs_n = next_obs_n

                rec.capture_frame()


            for j in range(self.num_agents):
                steps[j].append(step)
                mean_steps[j].append(np.mean(steps[j][-10:]))
                median_steps[j].append(np.median(steps[j][-10:]))
                all_rewards[j].append(np.sum(rewards[j]))
                mean_rewards[j].append(np.mean(all_rewards[j][-10:]))
                median_rewards[j].append(np.median(all_rewards[j][-10:]))

            s_ratios.append(successes/epoch)

            # close cap
            rec.close()

        print('success ratio = {}'.format(s_ratios[-1]))
        plot_multiagent_performance_old(self.checkpoint_path, steps, mean_steps, median_steps,
                     all_rewards, mean_rewards, median_rewards, s_ratios)
        self.env.close()