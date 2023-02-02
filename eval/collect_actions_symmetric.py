import argparse
import os
import pickle
import numpy as np  
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from multiagent.utils import VideoRecorder

# hack to get relative package references to work
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from algorithms.algo_utils import *
from baselines.bot_policies import *
from configs import Config_DDPG_Symmetric

class Trajectory_Collector():
    def __init__(self, env, config):
        # training properties
        self.env = env
        self.config = config
        self.num_agents = env.n
        self.n_epochs = config.n_epochs
        self.n_steps = config.n_steps
        self.discrete = False
        self.pred_policy = config.pred_policy
        self.pred_vel = config.pred_vel
        self.num_prey = config.nb_prey

        # environment properties
        self.directory = config.directory
        self.checkpoint_path = config.checkpoint_path

        self.render = config.render

        # tensorboard writer
        split_dir = self.directory.split('/')
        self.writer = SummaryWriter(log_dir=os.path.join('/'.join(split_dir[:-1]), 
                                'runs', split_dir[-1]))

        # init predators
        if self.pred_policy == 'ddpg':
            from algorithms.ddpg_symmetric import Symmetric_DDPG_Agent, Copy_DDPG_Agent
            # self.predators = [DDPG_Agent(env, config, self.writer, i) for i in range(self.env.num_preds)]
            #self.predators = [DDPG_Agent(env, config, self.writer, i) for i in range(self.env.num_preds)]
            self.predators = []
            self.reference_agent = None
            if self.checkpoint_path:
                for i in range(self.env.num_preds):
                    print('loading warm-up model!')
                    # init predators from checkpoint   
                    if self.config.checkpoint_epoch:
                        params = load_checkpoint(self.checkpoint_path, 'agent_{}'.format(i), epoch=self.config.checkpoint_epoch)
                        print(params.keys())
                    else:
                        params = load_checkpoint(self.checkpoint_path, 'agent_{}'.format(i))
                        print(params.keys())
                        
                    if "critic" in params.keys():
                        self.predators.append(Symmetric_DDPG_Agent(env, config, self.writer, i))
                        self.reference_agent = self.predators[-1]
                    else:
                        assert self.reference_agent is not None
                        self.predators.append(Copy_DDPG_Agent(env, config, self.reference_agent, i))
                    
                    self.predators[-1].load_params(params)
            else:
                raise ValueError('Path to checkpoint must be provided to test policy!')
        else:
            self.predators = [decentralized_predator(self.env, i, config.pred_policy, False) for i in range(self.env.num_preds)]
        self.num_preds = len(self.predators)
       
        # init prey as DDPG
        self.prey = [decentralized_prey(self.env, i+len(self.predators), config.prey_policy, False) for i in range(self.env.num_prey)]
        self.num_prey = len(self.prey)
        self.agents = self.predators + self.prey

        self.agent_keys = ['p{}'.format(i+1) for i in range(self.env.num_preds)]
        self.agent_keys.extend(['prey{}'.format(j+1) for j in range(self.env.num_prey)])

        assert len(self.agents) == self.num_agents

        # set start speed
        for i, a in enumerate(self.env.world.agents):
            if i < self.num_preds:
                a.max_speed = self.pred_vel

    def sample_actions(self, obs_n):
        actions = []
        for i in range(self.num_agents):
            if self.agents[i].learning_agent:
                a_i = self.agents[i].sample_action(obs_n[i], 0, False)
                a_i = np.array([a_i, np.zeros(self.env.world.dim_c)])
            else:
                a_i = self.agents[i].sample_action(obs_n[i])
            actions.append(a_i)
        
        return actions

    
    def run(self):
        trajectories = {
            'actions' : [],
            'positions' : []
        }
        successes = 0
        for epoch in tqdm(range(1, self.n_epochs+1)):
            # to record rollout
            if self.render:
                rec = VideoRecorder(self.env, path=os.path.join(self.directory, 'videos', 'run_{}.mp4'.format(epoch)))

            actions = {k:[] for k in self.agent_keys}
            positions = {k:[] for k in self.agent_keys}

            obs_n = self.env.reset()
            for step in range(1, self.n_steps+1):
                # print('step {}'.format(step))


                # sample action from policy and step env
                act_n = self.sample_actions(obs_n)
                next_obs_n, reward_n, done_n, info_n = self.env.step(act_n)

                for i in range(self.num_agents):
                    actions[self.agent_keys[i]].append(act_n[i][0])
                    positions[self.agent_keys[i]].append(next_obs_n[i][:2])
                    # positions[self.agent_keys[i]].append(next_obs_n[i][2:4])

                if self.render:
                    rec.capture_frame()

                # check done
                if all(done_n) or step == self.n_steps:
                    if step < self.n_steps:
                        successes += 1
                    trajectories['actions'].append(actions)
                    trajectories['positions'].append(positions)
                    break

                # update observation         
                obs_n = next_obs_n
            
            if self.render:
                rec.close()

            file = open(os.path.join(self.directory, 'trajectories.pkl'), "wb")
            pickle.dump(trajectories, file)
            file.close()

        print('success ratio = {}'.format(successes/epoch))
        file = open(os.path.join(self.directory, 'trajectories.pkl'), "wb")
        pickle.dump(trajectories, file)
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--env', type=str, default='simple_torus')
    parser.add_argument('--pred_policy', default='random', help='Predator strategy.')
    parser.add_argument('--prey_policy', default='noop', help='Prey strategy.')
    parser.add_argument('--pred_vel', type=float, default=1.0, help='predator velocity.')
    parser.add_argument('--n_steps', type=int, default=167, help='number of steps to run per epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--directory', type=str, default='results/stored_trajectories/', help='path to save')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to load checkpoint from')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='checkpoint epoch')
    parser.add_argument('--seed', type=int, default=72, help='checkpoint epoch')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--collaborative', action='store_true')
    parser.add_argument('--equivariant', action='store_true')
    parser.add_argument('--num_landmarks', type = int, default=0)
    parser.add_argument('--nb_pred', type=int, default=3)
    parser.add_argument('--nb_prey', type=int, default=1)

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    # hacky way to get configs to work for trained models
    from algorithms.ddpg_symmetric import Symmetric_DDPG_Agent
    config = Config_DDPG_Symmetric()
    config.env = args.env
    config.pred_policy = args.pred_policy
    config.prey_policy = args.prey_policy
    config.pred_vel = args.pred_vel
    config.n_steps = args.n_steps
    config.n_epochs = args.n_epochs
    config.directory = args.directory
    config.checkpoint_path = args.checkpoint_path
    config.checkpoint_epoch = args.checkpoint_epoch
    config.equivariant = args.equivariant
    config.collaborative = args.collaborative
    config.nb_pred = args.nb_pred
    config.nb_prey = args.nb_prey
    # config.mode = 'train'
    config.render = args.render
    config.num_landmarks = args.num_landmarks

    print("Equivariant: " + str(config.equivariant))
    print("Pred Vel: " + str(config.pred_vel))

    comm_envs = []
    if config.env in comm_envs:
        config.comm_env = True
    else:
        config.comm_env = False

    # make env
    scenario = scenarios.load(config.env + '.py').Scenario()
    world = scenario.make_world(config, discrete=False)
    # create multi-agent env
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.terminal)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    full_path = 'vel_{}/'.format(config.pred_vel)
    config.directory = os.path.join(config.directory, full_path)
        
    # results path
    if not os.path.exists(config.directory):
        os.makedirs(config.directory)
        os.makedirs(os.path.join(config.directory, 'videos'))


    runner = Trajectory_Collector(env, config)
    runner.run()
    env.close()
