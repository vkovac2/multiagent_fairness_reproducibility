import argparse
import os
import numpy as np  
from multiagent.utils import VideoRecorder
from datetime import datetime
from tqdm import tqdm

# hack to get relative package references to work
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from bot_policies import *
from algorithms.algo_utils import *
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


class MultiAgent_Baseline():
    def __init__(self, env, config):
        # training properties
        self.env = env
        self.config = config
        self.num_agents = env.n
        self.n_epochs = config.n_epochs
        self.n_steps = config.n_steps
        self.n_epochs_test = config.n_epochs_test
        self.discrete = False

        # environment properties
        self.directory = config.directory
        self.log_interval = config.log_interval

        # init predators
        self.predators = [decentralized_predator(self.env, i, config.pred_policy, False) for i in range(self.env.num_preds)]
        self.num_preds = len(self.predators)
       
        # init prey as DDPG
        self.prey = [decentralized_prey(self.env, i+len(self.predators), config.prey_policy, False) for i in range(self.env.num_prey)]
        self.num_prey = len(self.prey)
        self.agents = self.predators + self.prey

        assert len(self.agents) == self.num_agents


    def sample_actions(self, obs_n):
        actions = []
        for i in range(self.num_agents):
            a_i = self.agents[i].sample_action(obs_n[i])
            actions.append(a_i)
        
        return actions

    
    def test(self, render=False, step=False):
        steps = [[] for _ in range(self.num_agents)]
        avg_steps = [[] for _ in range(self.num_agents)]
        all_rewards = [[] for _ in range(self.num_agents)]
        successes = 0
        train_steps = 0
        s_ratios = []
        for epoch in tqdm(range(1, self.n_epochs_test+1)):
            rewards = [[] for _ in range(self.num_agents)]

            obs_n = self.env.reset()
            for step in range(1, self.n_steps+1):
                train_steps += 1

                # sample action from policy and step env
                act_n = self.sample_actions(obs_n)
                next_obs_n, reward_n, done_n, info_n = self.env.step(act_n)

                for i, o in enumerate(next_obs_n):
                    print('agent {} obs = \n{}'.format(i, o))

                for i, r in enumerate(reward_n):
                    print('agent {} reward = \n{}'.format(i, r))

                if render:
                    self.env.render()

                if args.step:
                    input('press enter to continue...')

                # collect rewards
                for i in range(self.num_agents):
                    rewards[i].append(reward_n[i])

                # check done
                if all(done_n) or step == self.n_steps:
                    if step < self.n_steps:
                        successes += 1
                    
                    for j in range(self.num_agents):
                        # logging
                        steps[j].append(step)
                        avg_steps[j].append(np.mean(steps[j][-10:]))
                        all_rewards[j].append(np.sum(rewards[j]))
                    s_ratios.append(successes/epoch)
                    print('success ratio = {}'.format(s_ratios[-1]))
                    break

                # update observation         
                obs_n = next_obs_n

        print('success ratio = {}'.format(s_ratios[-1]))
        self.env.close()


    def run(self):
        # evaluate
        steps = [[] for _ in range(self.num_agents)]
        avg_steps = [[] for _ in range(self.num_agents)]
        all_rewards = [[] for _ in range(self.num_agents)]

        for epoch in tqdm(range(1, self.n_epochs+1)):
            # to record rollout
            rec = VideoRecorder(self.env, path=os.path.join(self.directory, 'videos', 'run_{}.mp4'.format(epoch)))

            # reset
            obs_n = self.env.reset()
            step = 0
            rewards = [[] for _ in range(self.num_agents)]
            done = False

            # hit it!
            while not done:
                act_n = self.sample_actions(obs_n)
                next_obs_n, reward_n, done_n, _ = self.env.step(act_n)

                for i in range(self.num_agents):
                    rewards[i].append(reward_n[i])

                # update
                step += 1
                if all(done_n) or step == self.n_steps:
                    done = True

                # update observation                    
                obs_n = next_obs_n

                rec.capture_frame()


            for j in range(self.num_agents):
                steps[j].append(step)
                avg_steps[j].append(np.mean(steps[j][-10:]))
                all_rewards[j].append(np.sum(rewards[j]))

            # close cap
            rec.close()

        plot_multiagent_performance(self.directory, steps, avg_steps, all_rewards, s_ratios)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--mode', type=str, default='run')
    parser.add_argument('--env', type=str, default='simple_torus')
    parser.add_argument('--pred_policy', default='random', help='Predator strategy.')
    parser.add_argument('--prey_policy', default='noop', help='Prey strategy.')
    parser.add_argument('--n_steps', type=int, default=500, help='number of steps to run per epoch')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--n_epochs_test', type=int, default=10, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--directory', type=str, default='results/', help='path to save')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--log_interval', type=int, default=10, help='training logs update interval')
    parser.add_argument('--step', action='store_true', help='Step through env execution (for debugging).')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning.')
    parser.add_argument('--comm_type', default='none',help='Use perfect communication in speaking envs.')
    parser.add_argument('--use_sensor_range', action='store_true', help='Use sensor range for predators.')
    parser.add_argument('--sensing_range', type=float, default=4.5, help='sensing range')
    parser.add_argument('--comm_range', type=float, default=0.75, help='communication range')
    parser.add_argument('--comm_noise', type=float, default=0.5, help='communication channel noise')
    parser.add_argument('--pos_noise', type=float, default=0.5, help='predator position noise')
    parser.add_argument('--test_noise', type=float, default=0.5, help='test noise')
    parser.add_argument('--init_range_thresh', type=float, default=1.0, help='percentage predators init outside sensing range')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--nb_agents', type=int, default=3)
    parser.add_argument('--nb_prey', type=int, default=1)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    # np.random.seed(612)
    args.test_distance = args.distance_start = args.sensing_range

    # make env
    scenario = scenarios.load(args.env + '.py').Scenario()
    world = scenario.make_world(args, discrete=False)
    # create multi-agent env
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.terminal)


    args.directory = 'results/{}_{}_{}/{}'.format(args.pred_policy,
                                                 args.prey_policy,
                                                 args.env,
                                                'exp' + datetime.now().strftime("_%m_%d_%Y__%H_%M_%S"))


    # results path
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # figures path
    fig_path = os.path.join(args.directory, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # videos path
    vid_path = os.path.join(args.directory, 'videos')
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # checkpoints path
    ch_path = os.path.join(args.directory, 'checkpoints')
    if not os.path.exists(ch_path):
        os.makedirs(ch_path)

    runner = MultiAgent_Baseline(env, args)
    if args.mode == 'test':
        runner.test(args.render, args.step)
    else:
        runner.run()

    env.close()