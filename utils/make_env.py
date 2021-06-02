
import numpy as np
import gym
from utils.wrappers import NormalizedEnv, SubprocVecEnv, DummyVecEnv


'''
Functions to facilitate environment creation. Currently only supports parallelization of particle environments.
'''

def make_env(config):
    # make env
    if config.normalize_env:
        env = NormalizedEnv(gym.make(config.env))
    elif config.n_threads is not None:
        print('making parallel env')
        env = make_parallel_env(config)
    elif config.particle_env:
        print('making particle env')
        env = make_particle_env(config)
    else:
        env = gym.make(config.env)

    return env


# https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py
def make_particle_env(config):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(config.env + ".py").Scenario()
    # create world
    world = scenario.make_world(config, discrete=config.discrete)
    # create multiagent environment
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                                reward_callback=scenario.reward,
                                observation_callback=scenario.observation,
                                info_callback=scenario.benchmark_data,
                                done_callback=scenario.terminal)
    return env

# https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/main.py
def make_parallel_env(config):
    def get_env_fn(seed, rank):
        def init_env():
            env = make_particle_env(config)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if config.n_threads == 1:
        return DummyVecEnv([get_env_fn(config.seed, 0)])
    else:
        return SubprocVecEnv([get_env_fn(config.seed, i) for i in range(config.n_threads)])


