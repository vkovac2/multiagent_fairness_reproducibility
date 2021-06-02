import torch
from datetime import datetime
import sys
import os
import logging
import time
import shutil
from utils.misc import create_exp_dir

logger = logging.getLogger("logger")

#--------------------------------------
# Configs
#--------------------------------------
class BaseConfig(object):
    ######## general #########
    seed = 612
    n_threads = None

    ######## environment ########
    env = 'med_pnp'                             # environment 
    gamma = 0.99                                # discount factor
    discrete = False                            # discrete action space

    ######## training #########
    opt = 'adam'                                # optimizer
    actor_hidden = [128, 128]                   # actor hidden units (policy)
    critic_hidden = [128, 128, 128]             # critic hidden units (value function)
    actor_lr = 0.0001                           # actor learning rate
    critic_lr = 0.001                           # critic learning rate
    n_epochs = 125000                            # number of training epochs
    update_steps = 5                            # number of steps between policy updates
    batch_size = 512                            # batch size
    buffer_length = 500000                      # replay buffer length
    tau = 0.01                                  # polyak averaging factor
    clip_norm = 0.5                             # gradient clip norm
    use_curriculum = True                       # curriculum learning flag
    normalize = False                           # normalize inputs
    norm_obs_var_clip = 1e-6                    # threshold to clip obs variance 
    warmup_episodes = 1000                      # number of experience episodes before training begins
    checkpoint_interval = 5000                  # episodes between model checkpoints

    ######## testing #########
    n_epochs_test = 100                         # number of test epochs

    ######## book-keeping ########
    log_interval = 50                           # episodes between log updates
    checkpoint_interval = 1000                  # episodes between model checkpoints
    checkpoint_path = None                      # path for loading model checkpoints
    scripts = ['main.py', 'configs.py']

    def show(self):
        attrs = [attr for attr in dir(self) if (not attr.startswith('__') and attr != "show")]
        logger.info('\n'.join("%s: %s" % (item, getattr(self, item)) for item in attrs))


class Config_DDPG_Speed(BaseConfig): 
    algorithm = 'ddpg_speed'                    # algorithm name
    pred_vel_start = 1.2                        # curriculum start value
    pred_vel_end = 0.4                          # curriculum end value
    decay = 120000                               # number of episodes over curriculum
    pred_test_vel = 0.9                         # predator test speed
    epsilon_start = 1.0                        # epsilon start for e-greedy policy
    epsilon_end = 1.0                          # epsilon end for e-greedy policy
    test_prey = 'cosine'                        # bot policy to use for prey
    test_predator = 'greedy'                    # bot policy to use for predators

    # inherited from other configs
    use_sensor_range = False                     # predators have sensing range
    comm_type = 'none'                          # predators have perfect communication
    comm_range = 0.75                           # communication range for perfect communication
    comm_noise = 0.5                           # noise in communication channel
    distance_start = 4.5                        # curriculum start value
    distance_end = 4.5                          # curriculum end value
    init_range_thresh = 1.0                     # percentage predators init outside sensing range


#--------------------------------------
# Helper functions
#--------------------------------------
def preprocess(args):
    config = define_configs(args)
    logger = define_logger(args, config.directory)

    logger.info("\n"*10)
    logger.info("cmd line: python " + " ".join(sys.argv) + "\n"*2)
    logger.info("Simulation configurations:\n" + "-"*30)
    config.show()
    logger.info("\n" * 5)
    return config, logger


def define_configs(args):
    if args.algorithm == 'ddpg_speed':
        config = Config_DDPG_Speed()
        from algorithms import ddpg_speed
        config.multiagent_fn = ddpg_speed.DDPG_Runner
    else:
        raise ValueError("Invalid choice of configuration")

    config = read_flags(config, args)

    # process seed (same seed for train every time)
    # if config.mode == 'test':
    #     config.seed = 612
    # else:
    seed_tmp = time.time()
    config.seed = int((seed_tmp - int(seed_tmp))*1e6) if args.seed is None else args.seed
    print('random seed = {}'.format(config.seed))

    # process directory
    config.directory = "results/{}_{}/{}".format(config.algorithm,
                                                config.env,
                                                'exp' + datetime.now().strftime("_%m_%d_%Y__%H_%M_%S"))

    print('directory = {}'.format(config.directory))
    create_exp_dir(config.directory, scripts_to_save=config.scripts)

    # process env
    envs_to_norm = ['Pendulum-v0']
    if config.env in envs_to_norm:
        config.normalize_env = True
    else:
        config.normalize_env = False

    particle_envs = ['simple_torus', 'simple_torus_blind']
    if config.env in particle_envs:
        config.particle_env = True
    else:
        config.particle_env = False

    comm_envs = []
    if config.env in comm_envs:
        config.comm_env = True
    else:
        config.comm_env = False 

    print("Using torch version: {}".format(torch.__version__))
    print('{} GPUs'.format(torch.cuda.device_count()))

    return config

def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    return config

def define_logger(args, directory):
    logFormatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logger")

    logger.setLevel(logging.INFO)


    fileHandler = logging.FileHandler("{0}/logger.log".format(directory))

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if args.verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    return logger