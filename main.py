import argparse
import os
import numpy as np
import torch
from configs import preprocess
from utils.make_env import make_env

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(args):
    # process args
    config, logger = preprocess(args)


    # seeds
    print('seed = {}'.format(config.seed))
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # init env and runner
    env = make_env(config)
    runner = config.multiagent_fn(env, config)

    if config.mode == 'train':
        #TODO: Save this
        success_ratio = runner.train()
    else:
        if config.checkpoint_path is not None:
            runner.test()
        else:
            raise ValueError('Path to checkpoint must be provided to test policy!')

    env.close()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='reinforce')
    parser.add_argument('--gamma', type=float, default=None, help='discount factor')
    parser.add_argument('--tau', type=float, default=None, help='polyak averaging factor')
    parser.add_argument('--lr', type=float, default=None, help='learning rate ')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--buffer_length', type=int, default=None, help='replay buffer length')
    parser.add_argument('--n_steps', type=int, default=167, help='number of steps to run per epoch')
    parser.add_argument('--n_epochs', type=int, default=42000, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--log_interval', type=int, default=None, help='training logs update interval')
    parser.add_argument('--n_threads', type=int, default=None, help='number of threads for trajectory rollouts')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--collaborative', action='store_true')
    parser.add_argument('--world_size', type=float, default = 2.0)
    parser.add_argument('--lambda_coeff', type=float, default = 0.5)
    parser.add_argument('--decay', type=int, default=5000)
    parser.add_argument('--equivariant', action='store_true')
    parser.add_argument('--test_predator', type=str, default="greedy")
    parser.add_argument('--nb_pred', type=int, default=3)
    parser.add_argument('--nb_prey', type=int, default=1)

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)
