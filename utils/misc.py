import pickle
import os
import shutil
import sys
import random
import numpy as np
from scipy.spatial import distance
from collections import deque
import torch
import pandas as pd
import math 

if os.name == 'posix':
    import matplotlib
    # matplotlib.use("macOSX")
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


#---------------------------------------------------
# Directory (file creating, saving, loading, etc..)
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


def create_exp_dir(path, scripts_to_save=None):
    # results path
    if not os.path.exists(path):
        os.makedirs(path)

    # figures path
    fig_path = os.path.join(path, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # videos path
    vid_path = os.path.join(path, 'videos')
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # checkpoints path
    ch_path = os.path.join(path, 'checkpoints')
    if not os.path.exists(ch_path):
        os.makedirs(ch_path)

    # scripts path
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)

    # results path
    r_path = os.path.join(path, 'results')
    if not os.path.exists(r_path):
        os.makedirs(r_path)


def save_results(data, args):
    path = args.fp

    # construct file name
    fname = '{}_{}'.format(args.pred_policy, args.prey_policy)
    if args.bounds:
        fname += '_bounds'
        path = os.path.join(path, 'bounds')
    else:
        path = os.path.join(path, 'no_bounds')

    path = os.path.join(path, 'pred_vel_{}'.format(args.pred_vel))

    # make directory if it doesn't exist
    mkdir_p(path)

    # pickle dump
    print('path = {}'.format(os.path.join(path, '{}.pkl'.format(fname))))
    with open(os.path.join(path, '{}.pkl'.format(fname)), 'wb') as f:
        pickle.dump(data, f)

    f.close()

def save_baseline_data(data, args):
    path = args.fp

    # construct file name
    fname = '{}_{}'.format(args.pred_policy, args.prey_policy)
    path = os.path.join(path, 'no_bounds', 'radius_{}'.format(args.radius), 'world_size_{}'.format(args.world_size))
    path = os.path.join(path, 'pred_vel_{}'.format(args.pred_vel))

    # make directory if it doesn't exist
    mkdir_p(path)

    # pickle dump
    print('path = {}'.format(os.path.join(path, '{}.pkl'.format(fname))))
    with open(os.path.join(path, '{}.pkl'.format(fname)), 'wb') as f:
        pickle.dump(data, f)

    f.close()


def load_results(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

    
# ---------------------------------------------------
# Miscellaneous
# ---------------------------------------------------
# def parse_obs(obs, env, is_prey=False):
#     '''
#     Predator Observations:
#         agent velocity =        [0, 1]
#         agent position =        [2, 3]
#         landmark positions =    [4, 4+2L]
#         predator positions =    [4+2L+1, (4+2L+1) + 2(P-1)] 
#         prey positions =        [(4+2L+1) + 2(P-1) + 1, (4+2L+1) + 2(P-1) + 1 + 2E]
#         prey velocities =       [(4+2L+1) + 2(P-1) + 1 + 2E + 1, (4+2L+1) + 2(P-1) + 1 + 2E + 1 + 2E]
#     Prey Observations:
#         agent velocity =        [0, 1]
#         agent position =        [2, 3]
#         landmark positions =    [4, 4+2L]
#         predator positions =    [4+2L+1, (4+2L+1) + 2P] 
#         prey positions =        [(4+2L+1) + 2P + 1, (4+2L+1) + 2P + 1 + 2(E-1)]
#         prey velocities =       [(4+2L+1) + 2P + 1 + 2(E-1) + 1, (4+2L+1) + 2P + 1 + 2(E-1) + 1 + 2(E-1)]
#     '''

#     # gather info
#     n_predators = len([agent for agent in env.world.agents if agent.adversary])
#     n_prey = len([agent for agent in env.world.agents if not agent.adversary])
#     n_landmarks = len(env.world.landmarks)
#     key = env.env_key
#     w_size = env.world.size

#     # convert list observation into dict 
#     new_obs = {}

#     # parse position and velocity
#     new_obs['vel'] = obs[:2]
#     new_obs['pos'] = obs[2:4]

#     # parse landmarks
#     if n_landmarks > 0:
#         new_obs['landmarks'] = np.split(obs[4 : 4 + 2*n_landmarks], n_landmarks)
#     else:
#         new_obs['landmarks'] = []

#     if 'blind' in key and not is_prey:
#         # handle predator observations
#         new_obs['prey_pos'] = np.split(obs[4 + 2*n_landmarks: 4 + 2*n_landmarks + 2*n_prey], n_prey)
#     else:
#         # handle predator vs prey observations
#         if is_prey:
#              # parse for prey
#             new_obs['pred_pos'] = np.split(obs[4 + 2*n_landmarks: 4 + 2*n_landmarks + 2*n_predators], n_predators)
#             pp = new_obs['pos']
#         else:
#             # parse for predator
#             if n_predators > 1:
#                 new_obs['pred_pos'] = np.split(obs[4 + 2*n_landmarks: 4 + 2*n_landmarks + 2*(n_predators - 1)], n_predators-1)
#             new_obs['prey_pos'] = np.split(obs[4 + 2*n_landmarks + 2*(n_predators - 1) : 4 + 2*n_landmarks + 2*(n_predators - 1) + 2*n_prey], n_prey)
#             pp = new_obs['prey_pos'][0]
#             new_obs['rel_prey_pos'] = [toroidal_difference(p, pp, w_size) for p in new_obs['prey_pos']]

#         # relative positions
#         new_obs['rel_pos'] = toroidal_difference(new_obs['pos'], pp, w_size)
#         new_obs['rel_pred_pos'] = [toroidal_difference(p, pp, w_size) for p in new_obs['pred_pos']]

#     return new_obs
    
def parse_obs(obs, env, is_prey=False):
    '''
    Predator Observations:
        agent position =        [0, 1]
        landmark positions =    [2, 2+2L]
        predator positions =    [2+2L+1, (2+2L+1) + 2(P-1)] 
        prey positions =        [(2+2L+1) + 2(P-1) + 1, (2+2L+1) + 2(P-1) + 1 + 2E]

    Prey Observations:
        agent position =        [0, 1]
        landmark positions =    [2, 2+2L]
        predator positions =    [2+2L+1, (2+2L+1) + 2P] 
        prey positions =        [(2+2L+1) + 2P + 1, (2+2L+1) + 2P + 1 + 2(E-1)]
    '''

    # gather info
    n_predators = len([agent for agent in env.world.agents if agent.adversary])
    n_prey = len([agent for agent in env.world.agents if not agent.adversary])
    n_landmarks = len(env.world.landmarks)
    key = env.env_key
    w_size = env.world.size

    # convert list observation into dict 
    new_obs = {}

    # parse position
    new_obs['pos'] = obs[:2]

    # parse landmarks
    if n_landmarks > 0:
        new_obs['landmarks'] = np.split(obs[2 : 2 + 2*n_landmarks], n_landmarks)
    else:
        new_obs['landmarks'] = []

    if 'blind' in key and not is_prey:
        # handle predator observations
        new_obs['prey_pos'] = np.split(obs[2 + 2*n_landmarks: 2 + 2*n_landmarks + 2*n_prey], n_prey)
    else:
        # handle predator vs prey observations
        if is_prey:
             # parse for prey
            new_obs['pred_pos'] = np.split(obs[2 + 2*n_landmarks: 2 + 2*n_landmarks + 2*n_predators], n_predators)
            pp = new_obs['pos']
        else:
            # parse for predator
            if n_predators > 1:
                new_obs['pred_pos'] = np.split(obs[2 + 2*n_landmarks: 2 + 2*n_landmarks + 2*(n_predators - 1)], n_predators-1)
            new_obs['prey_pos'] = np.split(obs[2 + 2*n_landmarks + 2*(n_predators - 1) : 2 + 2*n_landmarks + 2*(n_predators - 1) + 2*n_prey], n_prey)
            pp = new_obs['prey_pos'][0]
            new_obs['rel_prey_pos'] = [toroidal_difference(p, pp, w_size) for p in new_obs['prey_pos']]

        # relative positions
        new_obs['rel_pos'] = toroidal_difference(new_obs['pos'], pp, w_size)
        new_obs['rel_pred_pos'] = [toroidal_difference(p, pp, w_size) for p in new_obs['pred_pos']]

    return new_obs

def toroidal_position(pos, size):
    # compute position in toroidal world (wraparound)
    pos %= size
    return pos
    
def one_hot_encode(num, size):
    return np.eye(size)[num]


def compute_distance(d1, d2, bounds, size):
    if bounds:
        return distance.euclidean(d1, d2)
    else:
        return toroidal_distance(d1, d2, size)


def toroidal_distance(d1, d2, size):
    dx = abs(d1[0] - d2[0])
    dy = abs(d1[1] - d2[1])

    if dx > size/2:
        dx = size - dx
    if dy > size/2:
        dy = size - dy

    return np.sqrt(dx*dx + dy*dy)


def compute_difference(d1, d2, bounds, size):
    if bounds:
        return d1 - d2
    else:
        return toroidal_difference(d1, d2, size)


def toroidal_difference(d1, d2, size):
    dx = d1[0] - d2[0]
    dy = d1[1] - d2[1]

    # adjust dx
    if dx > size/2:
        dx = dx - size
    elif dx < -size/2:
        dx = dx + size

    # adjust dy
    if dy > size/2:
        dy = dy - size
    elif dy < -size/2:
        dy = dy + size


    return np.array([dx, dy])


def compute_angle(origin, pt1, pt2, degrees=False):
    # vectors between 2 points and origin
    o_pt1 = pt1 - origin
    o_pt2 = pt2 - origin

    # angle between two points
    cos_angle = np.dot(o_pt1, o_pt2) / (np.linalg.norm(o_pt1) * np.linalg.norm(o_pt2))
    angle = np.arccos(cos_angle)

    if degrees:
        angle = np.degrees(angle)

    return angle


def repulsive_force(q, q_obst, bounds, world_size, k_rep=0.5, thresh=1.0):
    if compute_distance(q, q_obst, bounds, world_size) < thresh:
        f = k_rep * (1/compute_distance(q, q_obst, bounds, world_size) - 1 / thresh) * (1/compute_distance(q, q_obst, bounds, world_size)**2) * (compute_difference(q, q_obst, bounds, world_size) / compute_distance(q, q_obst, bounds, world_size))
        return f
    else:
        return np.zeros(2)


def attractive_force(q, q_goal, bounds, world_size, k_att=1.5, thresh=2.0):
    # if compute_distance(q, q_goal, bounds, world_size) < thresh:
    f = -k_att * compute_difference(q, q_goal, bounds, world_size)
    if not all(f == 0.0):
        return f / np.linalg.norm(f)
    else:
        return f

def cosine_potential(rs, thetas):
    A = np.sum([(1/r_i) * math.sin(t_i) for r_i, t_i in zip(rs, thetas)]) # y-coords
    B = np.sum([(1/r_i) * math.cos(t_i) for r_i, t_i in zip(rs, thetas)]) # x-coords
    theta = math.atan2(-A, -B)

    return theta


def cosine_cost(theta, rs, pred_thetas):
    cost = np.sum([(1/r_i) * math.cos(theta - t_i) for r_i, t_i in zip(rs, pred_thetas)])

    return cost


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c




