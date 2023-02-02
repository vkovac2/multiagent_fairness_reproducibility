import argparse
import os
import pickle
import math
import numpy as np
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns

# hack to get relative package references to work
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from utils.misc import *

N_VEC_OUTCOMES = 16 
N_SUM_OUTCOMES = 5

WORLD_SIZE = 2.0
PRED_SIZE = 0.075
PREY_SIZE = 0.05

STEPS = 500



def bin_angles(vecs, sums, m=None):
    # binning function that creates a matrix that counts co-occurrences 
    total_1, total_2, total_3 = 0, 0, 0
    if m is None:
        m = np.zeros((N_VEC_OUTCOMES, N_SUM_OUTCOMES))
    for a, b in zip(vecs, sums):
        m[a][b] += 1

    return m

def compute_joint(results_dict):
    joint = bin_angles(results_dict['reward_vectors'], results_dict['reward_sums'])
    joint /= np.sum(joint)  # normalize counts into a probability distribution

    # convert to dataframe
    h = 0.0
    p_vecs, p_sums, probs = [], [], []
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            # print('p_vec={}, p_sum={}, prob = {}'.format(i, j, joint[i][j]))
            p_vecs.append(i)
            p_sums.append(j)
            probs.append(joint[i][j])

            if joint[i][j] > 0.0:
                h += joint[i][j]*math.log(joint[i][j], 2)

    return pd.DataFrame({'reward_vectors':p_vecs, 'reward_sums':p_sums, 'prob':probs})

def compute_marginals(joint, key):
    probs = list(joint.groupby(key).sum()['prob'])
    return probs


def compute_conditional(joint, target, cond, val):
    '''
    computes P(target|cond=val)
    '''
    # remove irrelevant rows for conditioning variable
    conditional = joint[joint[cond]==val]

    # sum out extra variable
    conditional = conditional.groupby(target).sum()

    # normalize sums (replace NaN with 0.0 if needed)
    conditional['prob'] = conditional['prob'] / conditional['prob'].sum()
    conditional['prob'] = conditional['prob'].fillna(0.0)

    # index is target values, so can use standard indexing (no column needed)
    return conditional

def compute_entropy(p):
    h = 0.0
    for i in range(len(p)):
        if p[i] > 0.0:
            h += p[i]*math.log(p[i], 2)
    return -h

def compute_conditional_entropy(conditionals, sum_marginals):
    h = 0.0
    for i in range(N_SUM_OUTCOMES):
        h_i = 0.0
        # print('P(R|A={}) = {}'.format(i, conditionals['P(R|A={})'.format(i)]))
        for j in range(N_VEC_OUTCOMES):
            p_ra = conditionals['P(R|A={})'.format(i)][j]
            # print('P(R={}|A={}) = {}'.format(j, i, p_ra))
            if p_ra > 0.0:
                h_i += p_ra*math.log(p_ra, 2)
        # print('P(A={}) = {}'.format(i, sum_marginals[i]))
        # print('Entropy = {}'.format(h_i))
        # print('P(A={}) * \sum P(R|A) * log P(R|A) = {}'.format(i, sum_marginals[i] * h_i))
        h += sum_marginals[i] * h_i
    return -h


'''
Possible Reward Vector (i.e. Identity) Assignments R = [r1, r2, r3, r4]:
    0: [0,0,0,0]
    1: [0,0,0,1]
    2: [0,0,1,0]
    3: [0,1,0,0]
    4: [1,0,0,0]
    5: [0,0,1,1]
    6: [0,1,0,1]
    7: [0,1,1,0]
    8: [1,0,0,1]
    9: [1,0,1,0]
    10: [1,1,0,0]
    11: [0,1,1,1]
    12: [1,0,1,1]
    13: [1,1,0,1]
    14: [1,1,1,0]
    15: [1,1,1,1]
    
Possible Reward Outcomes:
    0: 0*r --> nobody captures
    1: 1*r --> one predator captures
    2: 2*r --> two predators capture
    3: 3*r --> three predators capture
    4: 4*r --> four predators capture
'''

# hard-coded way to assign index to reward vector --> could be cleaner
def vec_to_idx(vec):
    if np.array_equal(vec, np.zeros(4)):
        return 0
    elif np.array_equal(vec, np.array([0,0,0,1])):
        return 1
    elif np.array_equal(vec, np.array([0,0,1,0])):
        return 2
    elif np.array_equal(vec, np.array([0,1,0,0])):
        return 3
    elif np.array_equal(vec, np.array([1,0,0,0])):
        return 4
    elif np.array_equal(vec, np.array([0,0,1,1])):
        return 5
    elif np.array_equal(vec, np.array([0,1,0,1])):
        return 6
    elif np.array_equal(vec, np.array([0,1,1,0])):
        return 7
    elif np.array_equal(vec, np.array([1,0,0,1])):
        return 8
    elif np.array_equal(vec, np.array([1,0,1,0])):
        return 9
    elif np.array_equal(vec, np.array([1,1,0,0])):
        return 10
    elif np.array_equal(vec, np.array([0,1,1,1])):
        return 11
    elif np.array_equal(vec, np.array([1,0,1,1])):
        return 12
    elif np.array_equal(vec, np.array([1,1,0,1])):
        return 13
    elif np.array_equal(vec, np.array([1,1,1,0])):
        return 14
    elif np.array_equal(vec, np.ones(4)):
        return 15
    else:
        raise ValueError('Incorrect reward vector!')

def idx_to_vec(idx):
    if idx == 0:
        return '[0, 0, 0, 0]'
    elif idx == 1:
        return '[0, 0, 0, r]'
    elif idx == 2:
        return '[0, 0, r, 0]'
    elif idx == 3:
        return '[0, r, 0, 0]'
    elif idx == 4:
        return '[r, 0, 0, 0]'
    elif idx == 5:
        return '[0, 0, r, r]'
    elif idx == 6:
        return '[0, r, 0, r]'
    elif idx == 7:
        return '[0, r, r, 0]'
    elif idx == 8:
        return '[r, 0, 0, r]'
    elif idx == 9:
        return '[r, 0, r, 0]'
    elif idx == 10:
        return '[r, r, 0, 0]'
    elif idx == 11:
        return '[0, r, r, r]'
    elif idx == 12:
        return '[r, 0, r, r]'
    elif idx == 13:
        return '[r, r, 0, r]'
    elif idx == 14:
        return '[r, r, r, 0]'
    elif idx == 15:
        return '[r, r, r, r]'
    else:
        raise ValueError('Incorrect reward index!')
    
def compute_results(path, steps = STEPS):
    if not os.path.exists(path):
        print("File path " + path + " does not exist!")
        exit()    
        # return

    file = open(path, "rb")
    trajectories = pickle.load(file)
    file.close()
    
    p_keys = sorted(trajectories['positions'][0].keys())[:-1]
    vec_outcomes, sum_outcomes = [], []
    num_captures = 0
    for i in range(len(trajectories['positions'])):
        reward_vec = np.zeros(len(p_keys), dtype=np.int32)
        rew = -50.0
        if len(trajectories['positions'][i][p_keys[0]]) < steps:
            num_captures += 1
            rew = 50.0 - len(trajectories['positions'][i][p_keys[0]]) * 0.1
            for j, key in enumerate(sorted(p_keys)):
                # compute reward vector    
                pred_pos = trajectories['positions'][i][key][-1]
                if 'prey' in trajectories['positions'][i].keys():
                    prey_pos = trajectories['positions'][i]['prey'][-1]
                else:
                    prey_pos = trajectories['positions'][i]['prey1'][-1]
                    
                dist = toroidal_distance(pred_pos, prey_pos, WORLD_SIZE)
                if dist < PRED_SIZE + PREY_SIZE:
                    reward_vec[j] = 1

        vec_outcomes.append(vec_to_idx(reward_vec))
        sum_outcomes.append(np.sum(reward_vec))

    # store rewards in dict
    rewards = {
        'reward_vectors' : vec_outcomes,
        'reward_sums' : sum_outcomes
    }

    # print(vec_outcomes)

    # compute joint
    joint = compute_joint(rewards)

    # reward sum marginals
    sum_marginals = compute_marginals(joint, 'reward_sums')

    # reward vector conditioned on sum
    conditionals = {}
    for val in joint['reward_sums'].unique():
        cond = compute_conditional(joint, 'reward_vectors', 'reward_sums', val)
        conditionals['P(R|A={})'.format(val)] = list(cond['prob'])
    
    # I(R;A)
    h_cond = compute_conditional_entropy(conditionals, sum_marginals)

    # compare to uniform conditional entropy
    uniform_conditionals = {
        'P(R|A=0)' : [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'P(R|A=1)' : [0.0, 1/4, 1/4, 1/4, 1/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'P(R|A=2)' : [0.0, 0.0, 0.0, 0.0, 0.0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0.0, 0.0, 0.0, 0.0, 0.0],
        'P(R|A=3)' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1/4, 1/4, 1/4, 1/4, 0.0],
        'P(R|A=4)' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    h_uniform = compute_conditional_entropy(uniform_conditionals, sum_marginals)
    i_ra = h_uniform - h_cond

    results = {
        'reward' : rew,
        'capture_success' : num_captures/len(sum_outcomes),
        'info' : i_ra
    }

    return rewards, results


TEST_VELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

def main(config):

    # keys: 
    # velocity [float]
    # strategy [string]
    # collaborative [string]
    # equivariance [string] 
    # lamda [float]

    # data
    # capture_success: percentage of captures
    # reward: reward of experiment
    # info: distribution difference

    try:
        os.makedirs("plots4")
    except FileExistsError:
        # directory already exists
        pass
    

    #PLOT1
    if config.plot == 1 or config.plot == 0:
        print("Making plot 1...")
    
        x_pos = TEST_VELS

        path = os.path.join(config.fp, 'ddpg_4_agents_no_collab_no_equivar_vel_') 
        individual = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]
        
        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_') 
        shared = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]
        bar_width = 0.04

        x_pos.reverse()

        fig = plt.figure()
        plt.bar(x_pos, shared, bar_width, color='b', label="shared")
        plt.bar([x + bar_width for x in x_pos], individual, bar_width, color='r', label="individual")
        plt.xticks([x + bar_width/2 for x in x_pos], sorted(x_pos) )
        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('Capture Success %', weight="bold")
        plt.title('Mutual vs Individual Reward (Capture Success)', weight="bold")
        plt.legend(loc="upper right")

        plt.savefig('plots4/fig1.png')
        print("Done")
    

    if config.plot == 2 or config.plot == 0:
        print("Making plot 2...")

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_1.1.pkl')
        rewards, results = compute_results(path)
            
        #No Eq
        fig = plt.figure(figsize=(17,5))
        x_pos = [i for i in range(16)]
        vals = [rewards['reward_vectors'].count(i) / len(rewards['reward_vectors']) for i in range(16)]

        bar_width = 0.4


        plt.bar(x_pos, vals, bar_width, color='g')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)', weight="bold")
        plt.title('No Equivariance', weight="bold")
        plt.savefig('plots4/distrib_no_eq.png')



        # Eq
        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_1.1.pkl')
        rewards, results = compute_results(path)

        fig = plt.figure(figsize=(17,5))
        x_pos = [i for i in range(16)]
        vals2 = [rewards['reward_vectors'].count(i) / len(rewards['reward_vectors']) for i in range(16)]
        bar_width = 0.4

        
        plt.bar(x_pos, vals2, bar_width, color='b')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)', weight="bold")
        plt.title('Fair-E', weight="bold")
        plt.savefig('plots4/distrib_eq.png')


        fig = plt.figure(figsize=(17,8))
        ax1 = fig.add_subplot(2,1,1)
        ax1.bar(x_pos, vals, bar_width, color='g')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)', weight="bold")
        plt.title('No Equivariance', weight="bold")

        ax2 = fig.add_subplot(2,1,2)
        ax2.bar(x_pos, vals2, bar_width, color='b')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)', weight="bold")
        plt.title('Fair-E', weight="bold")

        plt.subplots_adjust(
                    # bottom=0.5,
                    # top=0.9,
                    wspace=0.4,
                    hspace=0.8)

        # Save the full figure...
        fig.savefig('plots4/fig2.png')
        print("Done")


    if config.plot == 3 or config.plot == 0:
        print("Making plot 3...")


        #FAIRNESS
        fig = plt.figure()
        x_pos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_') 
        one = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        # path = os.path.join(config.fp, 'greedy_vel_') 
        # two = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_no_collab_no_equivar_vel_') 
        three = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_') 
        four = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]
        

        plt.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
        # plt.plot(x_pos, two, ls='--', marker='.', label='Greedy')
        plt.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
        plt.plot(x_pos, four, ls='--', marker='*', label='Fair-E')

        plt.gca().invert_xaxis()
        
        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('I(R,Z)', weight="bold")
        plt.title('Team Fairness', weight="bold")
        plt.legend()
        plt.savefig('plots4/fairness.png')

        #UTIL
        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_') 
        one = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        # path = os.path.join(config.fp, 'greedy_vel_') 
        # two = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_no_collab_no_equivar_vel_') 
        three = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_') 
        four = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        fig = plt.figure()
        plt.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
        # plt.plot(x_pos, two, ls='--', marker='.', label='Greedy')
        plt.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
        plt.plot(x_pos, four, ls='--', marker='*', label='Fair-E')
        
        plt.gca().invert_xaxis()

        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('Capture Success %', weight="bold")
        plt.title('Team Utility', weight="bold")
        plt.legend()
        plt.savefig('plots4/utility.png')



        #Fairness - Util
        fig = plt.figure(figsize=(17,7))
        plt.rcParams.update({'font.size': 18})
        ax1 = fig.add_subplot(1,2,1)
        x_pos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        
        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_') 
        one = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        # path = os.path.join(config.fp, 'greedy_vel_') 
        # two = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_no_collab_no_equivar_vel_') 
        three = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_') 
        four = [compute_results(path + str(x) + '.pkl')[1]['info'] for x in x_pos]
        

        ax1.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
        # ax1.plot(x_pos, two, ls='--', marker='.', label='Greedy')
        ax1.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
        ax1.plot(x_pos, four, ls='--', marker='*', label='Fair-E')

        ax1.invert_xaxis()
        
        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('I(R,Z)', weight="bold")
        plt.title('Team Fairness', weight="bold")
        plt.legend()

        ax2 = fig.add_subplot(1,2,2)
        path = os.path.join(config.fp, 'ddpg_4_agents_collab_no_equivar_vel_') 
        one = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        # path = os.path.join(config.fp, 'greedy_vel_') 
        # two = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_no_collab_no_equivar_vel_') 
        three = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_') 
        four = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in x_pos]

        ax2.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
        # ax2.plot(x_pos, two, ls='--', marker='.', label='Greedy')
        ax2.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
        ax2.plot(x_pos, four, ls='--', marker='*', label='Fair-E')

        ax2.invert_xaxis()

        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('Capture Success %', weight="bold")
        plt.title('Team Utility', weight="bold")
        plt.legend()

        fig.savefig('plots4/fig3.png')
        print("Done")

    

    if config.plot == 4 or config.plot == 0:
        print("Making plot 4...")

        LAMDAS2 = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
        vels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

        fig = plt.figure()

        path = os.path.join(config.fp, 'ddpg_4_agents_collab_equivar_vel_') 
        # points = [compute_results(path + str(x) + '.pkl', 167)[1]['capture_success'] for x in vels]
        points = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in vels]

        plt.plot(vels ,points, linewidth=4.0, c='k')

        palette = plt.get_cmap('tab10')
        colors = [palette(i) for i in np.arange(len(LAMDAS2))]
        patches = [mpatches.Patch(color="k", label='Fair-E')]
        for i,l in enumerate(LAMDAS2):
            path = os.path.join(config.fp, 'ddpg_4_agents_collab_lambda_' + str(l) + '_vel_')
            # points = [compute_results(path + str(x) + '.pkl', 167)[1]['capture_success'] for x in vels]
            points = [compute_results(path + str(x) + '.pkl')[1]['capture_success'] for x in vels]

            plt.plot(vels ,points, c=colors[i], linewidth=2.0)
        
            patches.append(mpatches.Patch(color=colors[i], label='\u03BB = {}'.format(l)))

            
        plt.xlim(max(vels)+0.05, min(vels)-0.05)
        plt.xlabel('Pursuer Velocity', weight="bold")
        plt.ylabel('Capture Success %', weight="bold")
        plt.title('Fair-ER Train Performance (Capture Success)', weight="bold")
        plt.legend(handles=patches)

        fig.savefig('plots4/fig4.png')
        print("Done")




    if config.plot == 5 or config.plot == 0:

        # plot utility vs. lambda
        print("Making plot 5...")

        LAMDAS2 = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
        vels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fig = plt.figure(figsize=(17,20))
        # plt.rcParams.update({'font.size': 18})
        sns.set(style="white", font_scale=1.5)
        sns.set_context("poster")
        sns.set_palette("Greys_r")
        plt.rcParams["axes.grid"] = True


        for num, val in enumerate(vels):
            ax1 = fig.add_subplot(3,2,num+1)
            info_ys = []
            success_ys = []
            for x in LAMDAS2:
                path = os.path.join(config.fp, 'ddpg_4_agents_collab_lambda_' + str(x) + '_vel_' + str(val) + '.pkl')
                rewards, results = compute_results(path)

                info_ys.append(results['info'])
                success_ys.append(results['capture_success'])
            
            # palette = plt.get_cmap('tab10')
            # colors = [palette(i) for i in np.arange(len(info_ys))]
            # plt.ylim(-0.05, 1.1)
            # plt.xlim(-0.01, np.max(info_ys) + 0.05)
            # plt.scatter(info_ys, success_ys, s=14**2, c=colors, marker='o', edgecolors='black', linewidths=1.2, zorder=10)
            # patches = []

            plt.subplots_adjust(
                    # bottom=0.5,
                    # top=0.9,
                    wspace=0.4,
                    hspace=0.8)

            sns.lineplot(x=info_ys, y=success_ys, style=True, dashes=[(2,2)]*len(info_ys), markers=False, legend=False)
            palette = plt.get_cmap('tab10')
            colors = [palette(i) for i in np.arange(len(info_ys))]
            plt.scatter(info_ys, success_ys, s=14**2, c=colors, marker='o', edgecolors='black', linewidths=1.2, zorder=10)
            # plt.errorbar(info_ys, success_ys, linestyle="None", ecolor='black', elinewidth=1.5)
            plt.ylim(-0.05, 1.1)
            plt.xlim(-0.01, np.max(info_ys) + 0.05)
            patches = []
            for i,k in enumerate(LAMDAS2):
                patches.append(mpatches.Patch(color=colors[i], label='\u03BB = {}'.format(k)))

            
            if num >= 4:
                plt.xlabel('I(R;Z)', weight="bold")

            if num % 2 == 0:
                plt.ylabel('Capture Success %', weight="bold")


            plt.title('Velocity ' + str(val), weight="bold")

            plt.legend(handles=patches)

        



        plt.savefig('plots4/fig5.png')
        print("Done")



    # sns.set(style="white", font_scale=1.5)
    # sns.set_context("poster")
    # sns.set_palette("Greys_r")
    # plt.rcParams["axes.grid"] = True
    # sns.lineplot(x=info_ys, y=success_ys, style=True, dashes=[(2,2)]*len(info_ys), markers=False, legend=False)
    # palette = plt.get_cmap('tab10')
    # colors = [palette(i) for i in np.arange(len(info_ys))]
    # plt.scatter(info_ys, success_ys, s=14**2, c=colors, marker='o', edgecolors='black', linewidths=1.2, zorder=10)
    # plt.errorbar(info_ys, success_ys, yerr=success_y_errs, linestyle="None", ecolor='black', elinewidth=1.5)
    # plt.ylim(-0.05, 1.1)
    # plt.xlim(-0.01, np.max(info_ys) + 0.05)
    # patches = []
    # for i, k in enumerate(sorted(plot_data[key].keys())):
    #     patches.append(mpatches.Patch(color=colors[i], label='\u03BB = {}'.format(k.split('_')[-1])))
    # plt.legend(handles=patches, loc='lower right', prop={'size': 14})
    # plt.savefig('{}_info_vs_success.png'.format(key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--fp', type=str, default=None, help='path to load/save')
    parser.add_argument('--plot', type=int, default=0, help='plot to make')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)