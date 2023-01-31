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

N_VEC_OUTCOMES = 8 
N_SUM_OUTCOMES = 4

WORLD_SIZE = 2.0
PRED_SIZE = 0.075
PREY_SIZE = 0.05

STEPS = 167



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

def plot_bars(measures, bounds=None, labels=None, name=None, center_spines=False, y_min=0.0, y_max=1.1):
    # hatches = [''] * len(measures)
    # if len(measures) > 4:
    #     hatches[3] = '/'
    #     hatches[4] = 'x'

    palette = plt.get_cmap('tab20')
    fig = plt.figure()
    plt.ylim(y_min, y_max)
    for i in range(len(measures)):
        if bounds:
            # plt.bar(i, bounds[i], label='_nolegend_', color=palette(i*2+1), alpha=0.25, edgecolor='black', linestyle = '--', linewidth=1.75)
            plt.bar(i, bounds[i], label='_nolegend_', color=palette(i), alpha=0.25, edgecolor='black', linestyle = '--', linewidth=1.75)
        # plt.bar(i, measures[i], color=palette(i), alpha=0.75, hatch=hatches[i])
        plt.bar(i, measures[i], color=palette(i), alpha=0.75)

    if labels:
        plt.xticks(np.arange(len(measures)), labels, fontsize=6)

    # formatting
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if center_spines:
        plt.ylim(-2.0, 2.0)
    
        # spines
        ax.spines['bottom'].set_position('zero')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        # ticks
        ax.set_xticklabels([])
        label_offset = 0.05
        for label, (x, y) in zip(labels, enumerate(measures)):
            if y > 0.0:
                label_y = -label_offset
            else:
                label_y = y - label_offset
            ax.text(x, label_y, label, ha="center", va="top")

    plt.tight_layout()
    if name:
        plt.savefig('{}.png'.format(name))
    else:
        plt.savefig('figure.png')


'''
Possible Reward Vector (i.e. Identity) Assignments R = [r1, r2, r3]:
    0: [0,0,0]
    1: [0,0,1]
    2: [0,1,0]
    3: [1,0,0]
    4: [0,1,1]
    5: [1,0,1]
    6: [1,1,0]
    7: [1,1,1]
Possible Reward Outcomes:
    0: 0*r --> nobody captures
    1: 1*r --> one predator captures
    2: 2*r --> two predators capture
    3: 3*r --> three predators capture
'''

# hard-coded way to assign index to reward vector --> could be cleaner
def vec_to_idx(vec):
    if np.array_equal(vec, np.zeros(3)):
        return 0
    elif np.array_equal(vec, np.array([0, 0, 1])):
        return 1
    elif np.array_equal(vec, np.array([0, 1, 0])):
        return 2
    elif np.array_equal(vec, np.array([1, 0, 0])):
        return 3
    elif np.array_equal(vec, np.array([0, 1, 1])):
        return 4
    elif np.array_equal(vec, np.array([1, 0, 1])):
        return 5
    elif np.array_equal(vec, np.array([1, 1, 0])):
        return 6
    elif np.array_equal(vec, np.ones(3)):
        return 7
    else:
        raise ValueError('Incorrect reward vector!')

def idx_to_vec(idx):
    if idx == 0:
        return '[0, 0, 0]'
    elif idx == 1:
        return '[0, 0, r]'
    elif idx == 2:
        return '[0, r, 0]'
    elif idx == 3:
        return '[r, 0, 0]'
    elif idx == 4:
        return '[0, r, r]'
    elif idx == 5:
        return '[r, 0, r]'
    elif idx == 6:
        return '[r, r, 0]'
    elif idx == 7:
        return '[r, r, r]'
    else:
        raise ValueError('Incorrect reward index!')

TEST_VELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# TEST_VELS = [1.2, 1.1, 1.0, 0.9, 0.7, 0.6, 0.5]
INVALID_STRATS = ['lambda_0.4', 'lambda_0.7']

STRATS = [ "ddpg_symmetric", "ddpg_fair", "greedy"]
COLLABS = [ "collab", "no_collab"]
EQUIVARS = [ "equivar", "no_equivar"]
LAMDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99999]

def main(config):
    # load trajectories

    paths = []
    plot_data = {}

    #for each velocity    
    for vel in sorted(TEST_VELS):
        vel_results = {}
        #for each algorithm
        for strat in sorted(STRATS):
            # path = os.path.join(config.fp, "")
            # path += strat
            vel_results[strat] = {}

            #for colab or not
            for collab in COLLABS:
                # path = os.path.join(path, '_', collab)
                # path += '_' + collab
                
                vel_results[strat][collab] = {}

                #for equivariance or not
                for equivar in EQUIVARS:
                    # path = os.path.join(path, '_', equivar)
                    # path += '_' + equivar
                    
                    vel_results[strat][collab][equivar] = {}

                    #for every lamda (other than ddpg_fair, everything)
                    for lamda in LAMDAS:

                        if strat != "ddpg_fair" and lamda!=0:
                            # lamda = 0
                            continue
                        
                        vel_results[strat][collab][equivar][lamda] = {}                        

                        path = os.path.join(config.fp, "")
                        path += strat + '_' + collab
                        
                        #add equivariance to path
                        if strat != "ddpg_fair":
                            path += '_' + equivar
                        
                        #add lamda to path
                        if strat == "ddpg_fair":
                            path += '_' + 'lambda' + '_' + str(lamda)
                        elif strat == "greedy":
                            # greedy paths have no lamda and no equivariance
                            path = os.path.join(config.fp, "") + strat

                        path += '_' + 'vel' + '_' + str(vel) + '.pkl'

                        if not os.path.exists(path):
                            # print("BAD Path is: ", path)    
                            continue

                        file = open(path, "rb")
                        trajectories = pickle.load(file)
                        file.close()
                        
                        p_keys = sorted(trajectories['positions'][0].keys())[:-1]
                        vec_outcomes, sum_outcomes = [], []
                        num_captures = 0
                        for i in range(len(trajectories['positions'])):
                            reward_vec = np.zeros(len(p_keys), dtype=np.int)
                            rew = -50.0
                            if len(trajectories['positions'][i][p_keys[0]]) < STEPS:
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

                        # store results in dict
                        results = {
                            'reward_vectors' : vec_outcomes,
                            'reward_sums' : sum_outcomes
                        }

                        # compute joint
                        joint = compute_joint(results)

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
                            'P(R|A=0)' : [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            'P(R|A=1)' : [0.0, 1/3, 1/3, 1/3, 0.0, 0.0, 0.0, 0.0],
                            'P(R|A=2)' : [0.0, 0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 0.0],
                            'P(R|A=3)' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        }
                        h_uniform = compute_conditional_entropy(uniform_conditionals, sum_marginals)
                        i_ra = h_uniform - h_cond

                        vel_results[strat][collab][equivar][lamda] = {
                            'reward' : rew,
                            'capture_success' : num_captures/len(sum_outcomes),
                            'info' : i_ra
                        }

                        # print("Path is: ", path, num_captures/len(sum_outcomes))
                        paths.append((path, num_captures/len(sum_outcomes)))

        # store results for strategy
        plot_data[vel] = vel_results

    
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
    

    #PLOT1
    fig = plt.figure()
    x_pos = TEST_VELS

    individual = [plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos]
    shared = [plot_data[x]['ddpg_symmetric']['collab']['no_equivar'][0]['capture_success'] for x in x_pos]
    bar_width = 0.04

    x_pos.reverse()
    
    plt.bar(x_pos, shared, bar_width, color='b', label="shared")
    plt.bar([x + bar_width for x in x_pos], individual, bar_width, color='r', label="individual")
    plt.xticks([x + bar_width/2 for x in x_pos], sorted(x_pos) )
    plt.xlabel('Pursuer Velocity')
    plt.ylabel('Capture Success %')
    plt.title('Mutual vs Individual Reward (Capture Success)')
    plt.legend(loc="upper right")

    plt.savefig('success_vs_velocity.png')
    
    #-----------------------

    path = "results_new/ddpg_symmetric_collab_no_equivar_vel_1.1.pkl"
    if os.path.exists(path):
        
        file = open(path, "rb")
        trajectories = pickle.load(file)
        file.close()
        
        p_keys = sorted(trajectories['positions'][0].keys())[:-1]
        vec_outcomes, sum_outcomes = [], []
        num_captures = 0
        for i in range(len(trajectories['positions'])):
            reward_vec = np.zeros(len(p_keys), dtype=np.int)
            rew = -50.0
            if len(trajectories['positions'][i][p_keys[0]]) < STEPS:
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


        # store results in dict
        results = {
            'reward_vectors' : vec_outcomes,
            'reward_sums' : sum_outcomes
        }
        
        
        
        #PLOT2
        fig = plt.figure()
        x_pos = [i for i in range(8)]
        vals = [results['reward_vectors'].count(i) / len(results['reward_vectors']) for i in range(8)]

        bar_width = 0.4

    
        plt.bar(x_pos, vals, bar_width, color='g')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)')
        plt.title('No Equivariance')
        plt.savefig('individual.png')
    # #-----------------------

    path = "results_new/ddpg_symmetric_collab_equivar_vel_1.1.pkl"
    # path = "results_new/trajectories.pkl"
    if os.path.exists(path):
        file = open(path, "rb")
        trajectories = pickle.load(file)
        file.close()
        
        p_keys = sorted(trajectories['positions'][0].keys())[:-1]
        vec_outcomes, sum_outcomes = [], []
        num_captures = 0
        for i in range(len(trajectories['positions'])):
            reward_vec = np.zeros(len(p_keys), dtype=np.int)
            rew = -50.0
            if len(trajectories['positions'][i][p_keys[0]]) < STEPS:
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

        # store results in dict
        results2 = {
            'reward_vectors' : vec_outcomes,
            'reward_sums' : sum_outcomes
        }

        #PLOT3
        fig = plt.figure()
        x_pos = [i for i in range(8)]
        vals2 = [results2['reward_vectors'].count(i) / len(results2['reward_vectors']) for i in range(8)]
        bar_width = 0.4

        
        plt.bar(x_pos, vals2, bar_width, color='b')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)')
        plt.title('Fair-E')
        plt.savefig('shared.png')


        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.bar(x_pos, vals, bar_width, color='g')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)')
        plt.title('No Equivariance')

        ax2 = fig.add_subplot(2,1,2)
        ax2.bar(x_pos, vals2, bar_width, color='b')
        plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
        plt.ylabel('P(r)')
        plt.title('Fair-E')

        plt.subplots_adjust(
                    # bottom=0.5,
                    # top=0.9,
                    wspace=0.4,
                    hspace=0.8)

        # Save the full figure...
        fig.savefig('joined.png')


    #-----------------------


    #PLOT4
    fig = plt.figure()
    x_pos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    one = [plot_data[x]['ddpg_symmetric']['collab']['no_equivar'][0]['info'] for x in x_pos]
    two = [plot_data[x]['greedy']['no_collab']['no_equivar'][0]['info'] for x in x_pos]
    three = [plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['info'] for x in x_pos]
    four = [plot_data[x]['ddpg_symmetric']['collab']['equivar'][0]['info'] for x in x_pos]
    

    plt.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
    plt.plot(x_pos, two, ls='--', marker='.', label='Greedy')
    plt.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
    plt.plot(x_pos, four, ls='--', marker='*', label='Fair-E')

    plt.gca().invert_xaxis()
    
    plt.xlabel('Pursuer Velocity')
    plt.ylabel('I(R,Z)')
    plt.title('Team Fairness')
    plt.legend()
    plt.savefig('fairness.png')

    #PLOT5
    fig = plt.figure()
    plt.plot(x_pos, [plot_data[x]['ddpg_symmetric']['collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='+', label='No Equivariance')
    plt.plot(x_pos, [plot_data[x]['greedy']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='.', label='Greedy')
    plt.plot(x_pos, [plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='o', label='Individual Reward')
    plt.plot(x_pos, [plot_data[x]['ddpg_symmetric']['collab']['equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='*', label='Fair-E')
    
    plt.gca().invert_xaxis()

    plt.xlabel('Pursuer Velocity')
    plt.ylabel('Capture Success %')
    plt.title('Team Utility')
    plt.legend()
    plt.savefig('utility.png')



    #PLOTS 4-5
    fig = plt.figure(figsize=(17,7))
    plt.rcParams.update({'font.size': 18})
    ax1 = fig.add_subplot(1,2,1)
    x_pos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    one = [plot_data[x]['ddpg_symmetric']['collab']['no_equivar'][0]['info'] for x in x_pos]
    two = [plot_data[x]['greedy']['no_collab']['no_equivar'][0]['info'] for x in x_pos]
    three = [plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['info'] for x in x_pos]
    four = [plot_data[x]['ddpg_symmetric']['collab']['equivar'][0]['info'] for x in x_pos]
    

    ax1.plot(x_pos, one, ls='--', marker='+', label='No Equivariance')
    ax1.plot(x_pos, two, ls='--', marker='.', label='Greedy')
    ax1.plot(x_pos, three, ls='--', marker='o', label='Individual Reward')
    ax1.plot(x_pos, four, ls='--', marker='*', label='Fair-E')

    ax1.invert_xaxis()
    
    plt.xlabel('Pursuer Velocity')
    plt.ylabel('I(R,Z)')
    plt.title('Team Fairness')
    plt.legend()

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(x_pos, [plot_data[x]['ddpg_symmetric']['collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='+', label='No Equivariance')
    ax2.plot(x_pos, [plot_data[x]['greedy']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='.', label='Greedy')
    ax2.plot(x_pos, [plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='o', label='Individual Reward')
    ax2.plot(x_pos, [plot_data[x]['ddpg_symmetric']['collab']['equivar'][0]['capture_success'] for x in x_pos], ls='--', marker='*', label='Fair-E')
    
    ax2.invert_xaxis()

    plt.xlabel('Pursuer Velocity')
    plt.ylabel('Capture Success %')
    plt.title('Team Utility')
    plt.legend()

    # plt.subplots_adjust(
    #             # bottom=0.5,
    #             # top=0.9,
    #             wspace=0.4,
    #             hspace=0.8)

    # Save the full figure...
    


    fig.savefig('3bc.png')

    

    # x_pos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # print([plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['capture_success'] for x in x_pos])
    # print([plot_data[x]['ddpg_symmetric']['no_collab']['no_equivar'][0]['info'] for x in x_pos])

    
    # result = sorted(paths, key=lambda tup: tup[0])
    # for p in result:
    #     print(p)




    # plot utility vs. lambda
    print('plot 6:')

    LAMDAS2 = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
    vels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig = plt.figure(figsize=(17,20))
    # plt.rcParams.update({'font.size': 18})
    sns.set(style="white", font_scale=1.5)
    sns.set_context("poster")
    sns.set_palette("Greys_r")
    plt.rcParams["axes.grid"] = True

    # print([plot_data[val]['ddpg_fair']['collab']['no_equivar'][0.5] for val in vels])

    for num, val in enumerate(vels):
        ax1 = fig.add_subplot(3,2,num+1)
        info_ys = [plot_data[val]['ddpg_fair']['collab']['no_equivar'][x]['info'] for x in LAMDAS2]
        success_ys = [plot_data[val]['ddpg_fair']['collab']['no_equivar'][x]['capture_success'] for x in LAMDAS2]
        
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
            plt.xlabel('I(R;Z)')

        if num % 2 == 0:
            plt.ylabel('Capture Success %')


        plt.title('Velocity ' + str(val))

        plt.legend(handles=patches)

    



    plt.savefig('test.png')



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
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)