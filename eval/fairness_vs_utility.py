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

WORLD_SIZE = 6.0
PRED_SIZE = 0.075
PREY_SIZE = 0.05



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

TEST_VELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
# TEST_VELS = [1.2, 1.1, 1.0, 0.9, 0.7, 0.6, 0.5]
# TEST_VELS = [1.2]
INVALID_STRATS = ['lambda_0.4', 'lambda_0.7']

def main(config):
    # load trajectories
    strategies = os.listdir(config.fp)
    plot_data = {}    
    for vel in sorted(TEST_VELS):
        vel_results = {}
        for strat in sorted(strategies):
            if os.path.isdir(os.path.join(config.fp, strat)) and strat not in INVALID_STRATS:
                # path = os.path.join(config.fp, strat, 'vel_{}'.format(vel), 'trajectories.pkl')
                # seed_path = os.path.join(config.fp, strat, 'v1', 'plot_data', 'trajectories')
                seed_path = os.path.join(config.fp, "stored_trajectories")
                seeds = os.listdir(seed_path)

                rewards, caps, mis = [], [], []
                for seed in seeds:
                    if os.path.isdir(os.path.join(seed_path, seed)):
                        print(os.path.join(seed_path, seed))
                        print(strat, 'vel_{}'.format(vel), seed)
                        # path = os.path.join(seed_path, seed, 'vel_{}'.format(vel), 'trajectories.pkl')
                        path = os.path.join(seed_path, seed,  'trajectories.pkl')
                        
                        file = open(path, "rb")
                        trajectories = pickle.load(file)
                        file.close()

                        p_keys = sorted(trajectories['positions'][0].keys())[:-1]
                        vec_outcomes, sum_outcomes = [], []
                        num_captures = 0
                        for i in range(len(trajectories['positions'])):
                            reward_vec = np.zeros(len(p_keys), dtype=np.int)
                            rew = -50.0
                            if len(trajectories['positions'][i][p_keys[0]]) < 499:
                                num_captures += 1
                                rew = 50.0 - len(trajectories['positions'][i][p_keys[0]]) * 0.1
                                for j, key in enumerate(sorted(p_keys)):
                                    # compute reward vector
                                    pred_pos = trajectories['positions'][i][key][-1]
                                    prey_pos = trajectories['positions'][i]['prey'][-1]
                                    dist = toroidal_distance(pred_pos, prey_pos, WORLD_SIZE)
                                    if dist < PRED_SIZE + PREY_SIZE:
                                        reward_vec[j] = 1

                            vec_outcomes.append(vec_to_idx(reward_vec))
                            sum_outcomes.append(np.sum(reward_vec))

                        caps.append(num_captures/len(sum_outcomes))
                        rewards.append(rew)


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
                        print('info = {}'.format(i_ra))
                        print('cap success = {}\n'.format(num_captures/len(sum_outcomes)))

                        mis.append(i_ra)

                vel_results[strat] = {
                    'reward' : rewards,
                    'capture_success' : caps,
                    'info' : mis
                }

                

            # store results for strategy
            plot_data[vel] = vel_results
    


    #PLOT1
    import random
    fig = plt.figure()
    x_pos = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    x_pos.reverse()
    individual = [random.random() for x in x_pos]
    shared = [x+0.2 for x in individual]
    bar_width = 0.04
    
    plt.bar(x_pos, individual, bar_width, color='b')
    plt.bar([x + bar_width for x in x_pos], shared, bar_width, color='r')
    plt.xticks([x + bar_width/2 for x in x_pos], sorted(x_pos) )
    plt.xlabel('Pursuer Velocity')
    plt.ylabel('Capture USccess %')
    plt.title('Mutual vs Individual Reward (Capture Success)')
    plt.savefig('success_vs_velocity.png')
    
    #-----------------------

    # print(results['reward_vectors'])
    # print(results['reward_sums'])
    
    
    
    #PLOT2
    fig = plt.figure()
    x_pos = [i for i in range(8)]
    vals = [results['reward_vectors'].count(i) / len(results['reward_vectors']) for i in range(8)]
    bar_width = 0.4

    
    plt.bar(x_pos, vals, bar_width, color='g')
    plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
    plt.ylabel('P(r)')
    plt.title('No Equivariance')
    plt.savefig('individual_distr.png')
    # #-----------------------

    #PLOT3
    fig = plt.figure()
    x_pos = [i for i in range(8)]
    vals = [results['reward_sums'].count(i) / len(results['reward_sums']) for i in range(8)]
    bar_width = 0.4

    
    plt.bar(x_pos, vals, bar_width, color='b')
    plt.xticks(x_pos, [idx_to_vec(x) for x in x_pos] )
    plt.ylabel('P(r)')
    plt.title('Fair-E')
    plt.savefig('shared_distr.png')
    # #-----------------------



    algs = {} 
    print(plot_data.keys())
    for key in plot_data[1.0].keys():
        algs[key] = {
            'reward' : [],
            'capture_success' : [],
            'info' : []
        }

    for key in sorted(plot_data.keys()):
        for alg in plot_data[key].keys():
            algs[alg]['reward'].append(random.random())
            algs[alg]['capture_success'].append(random.random())
            algs[alg]['info'].append(random.random())
            # algs[alg]['reward'].append(plot_data[key][alg]['reward'])
            # algs[alg]['capture_success'].append(plot_data[key][alg]['capture_success'])
            # algs[alg]['info'].append(plot_data[key][alg]['info'])

    #PLOT4
    fig = plt.figure()
    x_pos = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    x_pos.reverse()

    import itertools
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 

    
    for key in algs.keys():
        plt.plot(x_pos, algs[key]['info'], ls='--', marker=next(marker))

    plt.xlabel('Pursuer Velocity')
    plt.ylabel('I(R,Z)')
    plt.title('Team Fairness')
    plt.savefig('fairness.png')

    return

    # plot utility vs. lambda
    print('plotting...')
    print(plot_data.keys())
    for key in sorted(plot_data.keys()):
        # xs = [float(x.split('_')[-1]) for x in sorted(plot_data[key].keys())]

        print(plot_data[key].keys())

        # plot reward
        reward_ys = [np.mean(plot_data[key][y]['reward']) for y in sorted(plot_data[key].keys())]
        reward__y_errs = [np.std(plot_data[key][y]['reward']) for y in sorted(plot_data[key].keys())]

        # fig = plt.figure()
        # plt.plot(xs, reward_ys, linewidth=3)
        # plt.ylim(-50.0, 50.0)
        # plt.savefig('/Users/nikogrupen/Desktop/{}_reward_vs_lambda.png'.format(key))

        # plot capture success
        success_ys = [np.mean(plot_data[key][y]['capture_success']) for y in sorted(plot_data[key].keys())]
        success_y_errs = [np.std(plot_data[key][y]['capture_success']) for y in sorted(plot_data[key].keys())]

        peos = [1,2,3,4,5]

        fig = plt.figure()
        # plt.plot(peos, success_ys, linewidth=3)
        # plt.ylim(0.0, 1.1)
        
        x_pos = [0, 1, 2, 3, 4, 5, 6, 7]
        bar_width = 0.04

        print(sorted(plot_data.keys()))
        print(success_ys)
        print(success_y_errs)
        
        plt.bar(sorted(plot_data.keys()), success_ys, bar_width, color='b')
        plt.bar([x + bar_width for x in sorted(plot_data.keys())], success_y_errs, bar_width, color='r')
        plt.xticks([x + bar_width/2 for x in x_pos], sorted(plot_data.keys()) )
        plt.xlabel('Number')
        plt.ylabel('Value')
        plt.title('Two Bars Side by Side')
        plt.savefig('{}_success_vs_lambda.png'.format(key))

        # plot MI
        info_ys = [np.mean(plot_data[key][y]['info']) for y in sorted(plot_data[key].keys())]
        # fig = plt.figure()
        # plt.plot(xs, info_ys, linewidth=3)
        # plt.ylim(0.0, 0.8)
        # # plt.savefig('/Users/nikogrupen/Desktop/{}_info_vs_lambda.png'.format(key))




        # plot MI vs. capture success
        fig = plt.figure()
        sns.set(style="white", font_scale=1.5)
        sns.set_context("poster")
        sns.set_palette("Greys_r")
        plt.rcParams["axes.grid"] = True
        sns.lineplot(x=info_ys, y=success_ys, style=True, dashes=[(2,2)]*len(info_ys), markers=False, legend=False)
        palette = plt.get_cmap('tab10')
        colors = [palette(i) for i in np.arange(len(info_ys))]
        plt.scatter(info_ys, success_ys, s=14**2, c=colors, marker='o', edgecolors='black', linewidths=1.2, zorder=10)
        plt.errorbar(info_ys, success_ys, yerr=success_y_errs, linestyle="None", ecolor='black', elinewidth=1.5)
        plt.ylim(-0.05, 1.1)
        plt.xlim(-0.01, np.max(info_ys) + 0.05)
        patches = []
        for i, k in enumerate(sorted(plot_data[key].keys())):
            patches.append(mpatches.Patch(color=colors[i], label='\u03BB = {}'.format(k.split('_')[-1])))
        plt.legend(handles=patches, loc='lower right', prop={'size': 14})
        plt.savefig('{}_info_vs_success.png'.format(key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--fp', type=str, default=None, help='path to load/save')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)


