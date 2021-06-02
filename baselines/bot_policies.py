import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from utils.misc import *


class Policy(object):
    def __init__(self):
        pass

    def sample_action(self, obs):
        raise NotImplementedError() 

# ---------------------------------------------------
# Predator Policies
# ---------------------------------------------------
class GreedyPolicy(Policy):
    def __init__(self, env, index):
        super(GreedyPolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        self.bounds = False if env.world.torus else True
        self.max_speed = self.env.world.agents[self.index].max_speed
        self.discrete = self.env.discrete_action_space

    def sample_action(self, obs):
        # take action that brings us closest to prey
        u = self.compute_best_action(obs)

        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        return np.array([u, np.zeros(self.env.world.dim_c)])

    def compute_best_action(self, obs, step_size=0.1):
        forces = []
        obs = parse_obs(obs, self.env)

        # find closest prey
        dists = [compute_distance(obs['pos'], y, self.bounds, self.env.world.size) for y in obs['prey_pos']]
        min_dist = np.argmin(dists)

        # attractive force to prey
        forces.append(attractive_force(obs['pos'], obs['prey_pos'][min_dist], self.bounds, 
                    world_size=self.env.world.size, k_att=1.5, thresh=self.env.world.size))
        
        # repulsive force (landmarks)
        for e in obs['landmarks']:
            forces.append(repulsive_force(obs['pos'], e, self.bounds, world_size=self.env.world.size, k_rep=0.75, thresh=1.5))
        
        # target = normalized sum of forces
        f = np.sum(forces, axis=0)
        if not all(f == 0.0):
            f = (f / np.max(abs(f))) * self.max_speed

        if self.discrete:
            # target point
            target = pos + f

            # approximate action
            dirs = [obs['pos'],
                    [obs['pos'] + step_size, obs['pos'][1]], # right
                    [obs['pos'][0] - step_size, obs['pos'][1]], # left
                    [obs['pos'][0], obs['pos'][1] + step_size], # up
                    [obs['pos'][0], obs['pos'][1] - step_size] # downs
                ]

            # compute action distance to target and select closest
            dists = [compute_distance(d, target, self.bounds, self.env.world.size) for d in dirs]
            return np.argmin(dists)
        else:
            return f


class PincerPolicy(Policy):
    def __init__(self, env, index, epsilon=2.5, update=True, increment=0.97):
        super(PincerPolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        self.bounds = False
        self.discrete = self.env.discrete_action_space
        self.update = update
        self.assignment = None
        self.max_speed = self.env.world.agents[self.index].max_speed
        self.increment = increment
        self.epsilon = epsilon
        self.max_idx = None
        self.max_score = None
        self.max_pos = None
        self.rads = None
        self.formation = None
        self.step_size = None
        self.actives, self.active_idxs = None, None
        self.n_predators = len([agent for agent in self.env.world.agents if agent.adversary])

    def sample_action(self, obs):
        # compute angle assignment
        obs = parse_obs(obs, self.env)

        if self.n_predators == 5:
            self.assignment, pos = self.role_assignment_five(obs)
        else:
            self.assignment, pos = self.role_assignment_three(obs)

        # take action that brings us closest to angle assignment
        u = self.compute_best_action(obs, pos, self.assignment)

        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        return np.array([u, np.zeros(self.env.world.dim_c)])


    def compute_best_action(self, obs, pos, pt, step_size=0.1):
        forces = []

        # attractive field (formation + prey)
        f = attractive_force(pos, pt, True, world_size=self.env.world.size, k_att=1.5)
        forces.append(f)

        # target = normalized sum of repulsive forces
        f = np.sum(forces, axis=0)
        f = (f / np.max(abs(f))) * self.max_speed

        if self.discrete:
            # target point
            target = obs['rel_pos'] + f

            # choose action that moves closest to assignment
            dirs = [obs['rel_pos'],
                    [obs['rel_pos'][0] + step_size, obs['rel_pos'][1]], # right
                    [obs['rel_pos'][0] - step_size, obs['rel_pos'][1]], # left
                    [obs['rel_pos'][0], obs['rel_pos'][1] + step_size], # up
                    [obs['rel_pos'][0], obs['rel_pos'][1] - step_size] # down
                ]

            dists = [compute_distance(d, target, self.bounds, self.env.world.size) for d in dirs]
            return np.argmin(dists)
        else:
            return f


    def role_assignment_three(self, obs):
        prey_pos = obs['rel_prey_pos'][0]

        # current always predator first
        ps = [obs['pos']] + [p for p in obs['pred_pos']]
        rel_ps = [obs['pos'] - obs['prey_pos'][0]] + [p - obs['prey_pos'][0] for p in obs['pred_pos']]

        rel_p_replicas = []
        for p in rel_ps:
            r = [
                np.array([p[0], p[1]]),
                np.array([p[0] + self.env.world.size, p[1]]),
                np.array([p[0] - self.env.world.size, p[1]]),
                np.array([p[0], p[1] + self.env.world.size]),
                np.array([p[0], p[1] - self.env.world.size]),
                np.array([p[0] + self.env.world.size, p[1] + self.env.world.size]),
                np.array([p[0] + self.env.world.size, p[1] - self.env.world.size]),
                np.array([p[0] - self.env.world.size, p[1] + self.env.world.size]),
                np.array([p[0] - self.env.world.size, p[1] - self.env.world.size])
            ]
            rel_p_replicas.append(r)
            
        # p_replicas[0] is current predator
        step = 0
        pred_pos, rel_pred_pos, pred_idxs, scores, thetas, all_pred_thetas, all_rs = [], [], [], [], [], [], []
        for i, rel_p1 in enumerate(rel_p_replicas[0]):
            for j, rel_p2 in enumerate(rel_p_replicas[1]):
                for k, rel_p3 in enumerate(rel_p_replicas[2]):
                    # r_i and theta_i for this set of replicas
                    p1 = rel_p1 + obs['prey_pos'][0]
                    p2 = rel_p2 + obs['prey_pos'][0]
                    p3 = rel_p3 + obs['prey_pos'][0]

                    pred_thetas = [
                        math.atan2(rel_p1[1], rel_p1[0]),
                        math.atan2(rel_p2[1], rel_p2[0]),
                        math.atan2(rel_p3[1], rel_p3[0])
                    ]

                    rs = [
                        compute_distance(prey_pos, rel_p1, True, self.env.world.size),
                        compute_distance(prey_pos, rel_p2, True, self.env.world.size),
                        compute_distance(prey_pos, rel_p3, True, self.env.world.size)
                    ]

                    # compute cosine potential and scores
                    theta = cosine_potential(rs, pred_thetas)
                    min_u = cosine_cost(theta, rs, pred_thetas)

                    # book-keeping
                    scores.append(min_u)
                    thetas.append(theta)
                    all_pred_thetas.append(pred_thetas)
                    all_rs.append(rs)
                    pred_pos.append([p1, p2, p3]) # p1 is current predator
                    rel_pred_pos.append([rel_p1, rel_p2, rel_p3])
                    pred_idxs.append([i, j, k])
                    step += 1


        # select best formation
        if self.max_idx is not None:
            new_idxs = []
            for i, p in enumerate(rel_p_replicas):
                dists = [distance.euclidean(self.max_pos[i], pos) for pos in p]
                new_idxs.append(np.argmin(dists))

            if new_idxs != pred_idxs[self.max_idx]:
                self.max_idx = np.argwhere([new_idxs==p_idx for p_idx in pred_idxs]).item()


        challenger_score = np.max(scores)*np.mean(all_rs[np.argmax(scores)])
        # if self.index == 0 and self.max_idx is not None:
        #     print('challenger score = {}'.format(challenger_score))
        #     print('current score = {}'.format(scores[self.max_idx]*np.mean(all_rs[self.max_idx]) + self.epsilon))
        #     print()
        if self.max_idx is None or (challenger_score > scores[self.max_idx]*np.mean(all_rs[self.max_idx]) + self.epsilon):
            # index and relative positions of new active formation
            self.max_idx = np.argmax(scores)
            self.max_pos = rel_pred_pos[self.max_idx]

            # step size, formation leader for active formation
            rs = all_rs[self.max_idx]
            self.step_size = np.min(rs) * 0.99
            self.formation_idx = np.argmin(rs)

            # equally-spaced formation target points starting from formation leader
            phase_angle = math.atan2(self.max_pos[self.formation_idx][1], self.max_pos[self.formation_idx][0])
            angles = np.linspace(phase_angle, 2*math.pi + phase_angle, self.n_predators, endpoint=False)
            pts = [prey_pos + np.array([math.cos(angle), math.sin(angle)]) * self.step_size for angle in angles]

            if self.formation_idx == 0:
                # predator is closest, pursue phase angle
                self.pred_idx = 0
            else:
                # predator is not closest, pursue another angle
                del rel_ps[self.formation_idx]
                idxs = [i+1 for i in range(len(rs)-1)]
                temp_pts = pts[1:]
                dists = []
                for p in rel_ps:
                    dists.append([compute_distance(p, pt, True, self.env.world.size) for pt in temp_pts])
                _, col_ind = linear_sum_assignment(np.array(dists))
                self.pred_idx = idxs[col_ind[0]]
        else:
            # update formation and step size
            self.max_pos = rel_pred_pos[self.max_idx]
            self.step_size = np.min(all_rs[self.max_idx])*self.increment
            phase_angle = math.atan2(self.max_pos[self.formation_idx][1], self.max_pos[self.formation_idx][0])
            angles = np.linspace(phase_angle, 2*math.pi + phase_angle, self.n_predators, endpoint=False)
            pts = [prey_pos + np.array([math.cos(angle), math.sin(angle)]) * self.step_size for angle in angles]

        return pts[self.pred_idx], self.max_pos[0]


    def role_assignment_five(self, obs):
        prey_pos = obs['rel_prey_pos'][0]

        # current always predator first
        ps = [obs['pos']] + [p for p in obs['pred_pos']]
        rel_ps = [obs['pos'] - obs['prey_pos'][0]] + [p - obs['prey_pos'][0] for p in obs['pred_pos']]

        rel_p_replicas = []
        for p in rel_ps:
            r = [
                np.array([p[0], p[1]]),
                np.array([p[0] + self.env.world.size, p[1]]),
                np.array([p[0] - self.env.world.size, p[1]]),
                np.array([p[0], p[1] + self.env.world.size]),
                np.array([p[0], p[1] - self.env.world.size]),
                np.array([p[0] + self.env.world.size, p[1] + self.env.world.size]),
                np.array([p[0] + self.env.world.size, p[1] - self.env.world.size]),
                np.array([p[0] - self.env.world.size, p[1] + self.env.world.size]),
                np.array([p[0] - self.env.world.size, p[1] - self.env.world.size])
            ]
            rel_p_replicas.append(r)
            
        # p_replicas[0] is current predator
        step = 0
        pred_pos, rel_pred_pos, pred_idxs, scores, thetas, all_pred_thetas, all_rs = [], [], [], [], [], [], []
        for i, rel_p1 in enumerate(rel_p_replicas[0]):
            for j, rel_p2 in enumerate(rel_p_replicas[1]):
                for k, rel_p3 in enumerate(rel_p_replicas[2]):
                    for l, rel_p4 in enumerate(rel_p_replicas[3]):
                        for m, rel_p5 in enumerate(rel_p_replicas[4]):
                            # r_i and theta_i for this set of replicas
                            p1 = rel_p1 + obs['prey_pos'][0]
                            p2 = rel_p2 + obs['prey_pos'][0]
                            p3 = rel_p3 + obs['prey_pos'][0]
                            p4 = rel_p4 + obs['prey_pos'][0]
                            p5 = rel_p5 + obs['prey_pos'][0]

                            pred_thetas = [
                                math.atan2(rel_p1[1], rel_p1[0]),
                                math.atan2(rel_p2[1], rel_p2[0]),
                                math.atan2(rel_p3[1], rel_p3[0]),
                                math.atan2(rel_p4[1], rel_p4[0]),
                                math.atan2(rel_p5[1], rel_p5[0])
                            ]

                            rs = [
                                compute_distance(prey_pos, rel_p1, True, self.env.world.size),
                                compute_distance(prey_pos, rel_p2, True, self.env.world.size),
                                compute_distance(prey_pos, rel_p3, True, self.env.world.size),
                                compute_distance(prey_pos, rel_p4, True, self.env.world.size),
                                compute_distance(prey_pos, rel_p5, True, self.env.world.size)
                            ]

                            # compute cosine potential and scores
                            theta = cosine_potential(rs, pred_thetas)
                            min_u = cosine_cost(theta, rs, pred_thetas)

                            # book-keeping
                            scores.append(min_u)
                            thetas.append(theta)
                            all_pred_thetas.append(pred_thetas)
                            all_rs.append(rs)
                            pred_pos.append([p1, p2, p3, p4, p5]) # p1 is current predator
                            rel_pred_pos.append([rel_p1, rel_p2, rel_p3, rel_p4, rel_p5])
                            pred_idxs.append([i, j, k, l, m])
                            step += 1


        # select best formation
        if self.max_idx is not None:
            new_idxs = []
            for i, p in enumerate(rel_p_replicas):
                dists = [distance.euclidean(self.max_pos[i], pos) for pos in p]
                new_idxs.append(np.argmin(dists))

            if new_idxs != pred_idxs[self.max_idx]:
                self.max_idx = np.argwhere([new_idxs==p_idx for p_idx in pred_idxs]).item()

        challenger_score = np.max(scores)*np.mean(all_rs[np.argmax(scores)])
        if self.max_idx is None or (challenger_score > scores[self.max_idx]*np.mean(all_rs[self.max_idx]) + self.epsilon):
            
            # index and relative positions of new active formation
            self.max_idx = np.argmax(scores)
            self.max_pos = rel_pred_pos[self.max_idx]

            # step size, formation leader for active formation
            rs = all_rs[self.max_idx]
            self.step_size = np.min(rs) * 0.99
            self.formation_idx = np.argmin(rs)

            # equally-spaced formation target points starting from formation leader
            phase_angle = math.atan2(self.max_pos[self.formation_idx][1], self.max_pos[self.formation_idx][0])
            angles = np.linspace(phase_angle, 2*math.pi + phase_angle, self.n_predators, endpoint=False)
            pts = [prey_pos + np.array([math.cos(angle), math.sin(angle)]) * self.step_size for angle in angles]

            if self.formation_idx == 0:
                # predator is closest, pursue phase angle
                self.pred_idx = 0
            else:
                # predator is not closest, pursue another angle
                del rel_ps[self.formation_idx]
                idxs = [i+1 for i in range(len(rs)-1)]
                temp_pts = pts[1:]
                dists = []
                for p in rel_ps:
                    dists.append([compute_distance(p, pt, True, self.env.world.size) for pt in temp_pts])
                _, col_ind = linear_sum_assignment(np.array(dists))
                self.pred_idx = idxs[col_ind[0]]
        else:
            # update formation and step size
            self.max_pos = rel_pred_pos[self.max_idx]
            self.step_size = np.min(all_rs[self.max_idx])*self.increment
            phase_angle = math.atan2(self.max_pos[self.formation_idx][1], self.max_pos[self.formation_idx][0])
            angles = np.linspace(phase_angle, 2*math.pi + phase_angle, self.n_predators, endpoint=False)
            pts = [prey_pos + np.array([math.cos(angle), math.sin(angle)]) * self.step_size for angle in angles]

        return pts[self.pred_idx], self.max_pos[0]

# ---------------------------------------------------
# Prey Policies
# ---------------------------------------------------
class EscapeClosestPolicy(Policy):
    def __init__(self, env, index):
        super(EscapeClosestPolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        self.bounds = False if env.world.torus else True
        self.discrete = False
        self.max_speed = self.env.world.agents[self.index].max_speed

    def sample_action(self, obs):
        if self.env.agents[self.index].captured:
            # agents cannot move if captured
            u = 0 if self.discrete else np.zeros(2)
        else:
            # take action that brings us furthest from predators
            u = self.compute_best_action(obs)

        # one-hot if necessary
        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        return np.array([u, np.zeros(self.env.world.dim_c)])


    def compute_best_action(self, obs, step_size=0.1):
        forces = []
        obs = parse_obs(obs, self.env, is_prey=True)

        # find nearest predator
        dists = [compute_distance(obs['pos'], y, self.bounds, self.env.world.size) for y in obs['pred_pos']]
        min_dist = np.argmin(dists)

        # repulsive force (predators)
        forces.append(repulsive_force(obs['pos'], obs['pred_pos'][min_dist], self.bounds, world_size=self.env.world.size, k_rep=1.5, thresh=self.env.world.size/2.0))

        # target = normalized sum of repulsive forces
        f = np.sum(forces, axis=0)
        if not all(f == 0.0):
            f = (f / np.max(abs(f))) * self.max_speed

        if self.discrete:
            # target point
            target = obs['pos'] + f

            # approximate action
            dirs = [obs['pos'],
                    [obs['pos'][0] + step_size, obs['pos'][1]], # right
                    [obs['pos'][0] - step_size, obs['pos'][1]], # left
                    [obs['pos'][0], obs['pos'][1] + step_size], # up
                    [obs['pos'][0], obs['pos'][1] - step_size] # down
                ]

            # compute action distance to target and select closest
            dists = [compute_distance(d, target, self.bounds, self.env.world.size) for d in dirs]
            return np.argmin(dists)
        else:
            return f


class EscapeCosinePolicy(Policy):
    def __init__(self, env, index):
        super(EscapeCosinePolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        self.bounds = False if env.world.torus else True
        self.discrete = False
        self.max_speed = self.env.world.agents[self.index].max_speed
        self.step = 0

    def sample_action(self, obs):
        # take action that brings us furthest from predators
        u = self.compute_best_action(obs)

        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        self.step += 1
        return np.array([u, np.zeros(self.env.world.dim_c)])


    def compute_best_action(self, obs, step_size=0.15):
        forces = []
        obs = parse_obs(obs, self.env, is_prey=True)


        rs = [compute_distance(obs['rel_pos'], y, self.bounds, self.env.world.size) for y in obs['rel_pred_pos']]
        pred_thetas = [math.atan2(p[1], p[0]) for p in obs['rel_pred_pos']]

        # compute directly
        theta = cosine_potential(rs, pred_thetas)

        # max speed in that direction
        f = np.array([math.cos(theta), math.sin(theta)] - obs['rel_pos'])
        forces.append(f)

        # target = normalized sum of repulsive forces
        f = np.sum(forces, axis=0)
        if not all(f == 0.0):
            f = (f / np.max(abs(f))) * self.max_speed

        if self.discrete:
            # target point
            target = obs['rel_pos'] + f

            # approximate action
            dirs = [obs['rel_pos'],
                    [obs['rel_pos'][0] + step_size, obs['rel_pos'][1]], # right
                    [obs['rel_pos'][0] - step_size, obs['rel_pos'][1]], # left
                    [obs['rel_pos'][0], obs['rel_pos'][1] + step_size], # up
                    [obs['rel_pos'][0], obs['rel_pos'][1] - step_size] # down
                ]

            # compute action distance to target and select closest
            dists = [compute_distance(d, target, self.bounds, self.env.world.size) for d in dirs]
            return np.argmin(dists)
        else:
            return f


# ---------------------------------------------------
# Agent-Agnostic Policies
# ---------------------------------------------------
class NoopPolicy(Policy):
    def __init__(self, env, index):
        super(NoopPolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        self.discrete = self.env.discrete_action_space

    def sample_action(self, obs):
        # ignore observation and no-op
        if self.discrete:
            u = 0

            if not self.env.discrete_action_input:
                u = one_hot_encode(u, self.num_actions)
        else:
            u = np.array([0., 0.])

        return np.array([u, np.zeros(self.env.world.dim_c)])


class RandomPolicy(Policy):
    def __init__(self, env, index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        # self.discrete = self.env.discrete_action_space
        self.discrete = False


    def sample_action(self, obs):
        if self.env.agents[self.index].captured:
            # agents cannot move if captured
            u = 0 if self.discrete else np.zeros(2)
        else:
            # take action that brings us furthest from predators
            u = np.random.randint(self.num_actions) if self.discrete else np.random.uniform(-1, 1, 2)

        # one-hot if necessary
        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        return np.array([u, np.zeros(self.env.world.dim_c)])


class RandomLinePolicy(Policy):
    def __init__(self, env, index):
        super(RandomLinePolicy, self).__init__()
        self.env = env
        self.index = index
        self.learning_agent = False
        self.num_actions = 5
        # self.discrete = self.env.discrete_action_space
        self.discrete = False
        self.num_preds = len([a for a in self.env.agents if a.adversary])

        self.step_count = 0
        self.move = None


    def sample_action(self, obs):
        # ignore observation and act randomly
        if self.env.agents[self.index].captured:
            # agents cannot move if captured
            self.move = 0 if self.discrete else np.zeros(2)
        elif self.move is None or self.step_count % 50 == 0:
            if self.discrete:
                self.move = np.random.randint(1, 5)
            else:
                f = np.random.uniform(-1.5, 1.5, 2) # change from int to angle --> force
                self.move = np.zeros(2) + f

        u = self.move

        if self.discrete and not self.env.discrete_action_input:
            u = one_hot_encode(u, self.num_actions)

        self.step_count += 1
        return np.array([u, np.zeros(self.env.world.dim_c)])


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def decentralized_predator(env, index, strategy, update):
    # select from predator strategies (noop, random, greedy, korf, circular)
    if strategy == 'noop':
        return NoopPolicy(env, index)
    elif strategy == 'random':
        return RandomPolicy(env, index)
    elif strategy == 'greedy':
        return GreedyPolicy(env, index)
    elif strategy == 'pincer':
        return PincerPolicy(env, index)
    else:
        raise ValueError('Invalid predator strategy. Valid strategies are: random, greedy, pincer.')


def decentralized_prey(env, index, strategy, update):
    # select from prey strategies (noop, random, closest, angles, circular)
    if strategy == 'noop':
        return NoopPolicy(env, index)
    elif strategy == 'random':
        return RandomPolicy(env, index)
    elif strategy == 'rand-line':
        return RandomLinePolicy(env, index)
    elif strategy == 'closest':
        return EscapeClosestPolicy(env, index)
    elif strategy == 'sum':
        return EscapeSumPolicy(env, index)
    elif strategy == 'cosine':
        return EscapeCosinePolicy(env, index)
    else:
        raise ValueError('Invalid predator strategy. Valid strategies are: noop, random, closest, sum, cosine.')








