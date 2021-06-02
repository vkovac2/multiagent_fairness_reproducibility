import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe
import gym
from utils.base_vec_env import VecEnv, CloudpickleWrapper


#---------------------------------------------------
# Envs
#---------------------------------------------------
# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


# https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/env_wrappers.py
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if np.isnan(data).all():
                # placeholder response for paused thread
                obs = [np.zeros(env.observation_space[0].shape)] * env.n
                rew = [0.] * env.n
                done = [True] * env.n
                info = [{'active' : False}] * env.n
                remote.send((obs, rew, done, info))
            else:
                # environment response
                obs, rew, done, info = env.step(data)
                remote.send((obs, rew, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        elif cmd == 'get_bot_params':
            params = {}
            params['n_predators'] = env.num_preds
            params['n_prey'] = env.num_prey
            params['n_landmarks'] = env.num_landmarks
            params['predator_max_speed'] = env.world.agents[0].max_speed
            params['prey_max_speed'] = env.world.agents[-1].max_speed
            params['world_size'] = env.world.size
            params['is_torus'] = env.world.torus
            params['discrete_action_space'] = env.discrete_action_space
            params['discrete_action_input'] = env.discrete_action_input
            params['dim_p'] = env.world.dim_p
            params['dim_c'] = env.world.dim_c
            remote.send(params)
        elif cmd == 'env_method':
            method = getattr(env, data[0])
            remote.send(method(*data[1]))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
        multiprocessing.set_start_method('forkserver' if forkserver_available else 'spawn')

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.processes = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.processes:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()

        self.remotes[0].send(('get_bot_params', None))
        self.bot_params = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def step_async(self, actions, paused_indices=None):
        # remotes = self._get_target_remotes(indices)
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if paused_indices is not None and i in paused_indices:
                action = [np.nan] * len(action)
                remote.send(('step', np.nan))
            else:
                remote.send(('step', action))

        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def call_env_method(self, method_name, *method_args):
        for remote in self.remotes:
            remote.send(('env_method', (method_name, method_args)))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
        self.closed = True

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None


        self.bot_params = {
            'n_predators' : env.num_preds,
            'n_prey' : env.num_prey,
            'n_landmarks' : env.num_landmarks,
            'predator_max_speed' : env.world.agents[0].max_speed,
            'prey_max_speed' : env.world.agents[-1].max_speed,
            'world_size' : env.world.size,
            'is_torus' : env.world.torus,
            'discrete_action_space' : env.discrete_action_space,
            'discrete_action_input' : env.discrete_action_input,
            'dim_p' : env.world.dim_p,
            'dim_c' : env.world.dim_c
        }

    def step_async(self, actions, paused_indices=None):
        # remotes = self._get_target_remotes(indices)
        new_actions = []
        for i, action in enumerate(actions):
            if paused_indices is not None and i in paused_indices:
                action = [np.nan] * len(action)

            new_actions.append(action)

        self.actions = new_actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos


    def call_env_method(self, method_name, *method_args):
        resps = []
        for env in self.envs:
            method = getattr(env, method_name)
            resps.append(method(*method_args))
        return resps

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return