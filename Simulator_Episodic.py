import seaborn.objects as so
import pandas as pd
import numpy as np
from utils import (norm_density,
                   timeit_debug, timeit_info,
                   estimate_occ_zero_cf,
                   estimate_occ_acf,
                   estimate_episodic_acf,
                   estimate_episodic_acf_v2,
                   jit_nopython,
                   jit_cache,
                   n_decimals,
                   verbose)
from numba import jit
import tqdm
import torch
from torch.distributions import Categorical

random_state = np.random.RandomState(1234)

class EpisodicSimulator(object):
    """
    FUNCTION: Samples from propagator, optionally provides episodic functionality like trajectory plotting.
    INPUTS: rho_init        = initial state density, None = random selection, 'env_default' = ENV.start
            mass            = assume particle as momentum, approximately corresponds to #previous states to avoid
                            # TODO reconceptualize as self-motion cue
            no_dwell        = True, sample away from current state (i.e. force jump, ->embedded discrete-time chain)
            diagnostics     = computes coverage efficiency/MSD etc of sampled trajectories
    """

    def __init__(self, PROP, rho_init='env_default', mass=1, no_dwell=True, label='SIMULATOR',
                 target_dir='simulations', episodic=False, **kwargs):
        self.PROP = PROP
        self.episodic = episodic  # Flag to enable episodic features
        if rho_init == 'env_default':
            self.rho_init = process_rho(self.ENV.start, self.n_state)
        else:
            self.rho_init = process_rho(rho_init, self.n_state)
        self.no_dwell = no_dwell
        self.mass = mass
        self.label = label
        self.target_dir = target_dir
        self.sequences_sampled = False
        if hasattr(self.ENV, 'R') or hasattr(self.ENV, 'R_state'):
            self.no_reward_func = False  # sampling functions will sample rewards
        else:
            self.no_reward_func = True  # sampling functions will not sample rewards
        for key, value in kwargs.items():
            setattr(self, key, value)

        np.random.seed()
        assert self.rho_init.size == self.PROP.n_state, 'Dimensionality of starting state density is incorrect.'

    # Properties
    @property
    def beta(self):
        return self.PROP.beta

    @property
    def n_dim(self):
        return self.PROP.n_dim

    @property
    def n_state(self):
        return self.PROP.n_state

    @property
    def ix_states(self):
        return self.PROP.states

    @property
    def world_array(self):
        return self.GEN.ENV.world_array

    @property
    def GEN(self):
        return self.PROP.GEN

    @property
    def ENV(self):
        return self.PROP.GEN.ENV

    @ENV.setter
    def ENV(self, value):
        self.PROP.GEN.ENV = value

    # Methods
    @timeit_debug
    @jit(nopython=jit_nopython, parallel=jit_nopython, cache=jit_cache)
    def evolve(self, n_step=1, rho_start=None, ignore_imag=True):
        if rho_start is None:
            rho_start = self.rho_init
        self._check_state_density(rho_state=rho_start)
        for i in range(n_step):
            rho_stop = np.dot(rho_start, self.PROP.etO)
            if not np.all(np.isreal(rho_stop)):
                if ignore_imag:
                    print('SIMULATOR: complex propagated density')
                    rho_stop = rho_stop.real
                else:
                    raise ValueError('SIMULATOR: complex propagation density')
            rho_stop = norm_density(rho_stop, beta=self.beta, type='l1')
            rho_start = rho_stop
        self._check_state_density(rho_state=rho_start)
        return rho_start

    @timeit_debug
    def _check_state_density(self, rho_state, n_decimals=n_decimals):
        assert (rho_state >= 0).all(), 'State density in negative range.'
        if np.allclose(rho_state.sum(), 1):
            return rho_state / rho_state.sum()
        else:
            raise ValueError(f'SIMULATOR: state density sums to {rho_state.sum():.8f} (!= 1).')

    @timeit_debug
    def _sample_state(self, rho_state, prev_states=None):
        if (self.no_dwell and self.mass != 0.) and prev_states is not None:
            mass = np.floor(self.mass).astype('int')
            states = prev_states
            except_states = states[~np.isnan(states)].astype('int')
            except_states = except_states[-np.min([mass, len(except_states)])]
            rho_state[except_states] = 0.
        rho_state = rho_state / rho_state.sum()
        return sample_discrete(rho_state)

    @timeit_debug
    def _sample_reward(self, state, prev_state=None):
        if prev_state is None:
            reward = self.ENV.R_state[state]
        else:
            reward = self.ENV.R[prev_state, state]
        return reward

    @timeit_info
    @jit(nopython=jit_nopython, parallel=jit_nopython, cache=jit_cache)
    def sample_sequences(self, n_step=100, n_samp=50, rho_start=None, fast_storage=True):
        self.sequences_generated = True
        self.fast_storage = fast_storage
        if rho_start is None:
            rho_start = self.rho_init
        rho_start = process_rho(rho_start, self.n_state)
        n_seq_steps = n_step + 1
        self.n_step = n_step
        self.n_samp = n_samp
        self.n_seq_steps = n_seq_steps
        self.ix_steps = np.arange(0, n_seq_steps, 1)
        self.ix_samps = np.arange(0, n_samp, 1)
        self.samp_times = self.ix_steps * (1 / self.PROP.tau)
        self.ix_slice = pd.IndexSlice

        state_seqs = np.zeros((n_samp, n_seq_steps))
        rhos = np.zeros((n_samp, n_seq_steps, self.n_state))
        rewards = np.zeros((n_samp, n_seq_steps))
        log_probs = torch.zeros((n_samp, n_seq_steps))

        iterator = tqdm(range(n_samp), desc='SAMPLING') if verbose else range(n_samp)

        for ns in iterator:
            state = self._sample_state(rho_start)
            state_seqs[ns, 0] = state
            rhos[ns, 0, :] = rho_start
            log_probs[ns, 0] = Categorical(torch.tensor(rho_start)).log_prob(torch.tensor(state)).item()

            if not self.no_reward_func:
                rewards[ns, 0] = self._sample_reward(state)

            rho_inter = process_rho(state, self.n_state)
            for n in range(1, n_seq_steps):
                rho_stop = self.evolve(n_step=1, rho_start=rho_inter)
                state = self._sample_state(rho_stop, prev_states=np.array(state_seqs[ns, :n]))
                state_seqs[ns, n] = state
                rhos[ns, n, :] = rho_stop
                log_probs[ns, n] = Categorical(torch.tensor(rho_stop)).log_prob(torch.tensor(state)).item()
                if not self.no_reward_func:
                    rewards[ns, n] = self._sample_reward(state)
                rho_inter = process_rho(state, self.n_state)

        if fast_storage:
            self.state_seqs = state_seqs.astype('int')
            self.rhos = rhos
            self.rewards = rewards
            self.log_probs = log_probs
        else:
            self.output_scalar.loc[self.ix_slice[:, :], 'state'] = state_seqs.flatten()
            self.output_vector.loc[self.ix_slice[:, :, :], 'rho_stop'] = rhos.flatten()
            self.output_scalar.loc[self.ix_slice[:, :], 'reward'] = rewards.flatten()
            self.output_scalar.loc[self.ix_slice[:, :], 'log_prob'] = log_probs.flatten()

    # Episodic-specific methods
    def plot_trajectory(self, samp=0):
        if self.episodic:
            coords = np.array(self._retrieve_state(samp=samp, step=None, coords=True))
            coords = pd.DataFrame(coords, columns=["x", "y"])
            return (
                so.Plot(self.ENV.info_state, x="x", y="y", color="color")
                .add(so.Dot())
                .scale(color=so.Nominal(), x=so.Temporal())
                .add(so.Line(coords, color="black"))
            )
        else:
            raise AttributeError("Episodic functionality is disabled.")

    def estimate_cf(self, axis=None, dist_occ=0, zero_pos=False):
        if self.episodic:
            samps_state = self.ENV.states[self.state_seqs]
            samps_state[:, :, 0] = self.ENV.semantic_mds[samps_state[:, :, 0].astype(int)]
            samps_state[:, :, 2] = self.ENV.spatial_mds[samps_state[:, :, 2].astype(int)]
            samps_state = samps_state[:, :, 0:3]
            self.acf_mean, self.acf_sem = estimate_episodic_acf_v2(samps_state, axis=axis)
        else:
            samps_state = self._retrieve_state(samp=None, step=None, coords=False)
            if zero_pos:
                self.acf_mean, self.acf_sem = estimate_occ_zero_cf(samps_state.T, d=dist_occ)
            else:
                self.acf_mean, self.acf_sem = estimate_occ_acf(samps_state.T, d=dist_occ)

def process_rho(rho, n_state):
    """
    FUNCTION: Process state distribution.
    INPUTS: rho = state distribution or state
    OUTPUTS: rho_out = state distribution
    NOTES: rho = None returns a uniform distribution
           rho = state returns a one-hot distribution
           else rho_out==rho
    """
    if rho is None:
        rho_out = np.ones((n_state))
        return rho_out/rho_out.sum()
    elif not hasattr(rho, "__len__") or np.array(rho).size < n_state:
        rho_out = np.zeros((n_state))
        rho_out[np.asarray(rho).astype('int')] = 1.
    else:
        rho_out = rho
    return rho_out/rho_out.sum()

def sample_discrete(p):
    """FUNCTION: discrete sample from 1:len(p) with prob p."""
    return np.random.choice(list(range(len(p))), 1, p=p)