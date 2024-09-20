#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.special import softmax as softmax_scipy
from sklearn.preprocessing import normalize
from functools import wraps
import logging
import time
#import config
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from scipy.stats import sem
import statsmodels.api as sm




import os
import sys
import numpy as np; np.seterr(all='raise')
import seaborn as sb
import matplotlib.pyplot as plt

verbose         = False # True to output detailed code progress statements
timing          = True
jit_nopython    = True
jit_cache       = True
sparsify        = False
debug           = False

# settings
# min_val = np.finfo(float).eps # sets lower bounds on probabilities
min_val = 10**-5 # sets lower bounds on probabilities
n_decimals = 6 # calculations rounded to n_decimals places to suppress numerical fluctuations

os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = str(1)
os.environ['NUMBA_DISABLE_JIT'] = str(1) # set to 1 to disable numba.jit, otherwise 0
os.environ['NUMBA_WARNINGS'] = str(1)

if not sys.warnoptions and verbose:
    import warnings
    warnings.simplefilter("default") # "default"/"error"
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

############################timer###########################################

logger = logging.getLogger('__SGS__')
if verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def timeit_debug(func):
    """This decorator prints the execution time for the decorated function."""
    if timing and verbose:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.debug(" {} ran in {}s".format(func.__name__, round(end - start, 5)))
            return result
        return wrapper
    else:
        return func

def timeit_info(func):
    """This decorator prints the execution time for the decorated function."""
    if timing:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(" {} ran in {}s".format(func.__name__, round(end - start, 5)))
            return result
        return wrapper
    else:
        return func
#####################################VISUALIZATION###############
# SETTINGS
overwrite_figures = True
page_width = 8      # A4 width in inches (minus margins)
row_height = 11/4. # A4 height in inches (minus margins) divided into 4 rows
figsize = (page_width, row_height)
nrow = 3
plot_context = 'paper'
plot_style = 'ticks'
palette = 'colorblind'
font_scale = 1.1
label_scale = 1.4


sb.set(context=plot_context, style=plot_style, palette=palette, font='sans-serif', font_scale=font_scale, color_codes=palette)
#plt.style.use(['FlexModEHC.mplrc']) # overwrite some custom adaptations
# pprint(sb.plotting_context())
toptitle_fontsize = 45
suptitle_fontsize = 35
color_background_val = 0.5
label_size = label_scale*plt.rcParams['axes.titlesize']
label_weight = plt.rcParams['axes.titleweight']

gridspec_kw = {'left':.01, 'bottom':.1, 'right':.99, 'top':.9, 'wspace':0.6, 'hspace':0.3}
cmap_state_density = plt.cm.bone_r
cmap_spec_density = plt.cm.autumn
cmap_statespaceBG = plt.cm.Greys
cmap_statespaceBG_val = 0.9
cmap_stateseq = plt.cm.cool
cmap_grid_code = plt.cm.jet
cmap_activation_prob = plt.cm.inferno
color_time_covered = 'black'
jitter_state = False
color_diff = 'red'
color_superdiff = 'blue'
color_turb = 'purple'
color_acmin = 'darkorange'

suptitle_yshift = 1.03

# graph variables
text_font_size = '70pt'
min_node_size = 150
max_node_size = 220
min_edge_size = 5
max_edge_size = 80
node_sizes = None
edge_sizes = None
min_node_lw = 1
max_node_lw = 10
node_size = (min_node_size+max_node_size)/2.
edge_size = (min_edge_size+max_edge_size)/2.
cmap_edge = None
color_index_edge = None

def remove_axes(ax):
    sb.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
######################################################################################3

# SETTINGS
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
diag_actions = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]  # UP-RIGHT, DOWN-RIGHT, UP-LEFT, DOWN-LEFT
nA = len(actions)  # assuming gridworlds
TOL = 0.001

from scipy.sparse import csr_matrix

if sparsify:
    from scipy.sparse.linalg import inv as invert_matrix
else:
    from scipy.linalg import inv as invert_matrix


@timeit_debug
def create_tensor(shape, sparse=sparsify, fill_value=0.0):
    if sparse:
        return csr_matrix(shape)
    else:
        return np.ones(shape) * fill_value


@timeit_debug
def row_norm(X):
    """L1 normalization"""
    # return X/X.sum(axis=1)
    return normalize(X.copy(), norm="l1", axis=1)  # handles zero denominators


@timeit_debug
def normalize_logsumexp(X, beta=1.0):
    """FUNCTION: Boltzmann normalization using logsumexp function.
       NOTES: Good for huge range spanning positive and negative values."""
    from scipy.special import logsumexp

    X = beta * X
    Y = X - X.min() + 1.0
    P = np.exp(np.log(Y) - logsumexp(Y))
    return P


@timeit_debug
def norm_density(V, beta=1.0, type="l1"):
    """
    FUNCTION: normalize to [0,1].
    INPUTS: V = values to normalize ("negative energies")
            beta = "inverse temperature" scaling
            type = type of normalization, L1, boltzmann
    """
    V[np.isinf(V)] = 0.0
    # shift into positive range
    # (alpha>1, sometimes results in negative Y values presumably due to precision issues)
    if (V < 0).any():
        V = V - V.min()
    if type == "l1":
        P = V / V.sum()
    elif type == "boltzmann":
        P = normalize_logsumexp(V, beta=beta)
    else:
        raise ValueError("Unknown normalization requested.")
    return P


@timeit_debug
def symnorm_graph_laplacian(X):
    from scipy.sparse.csgraph import laplacian

    return laplacian(X, normed=True)


@timeit_debug
def is_symmetric(X):
    return np.all(X.T == X)

@timeit_info
def eigen_decomp(X, real_part=False, right=True, sparse_comp=False):
    """
    FUNCTION: Computes the eigen-decomposition of X.
    INPUTS: X           = square matrix
            real_part   = suppress complex part of evals,evecs
            sparse_comp = sparse matrix computation
            right       = True, right-multiplying transition/generator matrix i.e.
                            dot(rho) = rho O
    NOTE: Eigenvectors are organized column-wise!
          Sparse format typically not faster for eigen-decomposition.
          # TODO check that numpy/scipy is compiled against openblas as this will parallelize eigen_decomp automagically.
    """
    if sparse_comp:
        import scipy.sparse.linalg as LA
        import scipy.sparse as sp

        X = sp.csr_matrix(X)
        if right:
            # right eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigsh(X)
            else:
                evals, EVECS = LA.eigs(X)
        else:
            # left eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigsh(X.T)
            else:
                evals, EVECS = LA.eigs(X.T)
    else:
        import numpy.linalg as LA

        if right:
            # right eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigh(X)
            else:
                evals, EVECS = LA.eig(X)
        else:
            # left eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigh(X.T)
            else:
                evals, EVECS = LA.eig(X.T)

    # eigenspectrum ordering from low-frequency (low abs e-vals) to high-frequency (high abs e-vals)
    ix = np.argsort(np.abs(evals))
    EVECS = EVECS[:, ix]
    evals = evals[ix]
    evals[np.abs(evals) < min_val] = 0.0

    if real_part:
        evals = np.real(evals)
        EVECS = np.real(EVECS)
    return evals, EVECS


@timeit_info
def process_eigen_grad(egrad, n_state):
    """
    Sets the eigenvector gradient.
    INPUTS:
    egrad = float the evector fraction to consider.
            positive float starts from top (low-frequency) EVECS.
            negative float starts from bottom (high-frequency) EVECS.
    OUTPUT:
    Efactors = vector of evec weights.
    """
    if hasattr(egrad, "__len__"):
        Efactors = egrad
        if Efactors.size < n_state:  #  default is to weight from low-frequency
            Efactors = np.pad(Efactors, (0, n_state - Efactors.size), "constant")
    else:
        Efactors = np.ones((np.floor(n_state * np.abs(egrad)).astype("int"),))
        if Efactors.size < n_state:
            if egrad > 0.0:  # take low-frequency EVECS
                Efactors = np.pad(Efactors, (0, n_state - Efactors.size), "constant")
            else:  # take high-frequency EVECS
                Efactors = np.pad(Efactors, (n_state - Efactors.size, 0), "constant")
    Efactors = Efactors[:n_state]
    return Efactors


def pos_dict(xymat):
    """Convert from xy matrix to pos_dict object used by networkx."""
    return {i: xymat[i, :] for i in range(xymat.shape[0])}

#######################Autocorrolation#########################################################

tol = 10 ** -3  # tolerance for bounds and linear constraints


def roll_pad(x, n, fill_value=0):
    """
    FUNCTION: rolls matrix x along columns
    """
    if n == 0:
        return x
    elif n < 0:
        n = -n
        return np.fliplr(np.pad(np.fliplr(x), ((0, 0), (n, 0)), mode='constant', constant_values=fill_value)[:, :-n])
    else:
        return np.pad(x, ((0, 0), (n, 0)), mode='constant', constant_values=fill_value)[:, :-n]


def estimate_occ_acf(data, d=0):
    """
    FUNCTION: estimate occupator autocorrelation
    INPUTS: data = (n_t, n_samp) matrix of samples
    """
    n_t = data.shape[0]
    n_samp = data.shape[1]
    n_state = data.max() + 1
    print(data)
    AC_samp = np.zeros((n_t, n_samp))
    for k in range(n_t):
        X = roll_pad(data.T, -k, (d + 1) * (n_state + 1)).T
        N = float(n_t - k)
        if d == 0:
            AC_samp[k, :] = (X == data).sum(0) / N
        else:
            AC_samp[k, :] = (np.abs(X - data) <= d).sum(0) / N
    AC = AC_samp.mean(1)
    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)
    # use variance across samples as a sample variance (does not take into account within-sample, cross-lag variance)
    return AC, AC_sem


def estimate_occ_zero_cf(data, d=0):
    """
    INPUTS: estimates occupator correlation with zero-time occupator.
    INPUTS: data = (n_t, n_samp) matrix of samples
    """
    n_t = data.shape[0]
    n_samp = data.shape[1]

    AC_samp = np.zeros((n_t, n_samp))
    for k in range(n_t):
        if d == 0:
            AC_samp[k, :] = (data[0, :] == data[k, :])
        else:
            AC_samp[k, :] = (np.abs(data[0, :] - data[k, :]) <= d)
    AC = AC_samp.mean(1)
    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)
    # use variance across samples as a sample variance (does not take into account within-sample, cross-lag variance)
    return AC, AC_sem


def estimate_episodic_acf(data, d=0):
    """
    FUNCTION: estimate episodic (time space, semantic space) autocorrelation
    INPUTS: data = (n_t, n_samp, n_vars) matrix of samples
    """

    n_samp = data.shape[0]
    n_t = data.shape[1] - 1
    n_vars = data.shape[2]

    # center data variables
    data = data - data.mean(axis=1, keepdims=True)

    AC_samp = np.zeros((n_t, n_samp))

    for t in range(n_t):
        data_roll = np.roll(data, t, axis=0)

        for samp in range(n_samp):

            Sxy = data[samp, t, :].reshape(n_vars, -1) @ data_roll[samp, t, :].reshape(-1, n_vars)
            Sxy = np.linalg.eigvals(Sxy)
            Sxy = sum([l for l in Sxy if l > 0])

            Sxx = data[samp, t, :].reshape(n_vars, -1) @ data[samp, t, :].reshape(-1, n_vars)
            Sxx = np.linalg.eigvals(Sxx)
            Sxx = sum([l for l in Sxx if l > 0])

            Syy = data_roll[samp, t, :].reshape(n_vars, -1) @ data_roll[samp, t, :].reshape(-1, n_vars)
            Syy = np.linalg.eigvals(Syy)
            Syy = sum([l for l in Syy if l > 0])

            if Sxx == 0 or Syy == 0:
                if Sxy == 0:
                    AC_samp[t, samp] = 1
                else:
                    # raise exception
                    raise ValueError('if Sxx or Syy is zero, Sxy must be zero')
            else:
                AC_samp[t, samp] = Sxy / (Sxx * Syy)

    AC = AC_samp.mean(1)

    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)

    return AC, AC_sem


def estimate_episodic_acf_v2(data, axis=None):
    """
    FUNCTION: estimate episodic (time space, semantic space) autocorrelation
    INPUTS: data = (n_t, n_samp, n_vars) matrix of samples
    """
    n_samp = data.shape[0]
    n_t = data.shape[1]
    n_vars = data.shape[2]

    AC_samp = np.zeros((n_t, n_samp, n_vars))

    if axis is None:
        axis = list(range(n_vars))
    elif type(axis) is int:
        axis = [axis]

    for var in axis:
        for samp in range(n_samp):
            acor = sm.tsa.acf(data[samp, :, var], nlags=n_t - 1, fft=False)
            AC_samp[:, samp, var] = np.abs(acor)

    # AC_samp = AC_samp[:, :, axis].reshape(n_t, n_samp, -1).mean(axis=2)
    AC_samp = AC_samp[:, :, axis].reshape(n_t, n_samp, -1)
    AC = AC_samp.mean(axis=1)

    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)

    return AC, AC_sem




