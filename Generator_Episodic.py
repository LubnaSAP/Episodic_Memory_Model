# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from utils import (row_norm, eigen_decomp, is_symmetric, symnorm_graph_laplacian, timeit_debug,
                   timeit_info, min_val)


class Generator(object):
    @timeit_info
    def __init__(
        self,
        ENV=None,
        Q=None,
        T=None,
        W=None,
        jump_rate=15.0,
        forward=True,
        symmetrize=False,
    ):
        """
        FUNCTION: Constructs eigensystem for infinitesimal generator corresponding to underlying state-space.
        INPUTS: ENV = environment with accessibility matrix ENV.A
                Q = explicit generator
                T = stochastic matrix
                W = weight matrix
                jump_rate = rate at which particle jumps from current state (higher = more likely to jump)
                            (only applies if generator is being constructed from DTMC e.g. T/A)
                forward = True, forward in time => right-multiplying matrix
                symmetrize = symmetrize generator
        NOTES: State-space can be specified in terms of an environment ENV, generator Q, stochastic matrix T, or weight matrix W.
        Primacy given to environment variable ENV.
        """
        if ENV is not None:
            if hasattr(ENV, "T"):
                self.T = ENV.T
                Q = stochmat2generator(T=self.T, jump_rate=jump_rate)
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from environment transition matrix with jump_rate %.2f"
                    % jump_rate
                )
            else:
                Q = adjmat2generator(ENV.A_adj, jump_rate=jump_rate)
                self.T = generator2stochmat(Q)
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from environment adjacency matrix with jump_rate %.2f"
                    % jump_rate
                )
            self.ENV = ENV
        elif Q is None:
            if T is not None:
                Q = stochmat2generator(T=T, jump_rate=jump_rate)
                self.T = T
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from arbitrary transition matrix with jump_rate %.2f"
                    % jump_rate
                )
            else:
                Q = weightmat2generator(W=W, normsym=symmetrize)
                self.W = W
                self.T = generator2stochmat(Q)
                print("GENERATOR: generator constructed from arbitrary weight matrix")
        else:
            print("GENERATOR: explicit generator provided")
        # record variables
        self.Q = Q
        self.n_state = self.Q.shape[0]
        self.jump_rate = jump_rate
        self.forward = forward
        self.symmetrize = symmetrize
        self.process_generator()



    def process_generator(self, Q=None, check=True):
        if Q is not None:
            self.Q = Q

        if check:
            self._check_generator()

        # eigen_decompositions
        evals_fwd, EVEC_fwd = eigen_decomp(
            self.Q, right=True
        )  # propagates forward in time
        evals_bwd, EVEC_bwd = eigen_decomp(
            self.Q, right=False
        )  # propagates backward in time
        self.EVEC_fwd = EVEC_fwd
        self.EVECinv_fwd = LA.inv(EVEC_fwd)
        self.evals_fwd = evals_fwd
        self.EVEC_bwd = EVEC_bwd
        self.EVECinv_bwd = LA.inv(EVEC_bwd)
        self.evals_bwd = evals_bwd
        evals_info(self.evals_fwd)

        # polar coordinates
        self.eradians_fwd = np.angle(self.evals_fwd)
        self.edegrees_fwd = np.angle(self.evals_fwd, deg=True)
        self.eradii_fwd = np.abs(evals_fwd)
        self.eradians_bwd = np.angle(self.evals_bwd)
        self.edegrees_bwd = np.angle(self.evals_bwd, deg=True)
        self.eradii_bwd = np.abs(evals_bwd)



    def _check_generator(self):
        """
        Checks whether Q is a generator.
        """
        self.Q = check_generator(self.Q, symmetrize=self.symmetrize)


@timeit_debug
def stochmat2generator(T, jump_rate=10.0):
    """
    Returns the CTMC generator defined by the embedded DTMC T and
    jump intensity jump_rate (equiv. time constant).
    T            = DTMC stochastic matrix
    jump_rate    = jump intensity parameter (higher is more jumps)
    """
    assert np.allclose(T.sum(1), 1), "rows of T do not sum to 1"
    Q = jump_rate * (T - np.eye(T.shape[0]))
    check_generator(Q)
    return Q


@timeit_debug
def weightmat2generator(W, normsym=True):
    """
    FUCNTION:
        Returns the CTMC generator defined by graph weights W (can be negative).
    INPUTS:
        W       = weight matrix for graph
        normsym = returns normalized symmetric graph Laplacian, otherwise standard W generator
    Equals symmetric normalized graph Laplacian.

    """
    check_weightmat(W)
    if normsym:
        Q = -symnorm_graph_laplacian(W)
        Q = set_generator_diagonal(Q)
    else:
        Q = set_generator_diagonal(W)
    # check_generator(Q)
    return Q


def adjmat2generator(A, jump_rate=1.0):
    """
    Returns the CTMC generator defined by diffusion on graph with weighted
    adjacency matrix A and jump intensity jump_rate (equiv. time constant).
    A           = weighted adjacency matrix for graph
    jump_rate    = jump intensity parameter (somewhat redundant so defaults to 1)
    """
    assert (
        np.diag(A) == 0
    ).all(), "Adjacency matrix should be zero on the diagonal (no self-adjacencies)"
    Q = jump_rate * A.astype("float")
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def generator2weightmat(Q):
    W = deepcopy(Q.astype("float"))
    W[np.eye(W.shape[0], dtype=bool)] = 0.0
    return W


@timeit_debug
def generator2stochmat(Q, tau=0.0, zero_diag=True):
    """
    FUNCTION: CTMC generator to DTMC transition matrix.
    INPUTS: Q           = generator
            tau         = prior on transition probability
            zero_diag   = zero out diagonal
    """
    T = Q.astype("float").copy()
    if zero_diag:
        T[np.eye(T.shape[0]).astype("bool")] = 0
    else:
        jump_rate = np.diagonal(T)
        T = T / jump_rate + np.eye(T.shape)
    T = row_norm(T)
    T = row_norm(T + tau)
    return T


def symmetrize_generator(Q):
    Qsym = (Q + Q.T) / 2.0
    Qsym = set_generator_diagonal(Qsym)
    return Qsym


def set_generator_diagonal(Q):
    Q[np.eye(Q.shape[0], dtype=bool)] = 0.0
    # Q = np.round(Q,10)
    for i in range(Q.shape[0]):
        Q[i, i] = -np.sum(Q[i, :])
    # Q[np.abs(Q)<10**-10] = 0.
    return Q


def check_generator(Q, symmetrize=False):
    """
    Checks whether Q is a generator.
    """
    is_gen = True
    if not np.allclose(Q.sum(1), 0):
        print("GENERATOR: matrix rows do not sum to 0.")
        is_gen = False
    else:
        print("GENERATOR: matrix rows sum to 0.")
    if np.any(Q[~np.eye(Q.shape[0], dtype=bool)] < 0.0):
        print("GENERATOR: some matrix off-diagonals are negative.")
        is_gen = False
    if np.any(np.diag(Q) > 0.0):
        print("GENERATOR: some matrix diagonals are non-negative.")
        is_gen = False
    if is_symmetric(Q):
        print("GENERATOR: generator is symmetric.")
    else:
        print("GENERATOR: generator is not symmetric.")
        if symmetrize:
            Q = symmetrize_generator(Q)
            assert is_symmetric(Q)
            print("GENERATOR: generator symmetrized.")
    if is_gen:
        print("GENERATOR: Q is a generator with shape", Q.shape, ".")
    else:
        raise ValueError("GENERATOR: Q is not a generator.")

    return Q

def evals_info(evals):
    """Prints eigenvalue information."""
    print(
        "EIGENSPECTRUM: algebraic multiplicity of zero eigenvalue =",
        np.sum(evals == 0.0),
    )
    if np.unique(evals).size != evals.size:
        unique, counts = np.unique(evals, return_counts=True)
        comb = np.vstack((unique[counts > 1], counts[counts > 1])).T
        print("EIGENSPECTRUM: algebraic multiplicity > 1.")
    if LA.norm(evals[np.iscomplex(evals)]) > min_val:
        print("EIGENSPECTRUM: complex eigenvalues:", evals[np.iscomplex(evals)])
    if np.any(np.real(evals) > 0.0):
        print("EIGENSPECTRUM: real components of eigenvalues in positive domain:")
        print((evals[evals > 0.0]))

def check_weightmat(W):
    """
    Checks whether W is a weight matrix.
    """
    if not np.all(np.diag(W) == 0):
        raise ValueError("WEIGHT MATRIX: nonzero on the diagonal.")
    if not is_symmetric(W):
        raise ValueError("WEIGHT MATRIX: not symmetric.")