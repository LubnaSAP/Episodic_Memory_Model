#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from utils import process_eigen_grad, timeit_debug, timeit_info


class Propagator(object):
    """
    FUNCTION: Propagates, enough said.
    INPUTS: GEN = generator.
    """

    @timeit_info
    def __init__(
        self,
        GEN,
        nu=None,
        sigma=1.0,
        tau=1.0,
        alpha=1.0,
        beta=1.0,
        spec_noise=0.0,
        power_spec=None,
        strict=False,
        label=None,
    ):
        """
        FUNCTION: Processes multiple Q-generators.
        INPUTS: GEN         = generator instance
                nu          = if not None, sigma scaling as a function of n_state (over-rides sigma)
                sigma       = spatial constant (diffusion speed/scale is correlated with sigma^2)
                tau         = tempo parameter (diffusion speed/scale is inversely correlated with tau)
                alpha       = stability parameter
                beta        = softmax normalization temperature
                spec_noise  = add zero-mean noise to power spectrum with variance spec_noise
                power_spec    = use specific power spectrum
                strict      = strict checks on propagator kernel
                label       = propagator description (e.g. diffusion/superdiffusion/turbulence), if None determined by alpha
        NOTES: inverse tau  = amount of time per circuit iteration
               sigma and tau are not exactly inversely related in anomalous regimes
        """
        self.GEN = GEN
        self.nu = nu
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.spec_noise = float(spec_noise)
        self.strict = strict
        self.states = np.array(list(range(self.n_state)))
        if label is None:
            if alpha == 1:
                self.label = "diffusion"
            elif alpha < 1:
                self.label = "superdiffusion"
            elif alpha > 1:
                self.label = "turbulence"
        else:
            self.label = label

        if nu is not None:
            assert 0 < self.nu <= 1, "PROPAGATOR: nu out of range"
            self.sigma = np.sqrt(self.nu * self.n_state / 2.0)

        assert 0 < self.sigma, "PROPAGATOR: sigma out of range"
        assert 0 < self.tau, "PROPAGATOR: tau out of range"
        assert 0 < self.alpha, "PROPAGATOR: alpha out of range"
        assert 0 < self.beta <= 1, "PROPAGATOR: beta out of range"

        # secondary parameters
        self.secondary_params()

        # construct propagator
        self.compute_kernels(power_spec=power_spec, strict=strict)


    @property
    def n_state(self):
        return self.GEN.n_state

    @property
    def n_dim(self):
        return self.ENV.n_dim

    @property
    def ENV(self):
        return self.GEN.ENV

    @property
    def world_array(self):
        return self.GEN.ENV.world_array

    @property
    def U(self):
        return self.GEN.EVEC_fwd

    @property
    def Uinv(self):
        return self.GEN.EVECinv_fwd

    @property
    def L(self):
        return self.GEN.evals_fwd


    def secondary_params(self):
        """ computes secondary parameters from primary sigma/alpha/tau/L """
        # factor of 2 since "gaussian" parametrized as alpha=1 in discrete case
        self.sigma_alpha = self.sigma ** (2 * self.alpha)
        # alpha-modulated diffusion constant
        self.K_alpha = self.sigma_alpha / self.tau
        # scale eigenvalues by stability parameter alpha
        self.L_alpha = np.abs(self.L) ** (self.alpha)

    @timeit_debug
    def spectral_density(self, t=1, k=None):
        """computes the spectral density as a function of time displacement t and spectral component k
        depends on alpha/beta/sigma and generator eigenvalues"""
        assert hasattr(self, "L_alpha"), "L_alpha unavailable?"
        # dilation in frequency space
        x = self.sigma_alpha * self.L_alpha * t / self.tau
        d = np.exp(-x)
        if k is None:
            return d
        else:
            return d[k]

    @timeit_debug
    def weight_spec_comps(self, eigen_grad=None):
        """
        FUNCTION: Weights propagator kernels.
        INPUTS: eigen_grad = desired gradient on eigen-decomposition
                None implies all weights equal to 1
        NOTES:
        """
        self.eigen_grad = eigen_grad
        self.n_kernels = self.n_state
        if eigen_grad is None:
            self.spec_comp_weights = np.ones((self.n_kernels,))
        else:
            self.spec_comp_weights = process_eigen_grad(self.eigen_grad, self.n_state)

    @timeit_debug
    def set_power_spec(self, power_spec=None):
        if power_spec is None:
            self.power_spec = self.spectral_density(t=1.0)
        else:
            assert power_spec.size == self.n_state, "power spectrum wrong shape"
            self.power_spec = power_spec
        if self.spec_noise != 0.0:
            self.power_spec_noisefree = self.power_spec.copy()
            self.power_spec *= np.random.normal(
                loc=1, scale=self.spec_noise, size=self.power_spec.size
            )
            self.power_spec = self.power_spec.clip(
                self.power_spec_noisefree.min(), None
            )

    @timeit_debug
    # @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def compute_kernels(
        self,
        power_spec=None,
        suppress_imag=True,
        strict=False,
        atol=1.0e-2,
        rtol=1.0e-2,
    ):
        """
        FUNCTION: Computes propagator kernels.
        INPUTS: power_spec = power spectrum, None implies computed from alpha/tau etc
                suppress_imag = suppress any imaginary components
                strict = True implies extra checks on propagator structure
        NOTES: Depends on sigma/tau/alpha
        """
        # self.L, self.U, self.Uinv set as properties
        # set propagator kernel weights
        self.weight_spec_comps()
        self.set_power_spec(power_spec=power_spec)
        self.etD = (
            np.eye(self.n_state) * self.power_spec
        )  # spectral power as a diagonal matrix
        # re-weighting
        self.wetD = self.spec_comp_weights * self.etD
        # map onto basis set in frequency space "decayed" in time
        self.spec_basis = np.matmul(self.U, self.wetD)
        # propagator (i.e. map to basis set in state-space future time)
        self.etO = np.matmul(self.spec_basis, self.Uinv)
        if suppress_imag:
            if not np.all(np.isreal(self.etO)):
                etO_complex = deepcopy(self.etO)
                print("PROPAGATOR: squashing imaginary components.")
                # self.etO = self.etO.real
                self.etO = np.abs(self.etO)  # seems to work better
                if strict:
                    assert np.allclose(
                        self.etO, etO_complex
                    ), "PROPAGATOR: propagator kernel is complex."
        if strict:
            assert np.allclose(self.etO.min(), 0, atol=atol, rtol=rtol), (
                "PROPAGATOR: propagator taking values %.2f significantly <0"
                % self.etO.min()
            )
            assert (self.etO <= 1).all(), "PROPAGATOR: propagator kernel values > 1."
            assert np.allclose(
                self.etO.sum(1), 1, atol=atol, rtol=rtol
            ), "PROPAGATOR: probability density not conserved."
        self.activation_matrix()


    def activation_matrix(self, thresh=0.1):
        """ self.AMT = thresholded activation matrix at self.etO>thresh"""
        assert 0 < thresh < 1, "thresh parameter out of bounds"
        self.thresh = thresh
        self.AMT = self.etO >= thresh


    def plot_activation_matrix(self):
        """ Plot the activation matrix. """


        if hasattr(self, 'AMT'):
            print("Shape of self.AMT:", self.AMT.shape)
            if self.AMT.ndim == 2:
                plt.figure(figsize=(10, 8))
                sns.heatmap(self.AMT, cmap="viridis", annot=False, cbar=True)
                plt.title("Thresholded Activation Matrix")
                plt.xlabel('Nodes')
                plt.ylabel('Nodes')
                plt.show()
            else:
                print("Error: self.AMT is not a 2D matrix.")
        else:
            print("Error: self.AMT is not defined.")

    def plot_et0_matrix(self):
        """Plots et0 matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.etO, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("et0 matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)
