import numpy as np
import torch
import torch.nn as nn
from scipy.stats import laplace, norm


class Postprocessing(nn.Module):

    def __init__(self, dim, return_probs=True, uncert="std", method="raw"):
        """
        Construct the Prior Layer translating instantaneous bin probabilities into contextualized HR predictions.
        :param dim: number of bins
        :param return_probs: returns contextualized bin probabilities if set to True, HR estimates in BPM otherwise.
        :param uncert: The uncertainty measure to use. One of ["entropy", "std"].
        :param kwargs: passed to parent class
        """
        super(Postprocessing, self).__init__()
        self.dim = dim
        self.method = method
        self.return_probs = return_probs
        self.uncert = uncert
        bins = np.array([i for i in [-3/dim] + list(np.linspace(0, 1, dim-1)) + [1+3/dim]])
        bins = (bins[1:] + bins[:-1])/2
        self.bins = nn.Parameter(torch.tensor(bins, dtype=torch.float32), requires_grad=False)


    def fit_layer(self, ys, distr="laplace", sparse=False, learn_state_prior=False):
        pass


    def forward(self, probs, preds, method=None):
        method = method or self.method
        if method == "raw":
            # return input
            return self.forward_raw(probs, preds)
        else:
            raise NotImplementedError(f"Unknown method: {method}")
        
    def forward_raw(self, probs, preds):
        """
        Returns the raw input probabilities.
        :param ps: tf.tensor of shape (n_samples, n_bins) containing probabilities
        :return: probs : tf.tensor of same shape, only returned if return_probs=True
                 E_x : tf.tensor of shape (n_samples,) containing the expected HR, only if return_probs=False
                 uncert : tf.tensor of shape (n_samples,) containing est. uncertainty of the prediction
        """
        uncert = self._compute_uncertainty(probs)

        if self.return_probs:
            return preds, uncert, probs
        else:
            
            return preds, uncert

    def _compute_expectation(self, probs):
        E_x = torch.sum(probs * self.bins[None, :], axis=1)
        return E_x
    
    def _compute_uncertainty(self, probs):
        if self.uncert == "std":
            E_x = self._compute_expectation(probs)
            E_x2 = torch.sum(probs * self.bins[None, :] ** 2, axis=1)
            uncert = torch.sqrt(E_x2 - E_x**2)
        elif self.uncert == "entropy":
            uncert = -torch.sum(probs * torch.log(probs + 1e-10), axis=1)
        else:
            raise NotImplementedError(f"Unknown uncertainty measure: {self.uncert}")
        
        return uncert

class BeliefPPG(Postprocessing):
    """
    Implements functionality for the belief propagation / decoding framework.
    The layer is intended to be added as last layer to a trained network with instantaneous HR
     bin probabilities as output. It should be fit separately on training data before doing so.
     When added, the model produces either contextualized probabilities or BPM HR estimates as output.
    """

    def __init__(self, dim, return_probs=True, uncert="std", method="sumprod"):
        """
        Construct the Prior Layer translating instantaneous bin probabilities into contextualized HR predictions.
        :param dim: number of bins
        :param return_probs: returns contextualized bin probabilities if set to True, HR estimates in BPM otherwise.
        :param uncert: The uncertainty measure to use. One of ["entropy", "std"].
        :param kwargs: passed to parent class
        """
        super(BeliefPPG, self).__init__(dim, return_probs, uncert, method)
        self.state_prior = torch.tensor(np.ones(dim) / dim, dtype=torch.float32)
        self.state = nn.Parameter(self.state_prior.clone(), requires_grad=False)
        #self.dim = dim
        #self.method = method
        self.transition_prior = nn.Parameter(torch.zeros((self.dim, self.dim), dtype=torch.float32), requires_grad=False)
        #self.return_probs = return_probs
        #self.uncert = uncert
        #bins = np.array([i for i in [-3] + list(np.linspace(0, dim, dim-1)) + [dim+3]])
        #bins = (bins[1:] + bins[:-1])/2
        #self.bins = nn.Parameter(torch.tensor(bins, dtype=torch.float32), requires_grad=False)
        #self.bins = nn.Parameter(torch.tensor([i for i in [-3] + list(np.linspace(0, dim, dim-1)) + [dim+3]], dtype=torch.float32), requires_grad=False)



    def _fit_distr(self, diffs, distr):
        """
        Fits a Gaussian/Laplacian prior to heart rate changes.
        :param diffs: heart rate changes in BPM for consecutive 8s windows with 2s shift
        :param distr: uses Laplacian distribution when set to true. Default is Gaussian. For the Gaussian,
        differences are assumed to be log differences.
        :return: mean and stddev of fitted Gaussian/Laplacian
        """
        if distr == "laplace":
            mu, sigma = laplace.fit(diffs)
        else:
            mu, sigma = norm.fit(diffs)
        return mu, sigma

    def fit_layer(self, ys, distr="laplace", sparse=False, learn_state_prior=False):
        """
        Precomputes a prior matrix based on heart rate changes.
        :param ys: list of ground truth HR values with same strides as labels
        :param distr: whether to fit a Laplacian or Gaussian on the (log) diffs
        :param sparse: whether to round very low probabilities down to zero.
                    Results in theoretically higher efficiency.
        """
        if distr == "laplace":
            diffs = np.concatenate([y[1:] - y[:-1] for y in ys], axis=0)
        elif distr == "gauss":
            diffs = np.concatenate([np.log(y[1:]) - np.log(y[:-1]) for y in ys], axis=0)
        else:
            raise NotImplementedError(r"Unknown prior %s" % distr)

        mu, sigma = self._fit_distr(diffs, distr)

        for i in range(self.dim):
            for j in range(self.dim):
                if sparse and abs(i - j) > 10 * 60 / self.dim:
                    self.transition_prior[i][j] = 0.0
                else:
                    if distr == "laplace":
                        self.transition_prior[i][j] = laplace.cdf(
                            abs(i - j) + 1, mu, sigma
                        ) - laplace.cdf(abs(i - j) - 1, mu, sigma)
                    elif distr == "gauss":
                        log_diffs = [
                            np.log(i1) - np.log(i2)
                            for i1 in (i - 0.5, i + 0.5)
                            for i2 in (j - 0.5, j + 0.5)
                        ]
                        max_logdiff = np.max(log_diffs)
                        min_logdiff = np.min(log_diffs)
                        self.transition_prior[i][j] = norm.cdf(max_logdiff, mu, sigma) - norm.cdf(
                            min_logdiff, mu, sigma
                        )

        # no need for normalization, probability leaks are handled during forward propagation
                        
        if learn_state_prior:
            state_prior = torch.concat(ys).sum(0)
            state_prior = state_prior / state_prior.sum()
            self.state_prior = state_prior.to(torch.float32)

    def _propagate_sumprod(self, ps):
        """
        Performs online belief propagation i.e. sum-product message passing
        New probabilities are calculated as T.p_{old} * p_{pred} and normalized, where * is the Hadamard product.
        :param ps: ps: tf.tensor of shape (n_samples, n_bins) containing probabilities
        :return: tf.tensor of same shape containing updated probabilities
        """
        # Since we assume that every batch is indipendent of the others, we can reset the state at the beginning of each batch
        self.state.data = self.state_prior.clone().to(ps.device)
        output = []
        for p in ps:
            # propagate (blurred) last observations
            p_prior = torch.matmul(self.transition_prior, self.state)
            # add current observations
            p_new = (p_prior * p) + 1e-10
            self.state.data = p_new / torch.sum(p_new)
            output.append(self.state.clone().detach())
        return torch.stack(output)

    def forward(self, probs, preds, method=None):
        method = method or self.method
        if method == "sumprod":
            return self.forward_sumprod(probs, preds)
        elif method == "viterbi":
            return self.forward_viterbi(probs, preds)
        elif method == "raw":
            # return input
            return self.forward_raw(probs, preds)
        else:
            raise NotImplementedError(f"Unknown method: {method}")
        
    
    def _update_prob(self, prev_maxprod, curr, trans_prob):
        """
        Applies one timestep of the forward pass of Viterbi Decoding.
        :param prev_maxprod: np.ndarray of shape (n_bins,) containing probabilities of most probable path per HR bin
        :param curr: np.ndarray of shape (n_bins,) containing the raw model beliefs for the current timestep
        :param trans_prob: np.ndarray of shape (n_bins, n_bins) containing transition probability matrix expressing prior
        :return: tuple (new_maxprod, ixes) containing new best path scores and backpointers respectively
        """
        curr_maxprod = torch.empty_like(prev_maxprod)
        ixes = torch.empty_like(prev_maxprod, dtype=torch.long)
        for i in range(len(curr)):
            curr_maxprod[i] = torch.max(prev_maxprod * trans_prob[:, i])
            ixes[i] = torch.argmax(prev_maxprod * trans_prob[:, i])

        curr_maxprod *= curr
        return curr_maxprod / curr_maxprod.sum(), ixes

    def forward_viterbi(self, probs, preds=None):
        """
        Performs Viterbi Decoding on the output probabilities.
        That is, uses max-product message passing to find the most likely HR trajectory according to the
        raw class probabilites.
        :param probs: np.ndarray of shape (n_timesteps, n_bins) of raw HR probabilities
        :param prior_layer: PriorLayer that has already been fit to training data
        :return: np.ndarray of shape (n_timesteps, ) of predictions for each step
        """

        best_paths = []
        prev_maxprod = torch.full((self.dim,), 1 / self.dim, dtype=torch.float32, device=probs.device)

        for j in range(len(probs)):
            prev_maxprod, paths = self._update_prob(prev_maxprod, probs[j], self.transition_prior)
            best_paths.append(paths)

        best_path = []
        curr_ix = torch.argmax(prev_maxprod).item()

        for i in range(len(best_paths) - 1):
            best_path.append(curr_ix)
            curr_ix = best_paths[-(i + 1)][curr_ix].item()

        best_path.append(curr_ix)

        pred_indices = torch.tensor([x for x in reversed(best_path)], dtype=torch.int64, device=probs.device)
        pred = self.bins[pred_indices]

        # for compatibility, we also return uncertainty, which is all 0 here
        # also probability, which is simply the one-hot encoded prediction
        uncertainty = torch.zeros(len(pred), dtype=torch.float32)


        if self.return_probs:
            probs = torch.zeros((len(pred), self.dim), dtype=torch.float32)
            for i, ix in enumerate(best_path):
                probs[i, ix] = 1.0
            return pred, uncertainty, probs

        return pred, uncertainty

    def forward_sumprod(self, probs, preds=None):
        """
        Calculates a stateful forward pass applying the transition prior (symbolically).
        Assumes batch consists of consecutive samples. Overrides parent function.
        :param probs: tf.tensor of shape (n_samples, n_bins) containing probabilities
        :return: probs : tf.tensor of same shape, only returned if return_probs=True
                 E_x : tf.tensor of shape (n_samples,) containing the expected HR, only if return_probs=False
                 uncert : tf.tensor of shape (n_samples,) containing est. uncertainty of the prediction
        """
        probs = self._propagate_sumprod(probs)

        E_x = self._compute_expectation(probs)

        uncert = self._compute_uncertainty(probs)

        if self.return_probs:
            return E_x, uncert, probs
        else:
            return E_x, uncert
        
    def get_config(self):
        """
        Helper function necessary to save model
        :return: dict config
        """
        config = {
            "dim": self.dim,
            "return_probs": self.return_probs,
            "method": self.method,
        }
        return config
    
    def _compute_expectation(self, probs):
        E_x = torch.sum(probs * self.bins[None, :], axis=1)
        return E_x
    
    def _compute_uncertainty(self, probs):
        if self.uncert == "std":
            E_x = self._compute_expectation(probs)
            E_x2 = torch.sum(probs * self.bins[None, :] ** 2, axis=1)
            uncert = torch.sqrt(E_x2 - E_x**2)
        elif self.uncert == "entropy":
            uncert = -torch.sum(probs * torch.log(probs + 1e-10), axis=1)
        else:
            raise NotImplementedError(f"Unknown uncertainty measure: {self.uncert}")
        
        return uncert



from filterpy.kalman import KalmanFilter,rts_smoother
from filterpy.common import Q_discrete_white_noise

class KalmanSmoothing(Postprocessing):
    def __init__(self, dim, step_size=8, noise=2, Q=1e-4, return_probs=True, uncert="std"):
        super(KalmanSmoothing, self).__init__(dim, return_probs, uncert)
        self.fk = KalmanFilter(dim_x=2, dim_z=1)
        self.fk.x = np.array([0.5, 0.])      # initial state (x and dx)
        self.fk.F = np.array([[1., step_size],
                            [0., 1.]])    # state transition matrix
        self.fk.H = np.array([[1., 0.]])    # Measurement function
        self.fk.P*= 100.                     # covariance matrix
        self.fk.R = noise                   # state uncertainty
        self.fk.Q = Q_discrete_white_noise(dim=2, dt=step_size, var=Q)  # process uncertainty

        
    
    def smooth(self, zs):
        mu, cov, _, _ = self.fk.batch_filter(zs)
        M, p, C, _ = self.fk.rts_smoother(mu, cov)
        return M[:,0]
    
    def forward(self, probs, preds):

        smoothed_expectation = self.smooth(preds)
        uncertainty = torch.zeros(len(preds), dtype=torch.float32)

        smoothed_expectation = torch.tensor(smoothed_expectation, dtype=torch.float32)

        if self.return_probs:
            probs = torch.zeros((len(smoothed_expectation), self.dim))
            for i, x in enumerate(smoothed_expectation):
                probs[i, int(x)] = 1
            return smoothed_expectation, uncertainty, probs

        return smoothed_expectation, uncertainty
    


def test_PriorLayer():
    # Define parameters
    dim = 10

    return_probs = True
    uncert = "entropy"

    # Create an instance of PriorLayer
    prior_layer = Postprocessing(dim, return_probs, uncert)

    # Generate random input data
    num_samples = 5
    num_bins = dim
    input_data = np.random.rand(num_samples, num_bins)

    # Fit the layer (dummy data, just for testing purposes)
    ys = [np.random.rand(num_samples) for _ in range(num_samples)]
    distr = "laplace"
    prior_layer.fit_layer(ys, distr)

    # Perform forward pass
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_probs, output_uncert = prior_layer(input_tensor)

    # Print output shapes (for demonstration)
    print("Output probabilities shape:", output_probs.shape)
    print("Output uncertainty shape:", output_uncert.shape)

    # Optionally, perform assertions or additional checks here
    # For instance, you could verify output shapes or check values within a certain range

class ViterbiDecoder(torch.nn.Module):
    def __init__(self, prior_layer):
        super(ViterbiDecoder, self).__init__()
        self.prior_layer = prior_layer


    
