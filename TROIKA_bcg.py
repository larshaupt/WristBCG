
#%%
import numpy as np
from numpy.lib.histograms import _hist_bin_sqrt
from numpy.polynomial import Polynomial
from pprint import pprint
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.linalg import hankel
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize
from tqdm import tqdm
import cr.sparse.cvx.focuss as focuss

#%%
import pandas as pd

def ssa(ts: np.ndarray, L: int, perform_grouping: bool = True, wcorr_threshold: float = 0.3, ret_Wcorr: bool = False):
    """
    Performs SSA on ts
    https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

    Parameters
    ----------
        ts : ndarray of shape (n_timestamps, )
            The time series to decompose
        L : int
            first dimension of the L-trajectory-matrix
        grouping : bool, default=True
            If True, perform grouping based on the w-correlations of the deconstructed time series
            using agglomerative hierarchical clustering with single linkage.
        wcorr_threshold : float, default=0.3
            The w-correlation threshold used with the agglomerative hierarchical clustering.
        ret_Wcorr : bool, default=False
            Whether the resulting w-correlation matrix should be returned.
        

    Returns
    ----------
        Y : ndarray of shape (n_groups, n_timestamps) if grouping is enabled and (L, n_timestamps) if it is disabled.
        Wcorr : ndarray
            The w-correlation matrix.
            Wcorr will only be returned if ret_Wcorr is True
    """

    N = len(ts)
    K = N - L + 1
    L_trajectory_matrix = hankel(ts[:L], ts[L-1:])  # (L, K)

    U, Sigma, V = np.linalg.svd(L_trajectory_matrix, full_matrices=False)  # (L, L); (d, ); (K, K)

    V = V.T  # (K, K)
    d = len(Sigma)


    deconstructed_ts = []
    for i in range(d):
        X_elem = np.array(Sigma[i] * np.outer(U[:, i], V[:, i]))  # (L, K)
        X_elem_rev = X_elem[::-1]  # (L, K)
        ts_i = np.array([X_elem_rev.diagonal(i).mean() for i in range(-L+1, K)])
        deconstructed_ts.append(ts_i)
    deconstructed_ts = np.array(deconstructed_ts)  # (d, N)

    if not perform_grouping and not ret_Wcorr:
        return deconstructed_ts


    w = np.concatenate((np.arange(1, L + 1), np.full((K - L,), L), np.arange(L - 1, 0, -1)))
    Wcorr_mat = pairwise_distances(deconstructed_ts, metric=lambda x, y: np.dot(w*x, y) / (np.linalg.norm(w*x) * np.linalg.norm(w*y)))

    if not perform_grouping:
        return deconstructed_ts, Wcorr_mat


    Wcorr_mat_dist = 1 - Wcorr_mat
    distance_threshold = 1 - wcorr_threshold
    agg_clust = AgglomerativeClustering(metric='precomputed', linkage='single',
                                        distance_threshold=distance_threshold, n_clusters=None)
    clust_labels = agg_clust.fit_predict(Wcorr_mat_dist)
    n_clusters = clust_labels.max() + 1
    grouped_ts = [np.sum(deconstructed_ts[clust_labels == cluster_id], axis=0)
                  for cluster_id in range(n_clusters)]
    grouped_ts = np.array(grouped_ts)

    if not ret_Wcorr:
        return grouped_ts

    Wcorr_mat = pairwise_distances(grouped_ts, metric=lambda x, y: np.dot(w*x, y) / (np.linalg.norm(w*x) * np.linalg.norm(w*y)))

    return grouped_ts, Wcorr_mat


class SSR:
    def __init__(self, M: int, N: int, f_s: float, p : float = 0.8, n_iter:int = 5, inverse_method='FOCUSS'):

        self.f_s = f_s
        self.N = N
        self.M = M
        self.p = p
        self.n_iter = n_iter

        if inverse_method.lower() == 'focuss':
            self.inv_func = self.FOCUSS
        else:
            self.inv_func = self.matrix_inverse

        if N is None:
            # resolution is set to 1 BPM
            N = 60 * f_s
        # construct Fourier matrix

        ns = np.arange(0, N)

        m, n = np.meshgrid(np.arange(M), ns, indexing='ij')
        self.Phi = np.exp(1j * 2 * np.pi / N * m * n)

        self.BPM = 60 * f_s / N * (ns)

    def transform(self, y: np.ndarray) -> np.ndarray:
        # calculate BPM-axis
        
        # construct sparse spectrum
        s = self.inv_func(self.Phi, y, p=self.p, n_iter=self.n_iter)
        # Band-pass frequencies in BPM
        BPM_bp = np.array([40, 200])
        # Band-pass spectrum
        s = self.BP(s, self.f_s, BPM_bp)
        return self.BPM, s

    def FOCUSS(self, Phi, y, p=1, n_iter=5):
        x = np.linalg.pinv(Phi) @ y
        #res = []
        for i in range(n_iter):
            x = focuss.step_noiseless(Phi, y, x, p=p)

            x_hat = np.abs(x)
            periodogram = x_hat
            #frequencies = ns / N * f_s
            #plt.plot(frequencies *60,periodogram, label=i)
            sparsity = np.linalg.norm(x, ord=p)
            estimation_quality = np.linalg.norm(Phi @ x - y, ord=p)
            #res.append((sparsity, estimation_quality))

            #print(f"Sparsity: {sparsity}, estimation quality: {estimation_quality}")
        #plt.legend()
        return x_hat

    def matrix_inverse(self, Phi, y, p = 1, n_iter = 5):
        # initialization of x
        x = np.ones((Phi.shape[1], 1))
        for _ in range(n_iter):
            W_pk = np.diag(x.flatten())
            q_k = self.MPinverse(np.dot(Phi, W_pk)).dot(y)
            x = np.dot(W_pk, q_k)
            sparsity = np.linalg.norm(x, ord=p)
            estimation_quality = np.linalg.norm(Phi @ x - y, ord=p)
            print(f"Sparsity: {sparsity}, estimation quality: {estimation_quality}")
        s = np.abs(x.flatten())**2
        return s

    def MPinverse(self, A):
        matrix = np.conj(A.T)
        A_plus = np.dot(matrix, np.linalg.inv(np.dot(A, matrix)))
        return A_plus

    def BP(self, SSRsig, Fs, BPM_bp):
        N = len(SSRsig)
        f_lo = BPM_bp[0] / 60
        f_hi = BPM_bp[1] / 60
        R = np.arange(int(np.floor(f_lo / Fs * N)), int(np.ceil(f_hi / Fs * N)) + 1)
        H = np.zeros(N)
        H[R] = 1
        SSRsig = SSRsig * H
        return SSRsig

class PeakFinder:
    def __init__(self) -> None:
        pass

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        #print(spectrum.shape)
        peaks = scipy.signal.find_peaks(spectrum)[0]
        peaks_sorted = peaks[np.argsort(spectrum[peaks])[::-1]]
        return peaks_sorted[0]

    def transform_first(self, spectrum: np.ndarray) -> np.ndarray:
        return self.transform(spectrum)

class SpectralPeakTracker:
    def __init__(self, n_freq_bins=4096, ppg_sampling_freq=125, delta_s_rel=16/4069, eta=.3, tau=0.00048828125, theta=0.00146484375, init_freq_bounds=None):
        self.n_freq_bins = n_freq_bins
        self.ppg_sampling_freq = ppg_sampling_freq
        
        # parameters for stage 2 (peak selection)
        self.delta_s = int(n_freq_bins * delta_s_rel)
        self.eta = eta
        
        # parameters for stage 3.1 (verification)
        self.tau = int(n_freq_bins * tau)
        self.theta = int(n_freq_bins * theta)

        self.history = [] # frequency indices

        if init_freq_bounds is not None:
            self.init_freq_bounds = init_freq_bounds
        else:
            self.init_freq_bounds = [50, 90]
        self.init_freq_bounds = (np.array(self.init_freq_bounds) / (60 * ppg_sampling_freq) * self.n_freq_bins).round().astype(int)

    def _get_N0_N1(self, spectrum):

        N_prev = self.history[-1]
        R_0_base_idx = N_prev - self.delta_s
        R_0_end_idx = N_prev + self.delta_s
        R_1_base_idx = 2 * (N_prev - self.delta_s - 1) + 1
        R_1_end_idx = 2 * (N_prev + self.delta_s - 1) + 1
        R_0_idx = np.arange(R_0_base_idx, R_0_end_idx + 1)
        R_1_idx = np.arange(R_1_base_idx, R_1_end_idx + 1)

        threshold = self.eta * np.max(spectrum[R_0_idx])
        peaks = scipy.signal.find_peaks(spectrum, height=threshold)[0]
        
        
        N_0 = peaks[(peaks >= R_0_base_idx) & (peaks <= R_0_end_idx)]
        N_1 = peaks[(peaks >= R_1_base_idx) & (peaks <= R_1_end_idx)]

        if len(N_0) == 0:
            N_0 = np.array([np.argmax(spectrum[R_0_idx]) + R_0_base_idx])

        return N_0, N_1

    def _get_N_hat(self, N_0, N_1):
        N_prev = self.history[-1]
        
        # Case 1
        N_hat = None
        for n_0 in N_0:
            for n_1 in N_1:
                if n_0 % n_1 == 0 or n_1 % n_0 == 0:
                    if N_hat is None or np.abs(N_hat - N_prev) > np.abs(n_0 - N_prev):
                        N_hat = n_0

        # Case 2
        if N_hat is None:
            Nf_set = np.concatenate((N_0, (N_1 - 1) / 2))
            N_hat_idx = np.argmin(np.abs(Nf_set - N_prev))
            N_hat = Nf_set[N_hat_idx]

        return N_hat

    def _verification_stage_1(self, N_hat):
        N_prev = self.history[-1]
        if N_hat - N_prev >= self.theta:
            N_cur = N_prev + self.tau
        elif N_hat - N_prev <= -self.tau:
            N_cur = N_prev - self.tau
        else:
            N_cur = N_hat

        return N_cur

    def transform_first(self, spectrum: np.ndarray):
        peaks = scipy.signal.find_peaks(spectrum)[0]
        peaks_sorted = peaks[np.argsort(spectrum[peaks])[::-1]]
        peaks_sorted = peaks_sorted[self.init_freq_bounds[0]:self.init_freq_bounds[1]]

        if len(peaks_sorted) == 0:
            N_cur = self.init_freq_bounds[0] + np.argmax(spectrum[self.init_freq_bounds[0]:self.init_freq_bounds[1]])
        else:
            N_cur = np.argmax(peaks_sorted)
        self.history.append(N_cur)
        
        return N_cur

    def transform(self, spectrum: np.ndarray):
        N_0, N_1 = self._get_N0_N1(spectrum)
        N_hat = self._get_N_hat(N_0, N_1)
        N_cur = self._verification_stage_1(N_hat).astype(int)
        self.history.append(N_cur)

        return N_cur

class Troika:
    def __init__(self, window_duration=10, acc_sampling_freq=100, cutoff_freqs=[0.4, 5], n_freq_bins=4096):
        self.window_duration = window_duration
        self.sampling_freq = acc_sampling_freq
        self.cutoff_freqs = cutoff_freqs
        self.acc_sampling_freq = acc_sampling_freq
        self.acc_window_len = window_duration * acc_sampling_freq
        self.n_freq_bins = n_freq_bins
        self.ssr = SSR(self.acc_window_len, self.n_freq_bins, self.acc_sampling_freq)


    def _get_dominant_frequencies(self, spectrum: np.ndarray, axis=-1, threshold=.5):
        """
        Given the frequency spectra of one or multiple signals, compute the dominant frequencies along a specified axis.

        Parameters
        ----------
            sig : ndarray
                The signals to compute the dominant frequencies on.
            axis : int, default=-1
                The axis along which to compute the dominant frequencies.
            threshold : float, default=0.5
                The threshold which divides dominant and non-dominant frequencies.
                A dominant frequency has a peak of amplitude of higher than threshold times the maximum amplitude in that spectrum

        Returns
        ----------
            dom_freqs : ndarray
                dom_freqs has the same shape as sig.
                Iff a value in sig corresponds to a dominant frequency, this value is set to True in dom_freqs.
        """

        max_amplitudes = np.max(spectrum, axis=axis, keepdims=True)
        dom_freqs = spectrum > threshold * max_amplitudes

        return dom_freqs

    def _temporal_difference(self, ts, k):
        """
        Perform the kth-order temporal-difference operation on the time series ts.
        The first-order temporal difference of a time series [h_1, h_2, ..., h_n] is
        another time series given by [h_2 - h_1, ..., h_n - h_(n-1)]
        and the kth order temporal difference is given by the first-order difference of the order k-1 difference.

        Parameters
        ----------
            ts : one-dimensional ndarray
                input array to compute the temporal difference on
            k : int
                the order

        Returns
        ----------
            ts_hat : ndarray of length len(ts) - k
                The computed temporal difference
        """
        diff = np.diff(ts, k)
        diff_pad = np.pad(diff, (0, k), mode='constant', constant_values=0)
        return diff_pad
    

    def _filter_ssa_groups(self, ssa_groups: list, threshold: float = 0.01):

        selected_indices = []
        for i in range(ssa_groups.shape[0]):
            _, periodogram = scipy.signal.periodogram(ssa_groups[i,:], nfft=4096 * 2 - 1)
            frequencies = np.linspace(0,100, 4096)
            max_amplitude = np.max(np.abs(periodogram))
            hr_frequenies = (frequencies > 0.5) & (frequencies < 3)

            if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
                selected_indices = np.append(selected_indices, i)

        selected_indices = np.array(selected_indices, dtype=int)
        acc_reconstructed = ssa_groups[selected_indices,:].sum(axis=0)
        return acc_reconstructed

    def transform(self, acc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            ppg : ndarray of shape (n_channels, n_timestamps)
                The PPG signal.
            acc : ndarray of shape (n_dimensions, n_timestamps)
                The accelaration data used for denoising.
        """
        n_acc_samples = acc.shape[1]

        n_acc_windows = acc.shape[0]
        n_windows = n_acc_windows
        current_window = 0
        progress_bar = tqdm(total=n_windows, initial=current_window)

        spt = SpectralPeakTracker(ppg_sampling_freq=self.sampling_freq, n_freq_bins=self.n_freq_bins, init_freq_bounds=[50, 90])
        #spt = PeakFinder()

        while current_window < n_windows:
            
            progress_bar.set_description(f"Calculating window {current_window}/{n_windows}")

            acc_window = acc[current_window,...]
            acc_x = acc_window[:,0]
            acc_y = acc_window[:,1]
            acc_z = acc_window[:,2]
            try:
                ssa_groups_x, wcorr_x = ssa(acc_x, 500, perform_grouping=True, ret_Wcorr=True)
                ssa_groups_y, wcorr_y = ssa(acc_y, 500, perform_grouping=True, ret_Wcorr=True)
                ssa_groups_z, wcorr_z = ssa(acc_z, 500, perform_grouping=True, ret_Wcorr=True)
            except np.linalg.LinAlgError:
                yield np.nan
                current_window += 1
                progress_bar.update()
                continue


            acc_reconstructed_x = self._filter_ssa_groups(ssa_groups_x)
            acc_reconstructed_y = self._filter_ssa_groups(ssa_groups_y)
            acc_reconstructed_z = self._filter_ssa_groups(ssa_groups_z)

            acc_reconstructed = np.sqrt(acc_reconstructed_x**2 + acc_reconstructed_y**2 + acc_reconstructed_z**2)
            
            bandpass_filter = butter(4, self.cutoff_freqs, btype='band', fs=self.sampling_freq, output='sos')
            acc_reconstructed = sosfiltfilt(bandpass_filter, acc_reconstructed)

            if current_window == 0:
                frequencies, sparse_acc_spectrum = self.ssr.transform(acc_reconstructed)
                prev_window_hr_idx = spt.transform_first(sparse_acc_spectrum)
                
                yield frequencies[prev_window_hr_idx]
            else:
                
                temporal_difference = self._temporal_difference(acc_reconstructed, 2)
                frequencies, sparse_acc_spectrum = self.ssr.transform(temporal_difference)
                prev_window_hr_idx = spt.transform(sparse_acc_spectrum)
                
                yield frequencies[prev_window_hr_idx]

            current_window += 1
            progress_bar.update()
            
        progress_bar.close()


"""
#%% test troika
    

from classical_utils import *
import config
import matplotlib.pyplot as plt

dataset_dir = config.data_dir_Apple_processed_100hz
split = 0
X_val, Y_val, pid_val, metrics_val, X_test, Y_test, pid_test, metrics_test = load_dataset(dataset_dir, split)

i = 10000
acc = X_test[i]
hr_true = Y_test[i]
pid = pid_test[i] 



#%%
ssa = SingularSpectrumAnalysis(window_size=500, groups=None)

acc_x = data_utils.butterworth_bandpass(acc[:,0], low=0.5, high=4, fs=100)
acc_y = data_utils.butterworth_bandpass(acc[:,1], low=0.5, high=4, fs=100)
acc_z = data_utils.butterworth_bandpass(acc[:,2], low=0.5, high=4, fs=100)
#acc_groups_x, wcorr_x = ssa_faster(acc_x, 500, perform_grouping=True, ret_Wcorr=True)
#acc_groups_y, wcorr_y = ssa_faster(acc_y, 500, perform_grouping=True, ret_Wcorr=True)
#acc_groups_z, wcorr_z = ssa_faster(acc_z, 500, perform_grouping=True, ret_Wcorr=True)
acc_groups_x = ssa.fit_transform(acc_x.reshape(1,-1))


#%%
def select_components(acc_groups, threshold=0.1):
    selected_indices = []
    for i in range(acc_groups.shape[0]):
        frequencies, periodogram = scipy.signal.periodogram(acc_groups[i,:], nfft=4096 * 2 - 1, fs=100)
        max_amplitude = np.max(np.abs(periodogram))
        hr_frequenies = (frequencies > 0.5) & (frequencies < 3)

        if np.any(periodogram[hr_frequenies] > max_amplitude*threshold):
            selected_indices = np.append(selected_indices, i)
    #print(selected_indices)
    selected_indices = np.array(selected_indices, dtype=int)
    acc_reconstructed = acc_groups[selected_indices,:].sum(axis=0)
    return acc_reconstructed

threshold = 0.01
acc_reconstructed_x = select_components(acc_groups_x, threshold)
acc_reconstructed_y = select_components(acc_groups_y, threshold)
acc_reconstructed_z = select_components(acc_groups_z, threshold)

acc_reconstructed = np.sqrt(acc_reconstructed_x**2 + acc_reconstructed_y**2 + acc_reconstructed_z**2)
acc_reconstructed = data_utils.butterworth_bandpass(acc_reconstructed, low=0.5, high=2.5, fs=100)
plt.plot(acc_reconstructed)
frequencies, periodogram = scipy.signal.periodogram(acc_reconstructed, nfft=4096 * 2 - 1, fs=100)
plt.show()
plt.plot(frequencies*60, periodogram)
plt.xlim(0,120)




#%%
# plot y log
#plt.yscale("log")
hr_frequenies_ind = (frequencies > 0.5) & (frequencies < 2)
hr_periodogram, hr_frequenies = periodogram[hr_frequenies_ind], frequencies[hr_frequenies_ind]
hr_peak_ind = scipy.signal.find_peaks(hr_periodogram)[0]
highest_hr_peak_ind = np.argsort(hr_periodogram[hr_peak_ind])[::-1][:10]
# take highest peaks in the range 0.5-2 Hz

highest_hr_peaks = hr_frequenies[hr_peak_ind[highest_hr_peak_ind]] * 60
highest_hr_peak = highest_hr_peaks[0]
closest_hr_peak = highest_hr_peaks[np.argmin(np.abs(highest_hr_peaks - hr_true))]
print(highest_hr_peak,closest_hr_peak, hr_true)


ts = acc_x
N = 1000
L = 600
N = len(ts)
K = N - L + 1

L_trajectory_matrix = hankel(ts[:L], ts[L-1:]) # (L, K)
U, Sigma, V = np.linalg.svd(L_trajectory_matrix) # (L, L); (d, ); (K, K)
V = V.T # (K, K)
d = len(Sigma)

#%%
V = V.T # (K, K)
d = len(Sigma)

deconstructed_ts = []
for i in range(d):
    X_elem = np.array(Sigma[i] * np.outer(U[:,i], V[:,i])) # (L, K)
    X_elem_rev = X_elem[::-1] # (L, K)
    ts_i = np.array([X_elem_rev.diagonal(i).mean() for i in range(-L+1, K)])
    deconstructed_ts.append(ts_i)
deconstructed_ts = np.array(deconstructed_ts) # (d, L, K)
# %%
import torch

# Assuming acc_x is already a torch tensor on CPU
ts = torch.Tensor(acc_x.copy())
N = len(ts)
L = 600
K = N - L + 1

L_trajectory_matrix = torch.tensor(hankel(ts[:L], ts[L-1:])).cuda() # (L, K)
ts = ts.cuda()
U, Sigma, V = torch.svd(L_trajectory_matrix)  # (K, K); (L, ); (L, L)
V = V.T  # (L, L)
d = Sigma.size(0)

deconstructed_ts = torch.zeros((d, L, K))
for i in range(d):
    X_elem = Sigma[i] * torch.ger(U[:, i], V[i, :])  # (K, L)
    X_elem_rev = torch.flip(X_elem, dims=[1])  # (K, L)
    ts_i = torch.tensor([torch.diagonal(X_elem_rev, offset).mean() for offset in range(-L + 1, K)])
    deconstructed_ts[i] = ts_i

deconstructed_ts = deconstructed_ts.permute(1, 0, 2)  # (L, d, K) """
# %%

""" 
troika = Troika(window_duration=10, step_duration=8, acc_sampling_freq=100, cutoff_freqs=[0.4, 5])

# %%
Y_pred, Y_true, Pids, Metrics = [], [], [], []

for hr_true, hr, pid, metric in zip(Y_test, troika.transform(X_test[100:]), pid_test, metrics_test):
    print(hr_true, hr)

    Y_pred.append(hr)
    Y_true.append(hr_true)
    Pids.append(pid)
    Metrics.append(metric)

    with open("troika_results.csv", "a") as f:
        f.write(f"{hr_true},{hr},{pid},{metric}\n")

df_results = pd.DataFrame({"y_true": Y_true, "hr_pred": Y_pred, "pid": Pids, "metric": Metrics})
df_results.to_pickle("troika_results.pkl")
 """
