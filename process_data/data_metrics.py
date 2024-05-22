
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest



def metric_angle_changes(signal, mag, fs, avg_window_size=5, window_size=10, step_size=1):
    """
    Detects angle changes in a signal and creates a mask based on thresholds.
    Took inspiration from van Hess et al. Estimating sleep parameters using an accelerometer without sleep diary

    Args:
    - signal (pd.DataFrame or array-like): Input signal as a pandas DataFrame or array-like structure.
    - fs (float): Sampling frequency of the signal.
    - avg_window_size (int, optional): Size of the averaging window in seconds. Default is 5.
    - angle_threshold (int, optional): Threshold value for angle change detection. Default is 1.

    Returns:
    - np.ndarray: Mask representing regions with small angle changes
    """

    # make sure signal is a pandas dataframe
    if not isinstance(signal, pd.DataFrame):
        signal = pd.DataFrame(signal, columns=['acc_x', 'acc_y', 'acc_z'])
    else:
        signal = signal[["acc_x", "acc_y", "acc_z"]].copy()

    def compute_angles(x, y, z):
        # Compute angles from accelerometer readings
        z_angle = np.arctan2(np.sqrt(x**2 + y**2), z) * 180 / np.pi
        y_angle = np.arctan2(np.sqrt(x**2 + z**2), y) * 180 / np.pi
        x_angle = np.arctan2(np.sqrt(y**2 + z**2), x) * 180 / np.pi
        return pd.Series([x_angle, y_angle, z_angle], index=['x_angle', 'y_angle', 'z_angle'])

    # Compute angles using rolling averaging and a step size of 1 second
    angles = signal.rolling(fs * avg_window_size, step=int(fs * step_size), center=True, min_periods=1).mean().apply(
        lambda x: compute_angles(x['acc_x'], x['acc_y'], x['acc_z']), axis=1)
    
    # Computes the difference between consecutive angles and takes the absolute value
    # Then computes the rolling average of the absolute differences with a window size of 10 seconds
    # Finally, takes the maximum value of the rolling average for each axis (x,y,z)
    metric = angles.diff().abs().rolling(window_size, step=1, center=True, min_periods=1).mean().max(axis=1)
    metric = metric.reindex_like(signal).interpolate(method='nearest').to_numpy()
    
    return metric

def metric_max(signal, mag, fs, window_size=10, step_size=1):
    """
    Computes the maximum value in a signal using a rolling window.

    Args:
    - signal (array-like): Input signal data.
    - fs (float): Sampling frequency of the signal.
    - window_size (int, optional): Size of the rolling window in seconds. Default is 10.

    Returns:
    - np.ndarray: Maximum value in the signal using a rolling window.
    """

    # Compute the maximum value in the signal using a rolling window
    mag = pd.Series(mag)
    metric = mag.abs().rolling(window=int(window_size * fs), center=True, min_periods=1, step=step_size*fs).max()
    metric = metric.reindex_like(mag).interpolate(method='nearest').to_numpy()

    return metric

def metric_std(signal, mag ,fs, window_size=10, step_size=1):
    """
    Computes the standard deviation of a signal using a rolling window.

    Args:
    - signal (array-like): Input signal data.
    - fs (float): Sampling frequency of the signal.
    - window_size (int, optional): Size of the rolling window in seconds. Default is 10.

    Returns:
    - np.ndarray: Standard deviation of the signal using a rolling window.
    """

    # Compute the standard deviation of the signal using a rolling window
    mag = pd.Series(mag)
    metric = mag.rolling(window=int(window_size * fs), center=True, min_periods=1, step=step_size*fs).std()
    metric = metric.reindex_like(mag).interpolate(method='nearest').to_numpy()
    return metric

def metric_mean(signal, mag, fs, window_size=10, step_size=1):
    """
    Computes the mean of a signal using a rolling window.

    Args:
    - signal (array-like): Input signal data.
    - fs (float): Sampling frequency of the signal.
    - window_size (int, optional): Size of the rolling window in seconds. Default is 10.

    Returns:
    - np.ndarray: Mean of the signal using a rolling window.
    """

    # Compute the mean of the signal using a rolling window
    mag = pd.Series(mag)
    metric = mag.rolling(window=int(window_size * fs), center=True, min_periods=1, step=step_size*fs).mean()
    metric = metric.reindex_like(mag).interpolate(method='nearest').to_numpy()
    return metric

def metric_mean_amplitude_deviation(signal ,mag, fs, window_size=10, step_size=1):
    """
    Computes the mean amplitude deviation of a signal using a rolling window.

    Args:
    - signal (array-like): Input signal data.
    - fs (float): Sampling frequency of the signal.
    - window_size (int, optional): Size of the rolling window in seconds. Default is 10.

    Returns:
    - np.ndarray: Mean amplitude deviation of the signal using a rolling window.
    """

    # Compute the mean amplitude deviation of the signal using a rolling window
    mag = pd.Series(mag)
    metric = mag.rolling(window=int(window_size * fs), center=True, min_periods=1, step=step_size*fs).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))))
    metric = metric.reindex_like(mag).interpolate(method='nearest').to_numpy()

    return metric

#%%

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create a simple 3D acceleration signal for testing
        np.random.seed(0)
        self.signal = np.random.randn(1000, 3)
        self.mag = np.linalg.norm(self.signal, axis=1)
        self.fs = 100  # 100 Hz sampling frequency

    def test_metric_angle_changes(self):
        result = metric_angle_changes(self.signal, self.mag, self.fs)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], self.signal.shape[0])

    def test_metric_max(self):
        result = metric_max(self.signal, self.mag, self.fs)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], self.signal.shape[0])

    def test_metric_std(self):
        result = metric_std(self.signal, self.mag, self.fs)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], self.signal.shape[0])

    def test_metric_mean(self):
        result = metric_mean(self.signal, self.mag, self.fs)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], self.signal.shape[0])

    def test_metric_mean_amplitude_deviation(self):
        result = metric_mean_amplitude_deviation(self.signal, self.mag, self.fs)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], self.signal.shape[0])
# %%
if __name__ == '__main__':
    unittest.main()