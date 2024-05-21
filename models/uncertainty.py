import torch
from torch import nn
import numpy as np


class Uncertainty_Wrapper(nn.Module):
    """
    A wrapper for adding uncertainty estimation to a base model.
    
    Args:
        base_model (nn.Module): The base neural network model.
        n_classes (int): Number of classes for classification tasks.
        return_probs (bool): Whether to return the class probabilities.
        uncertainty_model (str): The type of uncertainty model to use ('std' or 'entropy').
    """
    def __init__(self, base_model, n_classes, return_probs=True, uncertainty_model="std"):
        super(Uncertainty_Wrapper, self).__init__()
        
        # Attach base model components if they exist
        if hasattr(base_model, 'classifier'):
            self.classifier = base_model.classifier
        if hasattr(base_model, 'backbone'):
            self.backbone = base_model.backbone

        self.base_model = base_model
        self.n_classes = n_classes
        self.return_probs = return_probs
        self.uncertainty_model = uncertainty_model
        
        # Create bins for the class probabilities
        self.bins = np.array([-np.inf] + list(np.linspace(0, 1, n_classes-1)) + [np.inf])

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        logits, feat = self.base_model(x)
        probs = torch.softmax(logits, dim=1)
        uncertainty = self._compute_uncertainty_from_probs(probs)
        expectation = self._compute_expectation_from_probs(probs)
        
        if self.return_probs:
            return expectation, uncertainty, probs
        
        return expectation, uncertainty

    def _compute_uncertainty_from_probs(self, probs):
        """
        Compute uncertainty from probabilities.

        Args:
            probs (torch.Tensor): Class probabilities.

        Returns:
            torch.Tensor: Computed uncertainty.
        """
        if self.uncertainty_model == 'entropy':
            return -torch.sum(probs * torch.log(probs), dim=1)
        elif self.uncertainty_model == 'std':
            bins = torch.tensor(np.clip(self.bins, -3/self.n_classes, 1 + 3/self.n_classes), device=probs.device)
            bins = torch.diff(bins) / 2 + bins[:-1]
            E_x = torch.sum(probs * bins[None, :], axis=1)
            E_x2 = torch.sum(probs * bins[None, :] ** 2, axis=1)
            return torch.sqrt(E_x2 - E_x**2)
        else:
            raise NotImplementedError(f"Uncertainty model {self.uncertainty_model} not implemented")

    def _compute_expectation_from_probs(self, probs):
        """
        Compute expectation from probabilities.

        Args:
            probs (torch.Tensor): Class probabilities.

        Returns:
            torch.Tensor: Computed expectation.
        """
        bins = torch.tensor(np.clip(self.bins, -3/self.n_classes, 1 + 3/self.n_classes), device=probs.device)
        bin_length = bins[1:] - bins[:-1]
        bins = torch.diff(bins) / 2 + bins[:-1]
        new_probs = probs * bin_length
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)
        return torch.sum(new_probs * bins, axis=1)
    
    def _compute_expectation_from_samples(self, samples):
        """
        Compute expectation from samples.

        Args:
            samples (torch.Tensor): Sampled outputs.

        Returns:
            torch.Tensor: Computed expectation.
        """
        return samples.mean(axis=1)

    def _compute_uncertainty_from_samples(self, samples):
        """
        Compute uncertainty from samples.

        Args:
            samples (torch.Tensor): Sampled outputs.

        Returns:
            torch.Tensor: Computed uncertainty.
        """
        if self.uncertainty_model == "entropy":
            raise NotImplementedError
        elif self.uncertainty_model == 'std':
            return samples.std(axis=1)
        else:
            raise NotImplementedError(f"Uncertainty model {self.uncertainty_model} not implemented")


class Uncertainty_Regression_Wrapper(Uncertainty_Wrapper):
    """
    A wrapper for adding uncertainty estimation to regression models.
    """
    def __init__(self, base_model, n_classes, return_probs=True, uncertainty_model="std"):
        super(Uncertainty_Regression_Wrapper, self).__init__(base_model=base_model, n_classes=n_classes, return_probs=return_probs, uncertainty_model=uncertainty_model)
    
    def forward(self, x):
        """
        Forward pass for regression model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        preds, feat = self.base_model(x)
        uncertainty = torch.zeros_like(preds)
        expectation = preds
        
        if self.return_probs:
            pred_indices = torch.searchsorted(torch.tensor(self.bins).to(preds.device), preds)
            probs = torch.zeros((preds.shape[0], self.bins.shape[0] - 1))
            for i in range(preds.shape[0]):
                probs[i, pred_indices[i]] = 1
            return expectation, uncertainty, probs
        
        return expectation, uncertainty


class Ensemble_Wrapper(Uncertainty_Wrapper):
    """
    A wrapper for ensemble models to add uncertainty estimation.
    """
    def __init__(self, base_model, model_paths, n_classes, return_probs=True, uncertainty_model="std"):
        super(Ensemble_Wrapper, self).__init__(base_model=base_model, n_classes=n_classes, return_probs=return_probs, uncertainty_model=uncertainty_model)
        self.model_paths = model_paths
        self.n_samples = len(model_paths)

    def forward(self, x):
        """
        Forward pass for ensemble model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        out_samples = []
        for model_path in self.model_paths:
            self.base_model.load_state_dict(torch.load(model_path)['trained_backbone'])
            self.base_model.eval()
            out_samples.append(self.base_model(x)[0])

        out_samples = torch.cat(out_samples, dim=1)
        
        # Calculate mean and uncertainty
        probs = torch.tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.bins, density=False)[0] / self.n_samples, 1, out_samples.cpu().numpy()), dtype=x.dtype)
        uncertainty = self._compute_uncertainty_from_samples(out_samples)
        expectation = self._compute_expectation_from_samples(out_samples)
        
        if self.return_probs:
            return expectation, uncertainty, probs
        
        return expectation, uncertainty


class NLE_Wrapper(Uncertainty_Wrapper):
    """
    A wrapper for models with Laplace distribution-based uncertainty estimation.
    """
    def __init__(self, base_model, n_classes, return_probs=True, uncertainty_model="std"):
        super(NLE_Wrapper, self).__init__(base_model=base_model, n_classes=n_classes, return_probs=return_probs, uncertainty_model=uncertainty_model)
        assert self.uncertainty_model == 'std', 'NLE only supports variance as uncertainty model'
        self.bins = torch.Tensor(self.bins)
    
    def forward(self, x):
        """
        Forward pass for NLE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        out, feat = self.base_model(x)

        uncertainty = out[1]
        # computing the std, which is sqrt(2) * s for Laplace distribution
        uncertainty = torch.exp(uncertainty) * np.sqrt(2) 
        expectation = out[0]
        
        if self.return_probs:
            probs = self._compute_probs(expectation, uncertainty)
            return expectation, uncertainty, probs
        
        return expectation, uncertainty
    
    def _compute_probs(self, expectation, uncertainty):
        """
        Compute class probabilities for NLE model.

        Args:
            expectation (torch.Tensor): Expected values.
            uncertainty (torch.Tensor): Uncertainty values.

        Returns:
            torch.Tensor: Computed probabilities.
        """
        device = expectation.device
        expectation = expectation.detach().cpu()
        uncertainty = uncertainty.detach().cpu()

        def laplace_cdf(x, mu, std):
            b = std / np.sqrt(2)
            cdf_values = 0.5 * (1 + np.sign(x - mu) * (1 - np.exp(-np.abs(x - mu) / b)))
            return cdf_values

        bins = np.clip(self.bins, -3/self.n_classes, 1 + 3/self.n_classes)
        cdf_values = laplace_cdf(bins, expectation, uncertainty)
        bin_probs = cdf_values[:, 1:] - cdf_values[:, :-1]
        bin_probs = bin_probs / bin_probs.sum(axis=1, keepdims=True)

        return bin_probs.to(device)


class MC_Dropout_Wrapper(Uncertainty_Wrapper):
    """
    A wrapper for Monte Carlo Dropout-based uncertainty estimation.
    """
    def __init__(self, base_model, n_classes=64, n_samples=100):
        super(MC_Dropout_Wrapper, self).__init__(base_model=base_model, n_classes=n_classes)
        self.n_samples = n_samples

    def forward(self, x):
        """
        Forward pass with Monte Carlo Dropout.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        out, feat = self.base_model(x)
        
        # Enable dropout during inference
        self.train()
        out_samples = []
        for _ in range(self.n_samples):
            out_sample, _ = self.base_model(x)
            out_samples.append(out_sample)

        # Stack the results along a new dimension
        out_samples = torch.cat(out_samples, dim=1)

        # Calculate mean and uncertainty
        probs = torch.tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.bins, density=False)[0] / self.n_samples, 1, out_samples.cpu().numpy()), dtype=x.dtype).to(x.device)
        uncertainty = self._compute_uncertainty_from_samples(out_samples)
        expectation = self._compute_expectation_from_samples(out_samples)

        if self.return_probs:
            return expectation, uncertainty, probs
        
        return expectation, uncertainty


class BNN_Wrapper(Uncertainty_Wrapper):
    """
    A wrapper for Bayesian Neural Network-based uncertainty estimation.
    """
    def __init__(self, base_model, n_classes=64, n_samples=100):
        super(BNN_Wrapper, self).__init__(base_model=base_model, n_classes=n_classes)
        self.n_samples = n_samples

    def forward(self, x):
        """
        Forward pass for Bayesian Neural Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: expectation, uncertainty, and optionally probabilities.
        """
        out_samples = []
        for _ in range(self.n_samples):
            out_sample, _ = self.base_model(x)
            out_samples.append(out_sample)

        # Stack the results along a new dimension
        out_samples = torch.cat(out_samples, dim=1)

        # Calculate mean and uncertainty
        probs = torch.tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.bins, density=False)[0] / self.n_samples, 1, out_samples.cpu().numpy()), dtype=x.dtype).to(x.device)
        uncertainty = self._compute_uncertainty_from_samples(out_samples)
        expectation = self._compute_expectation_from_samples(out_samples)

        if self.return_probs:
            return expectation, uncertainty, probs
        
        return expectation, uncertainty