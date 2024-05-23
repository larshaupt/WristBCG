import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)



class NLELoss(nn.Module):
    def __init__(self, ratio=1.0):
        super(NLELoss, self).__init__()
        self.ratio = ratio

    def forward(self, predicted_values, true_labels):
        #log_uncertainties = predicted_values[..., 1]
        #predicted_values = predicted_values[..., 0]
        predicted_values, log_uncertainties = predicted_values
        abs_diff = torch.abs(predicted_values - true_labels)
        return torch.mean(abs_diff) * self.ratio + torch.mean(abs_diff * torch.exp(-log_uncertainties) + log_uncertainties) * (1 - self.ratio)
        #return torch.mean(torch.abs(predicted_values - true_labels) / log_uncertainties + torch.log(2*log_uncertainties))


def ece_loss(y_true, y_pred, bins=10):

    """
    Calculate the Expected Calibration Error (ECE) between predicted probabilities and true labels.

    ECE measures the discrepancy between predicted probabilities (y_pred) and empirical accuracy (y_true).

    Parameters:
    y_true (array-like): True labels. Each row corresponds to a sample, and each column corresponds to a class.
    y_pred (array-like): Predicted probabilities. Each row corresponds to a sample, and each column corresponds to a class.
    bins (int, optional): Number of equally spaced bins for dividing the range of predicted probabilities. Default is 10.

    Returns:
    float: The Expected Calibration Error (ECE) value.

    Notes:
    - The best possible value of ECE is 0, indicating perfect calibration.
    - The worst possible value of ECE is 1, indicating complete miscalibration.
    """
        
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_pred, axis=1)
    
    #accuracies = y_true[np.arange(len(y_true)), np.argmax(y_pred, axis=1)]
    accuracies = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece
