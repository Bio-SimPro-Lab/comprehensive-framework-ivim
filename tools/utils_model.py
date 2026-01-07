import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto

#------------------------------------------------------------------------------
# Network
#------------------------------------------------------------------------------

def IVIM(Dt, Fp, Dp, b_values):

    X_normalized = (Fp.unsqueeze(1) * torch.exp(-b_values * Dp.unsqueeze(1))
                    + (1 - Fp.unsqueeze(1)) * torch.exp(-b_values * Dt.unsqueeze(1)))

    return X_normalized

class InferenceMode(Enum):
    MEAN = auto()
    MODE = auto()
    SAMPLE = auto()
    UNCERTAINTY = auto()
    FULL = auto()

class MLP(nn.Module):
    def __init__(self, b_values_no0, num_neurons, mix_components, mode):
        super(MLP, self).__init__()
        self.mode = mode
        self.b_val = b_values_no0
        self.num_neurons = num_neurons
        self.hidden = nn.Sequential(
            nn.Linear(len(self.b_val),self.num_neurons),
            nn.BatchNorm1d(self.num_neurons),
            nn.ELU(),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.BatchNorm1d(self.num_neurons),
            nn.ELU(),
        )
        if self.mode == "supervised":
            self.output = nn.Linear(self.num_neurons, 3)
        else:
            self.output = nn.Linear(self.num_neurons, 4)

    def forward(self, X):
        hidden = self.hidden(X)
        params = self.output(hidden) # Dp, Dt, Fp
        if self.mode == "supervised":
            Dp = torch.sigmoid(params[:, 0])  # Scale Dp to [0, 0.001]
            Dt = torch.sigmoid(params[:, 1])   # Scale Dt to [0, 0.01] (example range)
            Fp = torch.sigmoid(params[:, 2])
            return Dp, Dt, Fp
        else:
            Dp = torch.sigmoid(params[:, 0])
            Dt = torch.sigmoid(params[:, 1])
            Fp = torch.sigmoid(params[:, 2])
            # if normalize
            S0 = torch.sigmoid(params[:, 3])
            b_values_tensor = self.b_values_no0.view(1, -1)  # Reshape b-values
            b_values_tensor = b_values_tensor.expand(X.size(0), -1)  # Broadcast
            reconstructed = S0.unsqueeze(1) * (
                    Fp.unsqueeze(1) * torch.exp(-b_values_tensor * Dp.unsqueeze(1)) +
                    (1 - Fp.unsqueeze(1)) * torch.exp(-b_values_tensor * Dt.unsqueeze(1))
            )
            return reconstructed, Dp, Dt, Fp

# MDN for probabilistic regression
class IVIMMDN(nn.Module):
    def __init__(self, b_val, num_neurons, mix_components, mode):
        super().__init__()
        self.b_val = b_val
        self.num_neurons = num_neurons
        self.n_components = mix_components
        self.shared = nn.Sequential(
            nn.Linear(len(self.b_val), self.num_neurons),
            nn.BatchNorm1d(self.num_neurons),
            nn.ELU(),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.BatchNorm1d(self.num_neurons),
            nn.ELU(),
        )
        self.dt_mdn = MDNHead(self.num_neurons, self.n_components)
        self.fp_mdn = MDNHead(self.num_neurons, self.n_components)
        self.dp_mdn = MDNHead(self.num_neurons, self.n_components)

    def forward(self, x):
        base = self.shared(x)
        return {
            'dt': self.dt_mdn(base),
            'fp': self.fp_mdn(base),
            'dp': self.dp_mdn(base)
        }

    def inference(self, x, mode=InferenceMode.MEAN, n_samples=1):
        output = self.forward(x)

        dt = self.dt_mdn.inference(*output['dt'], mode=mode, n_samples=n_samples)
        fp = self.fp_mdn.inference(*output['fp'], mode=mode, n_samples=n_samples)
        dp = self.dp_mdn.inference(*output['dp'], mode=mode, n_samples=n_samples)

        return dt, fp, dp

class MDNHead(nn.Module):
    def __init__(self, in_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.pi = nn.Linear(in_dim, n_components)
        self.mu = nn.Linear(in_dim, n_components)
        self.sigma = nn.Linear(in_dim, n_components)

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=1)                   # [B, K]
        mu = torch.sigmoid(self.mu(x))                                     # [B, K]
        sigma = torch.exp(self.sigma(x)) + 1e-6             # [B, K]
        return pi, mu, sigma

    def inference(self, pi, mu, sigma, mode=InferenceMode.MEAN, n_samples=1):
        batch_size = pi.size(0)

        if mode is InferenceMode.MEAN:
            pred = torch.sum(pi * mu, dim=1)
            return pred

        elif mode is InferenceMode.MODE:
            comp = torch.argmax(pi, dim=1)
            pred = mu.gather(1, comp.unsqueeze(1)).squeeze(1)
            return pred

        elif mode is InferenceMode.SAMPLE:
            cum_pi = torch.cumsum(pi, dim=1)
            samples = []
            # Ensure the last value in each row is exactly 1.0
            cum_pi[:, -1] = 1.0
            for _ in range(n_samples):
                rvs = torch.rand(batch_size, 1, device=pi.device) #[B,1]
                comp = torch.searchsorted(cum_pi, rvs).squeeze(1)
                sampled_mu = mu.gather(1, comp.unsqueeze(1)).squeeze(1)
                sampled_sigma = sigma.gather(1, comp.unsqueeze(1)).squeeze(1)
                sampled_value = sampled_mu + sampled_sigma * torch.randn_like(sampled_mu)
                samples.append(sampled_value.unsqueeze(1))
            samples = torch.cat(samples, dim=1)  # (batch_size, n_samples)
            return samples

        elif mode is InferenceMode.UNCERTAINTY:
            mean = torch.sum(pi * mu, dim=1)
            mean_sq = torch.sum(pi * (mu ** 2 + sigma ** 2), dim=1)
            var = mean_sq - mean ** 2
            std = torch.sqrt(var + 1e-8)
            return mean, std

        elif mode is InferenceMode.FULL:
            return pi, mu, sigma

        else:
            raise ValueError(f"Inference mode {mode} not supported.")
