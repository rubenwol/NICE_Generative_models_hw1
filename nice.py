"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer.
"""


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()

        # some functions
        linear_func = lambda x: nn.Linear(in_features=mid_dim, out_features=mid_dim)
        relu_func = lambda x: nn.ReLU()

        # Layers
        self.fc_in = nn.Linear(in_features=in_out_dim // 2, out_features=mid_dim)

        self.hidden_layers = nn.ModuleList([f(x) for x in range(hidden) for f in (linear_func, relu_func)])

        self.fc_out = nn.Linear(in_features=mid_dim, out_features=in_out_dim // 2)

        self.relu = nn.ReLU()

        self.layers = self.hidden_layers

        self.layers.insert(0, self.fc_in)
        self.layers.insert(0, self.relu)
        self.layers.append(self.fc_out)

        self.coupling = nn.Sequential(*self.layers)

        self.mask_config = mask_config

    def forward(self, x, log_det_J=0, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        x1 = x[:, 0::2] if (self.mask_config % 2) == 0 else x[:, 1::2]
        x2 = x[:, 1::2] if (self.mask_config % 2) == 0 else x[:, 0::2]

        y1 = x1
        y2 = x2 + self.coupling(x1) if not reverse else x2 - self.coupling(x1)

        x = torch.zeros_like(x)
        x[:, 0::2] = y1 if (self.mask_config % 2) == 0 else y2
        x[:, 1::2] = y2 if (self.mask_config % 2) == 0 else y1

        return x, log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # some functions
        linear_func = lambda x: nn.Linear(in_features=mid_dim, out_features=mid_dim)
        relu_func = lambda x: nn.ReLU()

        # Layers
        self.fc_in = nn.Linear(in_features=in_out_dim // 2, out_features=mid_dim)

        self.hidden_layers = nn.ModuleList([f(x) for x in range(hidden) for f in (linear_func, relu_func)])

        self.fc_out = nn.Linear(in_features=mid_dim, out_features=in_out_dim // 2)

        self.relu = nn.ReLU()

        self.layers = self.hidden_layers

        self.layers.insert(0, self.fc_in)
        self.layers.insert(0, self.relu)
        self.layers.append(self.fc_out)

        self.coupling = nn.Sequential(*self.layers)

        self.mask_config = mask_config

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        x1 = x[:, 0::2] if (self.mask_config % 2) == 0 else x[:, 1::2]
        x2 = x[:, 1::2] if (self.mask_config % 2) == 0 else x[:, 0::2]

        # TODO fill in

        # return x, log_det_J


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps

        if not reverse:
            x = scale * x
        else:
            x = (scale ** -1) * x
        log_det_J = torch.sum(self.scale) + self.eps # why self.scale and not scale: because torch.sum(self.scale) == torch.sum(torch.log(scale))
        return x, log_det_J


"""Standard logistic distribution.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logistic = TransformedDistribution(Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device)), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type, in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        if self.coupling_type == 'additive':
            self.coupling_layers = nn.ModuleList(
                [AdditiveCoupling(in_out_dim, mid_dim, hidden, i % 2) for i in range(self.coupling)])
        else:
            self.coupling_layers = nn.ModuleList([AffineCoupling(in_out_dim, mid_dim, hidden, i % 2)
                                                  for i in range(self.coupling)])

        self.scaling = Scaling(in_out_dim)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """

        if self.coupling_type == 'additive':
            z, log_det_scale = self.scaling(z, reverse=True)
        for layers in reversed(self.coupling_layers):
            z, _ = layers(z, 0, reverse=True)
        return z

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        ldj = 0
        for i, layers in enumerate(self.coupling_layers):
            x, ldj = layers(x, log_det_J=ldj)
        if self.coupling_type == 'additive':
            x, log_det_scale = self.scaling(x)
            ldj += log_det_scale
        return x, ldj

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
