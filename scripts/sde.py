from typing import Tuple, Dict, Any

import abc
import numpy as np
import torch
import torch.nn as nn


# ---------------- Policy network ----------------

class MLP(nn.Module):
    """
    Policy MLP: maps (latent state, time) -> velocity field z(x_t, t).

    Input:
        x: [batch, latent_dim]
        t: [batch] or scalar
    Output:
        z: [batch, latent_dim]
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        t = t.view(-1, 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


class SchrodingerBridgePolicy(nn.Module):
    """
    Wrapper around an MLP that also holds direction and dynamics reference.
    """
    def __init__(self, config: Dict[str, Any], direction: str, dyn: "BaseSDE", net: nn.Module):
        super().__init__()
        assert direction in ("forward", "backward")
        self.config = config
        self.direction = direction
        self.dyn = dyn
        self.net = net

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure t broadcastable
        t = t.squeeze()
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.net(x, t)


# ---------------- SDE base + VE-SDE ----------------

class BaseSDE(metaclass=abc.ABCMeta):
    """
    Base SDE:
        dX_t = f(X_t, t) dt + g(t) dW_t
        where:
            - f(X_t, t) = drift (direction to move)
            - g(t)      = diffusion coefficient (noise scale)
            - dW_t      = Brownian motion (random noise)
            - dt        = infinitesimal time step
    """

    def __init__(self, config: Dict[str, Any], p, q):
        self.config = config
        self.dt = config["T"] / config["interval"]
        self.p, self.q = p, q  # samplers with .sample(batch_size)

    @abc.abstractmethod
    def _f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f(self, x: torch.Tensor, t: torch.Tensor, direction: str) -> torch.Tensor:
        return (1.0 if direction == "forward" else -1.0) * self._f(x, t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self._g(t)

    def dw(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x) * np.sqrt(self.dt)

    def propagate(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        direction: str,
        dw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x + (self.f(x, t, direction) + self.g(t) * z) * self.dt + self.g(t) * (
            self.dw(x) if dw is None else dw
        )

    def sample_traj(self, ts: torch.Tensor, policy: SchrodingerBridgePolicy) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out trajectories under a given policy.

        Returns:
            xs: [batch, len(ts), latent_dim]
            zs: [batch, len(ts), latent_dim]
        """
        direction = policy.direction
        init_dist = self.p if direction == "forward" else self.q
        ts_full = ts if direction == "forward" else torch.flip(ts, dims=[0])

        x = init_dist.sample(self.config["samp_bs"])
        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:]), device=x.device)
        zs = torch.empty_like(xs)

        for idx, t in enumerate(ts_full):
            z = policy(x, t)
            t_idx = idx if direction == "forward" else len(ts) - idx - 1
            xs[:, t_idx, ...], zs[:, t_idx, ...] = x, z
            x = self.propagate(t, x, z, direction)

        return xs, zs


class VESDE(BaseSDE):
    """
    Variance Exploding SDE for
        dX_t = f(X_t, t) dt + g(t) dW_t

    - drift f(x, t) = 0
    - diffusion g(t) increasing with t  (noise INCREASES over time)
    - Good for: starting from a simple prior and gradually revealing complexity
    """

    def __init__(self, config: Dict[str, Any], p, q):
        super().__init__(config, p, q)
        self.s_min, self.s_max = config["sigma_min"], config["sigma_max"]

    def _f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Zero drift
        return torch.zeros_like(x)

    def _g(self, t: torch.Tensor) -> torch.Tensor:
        # sigma(t) = sigma_min * (sigma_max / sigma_min)^t
        sigmas = self.s_min * (self.s_max / self.s_min) ** t
        return sigmas * np.sqrt(2 * np.log(self.s_max / self.s_min))


# ---------------- DSB loss helpers ----------------

def compute_div_gz(
    dyn: BaseSDE,
    ts: torch.Tensor,
    xs: torch.Tensor,
    policy: SchrodingerBridgePolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hutchinson trace estimator for div(g(t) * z_policy(x, t)).
    """
    zs = policy(xs, ts)
    g_ts = dyn.g(ts).view(-1, 1)
    gzs = g_ts * zs

    e = torch.randn_like(xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = (e_dzdx * e).sum(dim=-1)
    return div_gz, zs


def compute_dsb_loss_with_velocity_consistency(
    dyn: BaseSDE,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs_impt: torch.Tensor,
    policy: SchrodingerBridgePolicy,
    batch_size: int,
):
    """
    SB-style control loss + L2 velocity consistency.
    """
    div_gz, zs_policy = compute_div_gz(dyn, ts, xs, policy)

    # SB-style matching: energy + coupling + divergence
    loss_matching_per_sample = (zs_policy * (0.5 * zs_policy + zs_impt)).sum(dim=-1) + div_gz
    loss_matching = torch.mean(loss_matching_per_sample) * dyn.dt

    # Velocity consistency term: ||z_policy - z_traj||^2
    velocity_distance = torch.sum((zs_policy - zs_impt) ** 2, dim=-1)
    loss_vel_consistency  = torch.mean(velocity_distance)

    loss_dsb = loss_matching + loss_vel_consistency 
    return loss_dsb, loss_matching, loss_vel_consistency 