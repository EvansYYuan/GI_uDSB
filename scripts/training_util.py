from typing import Dict, Any, Sequence

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from sde import SchrodingerBridgePolicy

# ---------------- Training history processing ----------------

def init_training_history() -> Dict[str, list]:
    """Initialize history dict for joint training."""
    return {
        # Forward policy
       'f_loss_total': [], 'f_loss_vae': [], 'f_loss_dsb': [],
       'f_loss_vel_consistency': [], 'f_loss_bio': [],
        # Backward policy
       'b_loss_total': [], 'b_loss_vae': [], 'b_loss_dsb': [],
       'b_loss_vel_consistency': [], 'b_loss_bio': [],
    }


def update_training_history(
    history: Dict[str, list],
    direction: str,  # 'b' or 'f'
    avg_total: float,
    avg_vae: float,
    avg_dsb: float,
    avg_vel_consistency: float,
    avg_bio: float,
) -> None:
    """Append averaged losses for one epoch into history."""
    prefix = 'b_' if direction == 'b' else 'f_'
    history[f'{prefix}loss_total'].append(avg_total)
    history[f'{prefix}loss_vae'].append(avg_vae)
    history[f'{prefix}loss_dsb'].append(avg_dsb)
    history[f'{prefix}loss_vel_consistency'].append(avg_vel_consistency)
    history[f'{prefix}loss_bio'].append(avg_bio)


def plot_joint_training_history(training_history: Dict[str, Sequence[float]], out_path: str | None = None) -> None:
    """
    Plot joint training curves for forward/backward policies.

    Expects keys:
        'f_loss_total', 'f_loss_vae', 'f_loss_dsb', 'f_loss_vel_consistency', 'f_loss_bio',
        'b_loss_total', 'b_loss_vae', 'b_loss_dsb', 'b_loss_vel_consistency', 'b_loss_bio'.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Forward Policy
    axes[0, 0].plot(training_history["f_loss_total"], label="Total", linewidth=2.5, color="black")
    axes[0, 0].plot(training_history["f_loss_vae"], label="VAE", alpha=0.7)
    axes[0, 0].plot(training_history["f_loss_dsb"], label="SB", alpha=0.7)
    axes[0, 0].plot(training_history["f_loss_bio"], label="Biology", alpha=0.7)
    axes[0, 0].set_title("Forward Policy: Loss Components", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(training_history["f_loss_dsb"], label="DSB Total", linewidth=2, color="blue")
    axes[0, 1].plot(
        training_history["f_loss_vel_consistency"],
        label="Velocity consistency (L2)",
        linewidth=2,
        color="orange",
        linestyle="--",
    )
    axes[0, 1].set_title("Forward Policy: SB Decomposition", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(training_history["f_loss_bio"], linewidth=2.5, color="green")
    axes[0, 2].set_title("Forward Policy: Biology Penalty", fontsize=12, fontweight="bold")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Backward Policy
    axes[1, 0].plot(training_history["b_loss_total"], label="Total", linewidth=2.5, color="black")
    axes[1, 0].plot(training_history["b_loss_vae"], label="VAE", alpha=0.7)
    axes[1, 0].plot(training_history["b_loss_dsb"], label="SB", alpha=0.7)
    axes[1, 0].plot(training_history["b_loss_bio"], label="Biology", alpha=0.7)
    axes[1, 0].set_title("Backward Policy: Loss Components", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(training_history["b_loss_dsb"], label="DSB Total", linewidth=2, color="blue")
    axes[1, 1].plot(
        training_history["b_loss_vel_consistency"],
        label="Velocity consistency (L2)",
        linewidth=2,
        color="orange",
        linestyle="--",
    )
    axes[1, 1].set_title("Backward Policy: SB Decomposition", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(training_history["b_loss_bio"], linewidth=2.5, color="green")
    axes[1, 2].set_title("Backward Policy: Biology Penalty", fontsize=12, fontweight="bold")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_training_summary(training_history: Dict[str, Sequence[float]]) -> None:
    """
    Print final values of each loss component (backward & forward).
    """
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY - All Components")
    print("=" * 70)

    print("\nForward Policy:")
    print(f"  Final total loss:           {training_history['f_loss_total'][-1]:8.4f}")
    print(f"  Final VAE loss:             {training_history['f_loss_vae'][-1]:8.4f}")
    print(f"  Final SB loss:              {training_history['f_loss_dsb'][-1]:8.4f}")
    print(f"  Final Vel-consistency loss: {training_history['f_loss_vel_consistency'][-1]:8.4f}")
    print(f"  Final Biology loss:         {training_history['f_loss_bio'][-1]:8.4f}")

    print("\nBackward Policy:")
    print(f"  Final total loss:           {training_history['b_loss_total'][-1]:8.4f}")
    print(f"  Final VAE loss:             {training_history['b_loss_vae'][-1]:8.4f}")
    print(f"  Final SB loss:              {training_history['b_loss_dsb'][-1]:8.4f}")
    print(f"  Final Vel-consistency loss: {training_history['b_loss_vel_consistency'][-1]:8.4f}")
    print(f"  Final Biology loss:         {training_history['b_loss_bio'][-1]:8.4f}")

    print("\n" + "=" * 70)


# ---------------- Model saving/loading ----------------

def save_trained_models(
    model_dir: str,
    config: Dict[str, Any],
    dyn: Any,
    vae: Any,
    z_f: Any,
    z_b: Any,
    training_history: Dict[str, Sequence[float]],
    input_dim: int,
    latent_dim: int,
) -> None:
    """
    Save trained VAE, policies, configs, and training history.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save forward policy
    forward_policy_path = os.path.join(model_dir, "z_f_policy.pth")
    torch.save(
        {"model_state_dict": z_f.net.state_dict(), "config": config},
        forward_policy_path,
    )

    # Save backward policy
    backward_policy_path = os.path.join(model_dir, "z_b_policy.pth")
    torch.save(
        {"model_state_dict": z_b.net.state_dict(), "config": config},
        backward_policy_path,
    )

    # Save VAE
    vae_path = os.path.join(model_dir, "vae_joint_trained.pth")
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        },
        vae_path,
    )

    # Save training history
    history_path = os.path.join(model_dir, "training_history.json")
    history_json = {k: [float(v) for v in vals] for k, vals in training_history.items()}
    with open(history_path, "w") as f:
        json.dump(history_json, f, indent=2)

    # Save DYN (SDE) configuration
    dyn_config_path = os.path.join(model_dir, "dyn_config.json")
    dyn_config = {
        "sde_type": config["sde_type"],
        "sigma_min": config["sigma_min"],
        "sigma_max": config["sigma_max"],
        "T": config["T"],
        "interval": config["interval"],
        "t0": config["t0"],
        "dt": float(dyn.dt),
    }
    with open(dyn_config_path, "w") as f:
        json.dump(dyn_config, f, indent=2)

    # Save full config
    config_path = os.path.join(model_dir, "full_config.json")
    config_json = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in config.items()}
    config_json["data_dim"] = config_json["data_dim"]
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)


def load_trained_models(
    model_dir: str,
    device: torch.device,
    VAEClass: Any,
    policy_mlp_cls: Any,
    sde_cls: Any,
) -> Dict[str, Any]:
    """
    Load trained models and reconstruct VAE, SDE, policies, and training history.

    VAEClass: class for VAE (e.g. VAE_scRNA)
    policy_mlp_cls: class for policy MLP (e.g. MLP from sde.py)
    sde_cls: SDE dynamics class (e.g. VESDE)
    """
    # Load config
    with open(os.path.join(model_dir, "full_config.json")) as f:
        config = json.load(f)

    # Load VAE
    vae_ckpt = torch.load(os.path.join(model_dir, "vae_joint_trained.pth"), map_location=device)
    vae = VAEClass(input_dim=vae_ckpt["input_dim"], latent_dim=vae_ckpt["latent_dim"]).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    vae_decoder = lambda z: vae.decoder_output(vae.decoder(z))

    # Dummy samplers for SDE (only needed to reconstruct dyn)
    class DataSampler:
        def __init__(self, data, device=device):
            self.data = data
            self.device = device

        def sample(self, batch_size: int):
            indices = np.random.choice(len(self.data), size=batch_size, replace=True)
            return torch.tensor(self.data[indices], dtype=torch.float32).to(self.device)

    dummy_data = np.random.randn(100, config["data_dim"][0])
    p_sampler = DataSampler(dummy_data, device)
    q_sampler = DataSampler(dummy_data, device)

    # Reconstruct SDE dynamics and time grid
    dyn = sde_cls(config, p_sampler, q_sampler)
    ts = torch.linspace(config["t0"], config["T"], config["interval"]).to(device)

    # Load policies
    f_ckpt = torch.load(os.path.join(model_dir, "z_f_policy.pth"), map_location=device)
    net_f = policy_mlp_cls(input_dim=config["data_dim"][0], output_dim=config["data_dim"][0]).to(device)
    net_f.load_state_dict(f_ckpt["model_state_dict"])
    z_f = SchrodingerBridgePolicy(config, "forward", dyn, net_f).to(device)
    z_f.eval()

    b_ckpt = torch.load(os.path.join(model_dir, "z_b_policy.pth"), map_location=device)
    net_b = policy_mlp_cls(input_dim=config["data_dim"][0], output_dim=config["data_dim"][0]).to(device)
    net_b.load_state_dict(b_ckpt["model_state_dict"])
    z_b = SchrodingerBridgePolicy(config, "backward", dyn, net_b).to(device)
    z_b.eval()

    # Load training history
    with open(os.path.join(model_dir, "training_history.json")) as f:
        training_history = json.load(f)

    return {
        "config": config,
        "vae": vae,
        "vae_decoder": vae_decoder,
        "dyn": dyn,
        "ts": ts,
        "z_f": z_f,
        "z_b": z_b,
        "p_sampler": p_sampler,
        "q_sampler": q_sampler,
        "training_history": training_history,
        "device": device,
    }

