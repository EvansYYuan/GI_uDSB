from typing import Dict, Any, Tuple, Optional, Iterable

import torch

from sde import BaseSDE, SchrodingerBridgePolicy, compute_dsb_loss_with_velocity_consistency
from vae import compute_vae_loss
from bio_con import (
    compute_biological_loss_backward,
    compute_biological_loss_forward,
)
from training_util import (
    init_training_history,
    update_training_history,
)


def compute_loss_with_all_components(
    config: Dict[str, Any],
    dyn: BaseSDE,
    ts: torch.Tensor,
    xs: torch.Tensor,
    x_next: torch.Tensor,
    zs_impt: torch.Tensor,
    policy_opt: SchrodingerBridgePolicy,
    batch_x_gene: torch.Tensor,
    batch_x_next_gene: torch.Tensor,
    batch_x_original: torch.Tensor,
    batch_x_next_original: torch.Tensor,
    batch_mu: torch.Tensor,
    batch_log_var: torch.Tensor,
    vae_decoder,  # Callable[z -> x_gene]; left untyped to avoid circular imports
    grn_data: Optional[Dict[str, Any]],
    death_gene_indices: Optional[Iterable[int]] = None,
    birth_gene_indices: Optional[Iterable[int]] = None,
    train_direction: str | None = None,  # 'f' (t->t+1, death) or 'b' (t+1->t, birth)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Total joint objective:
        L_total = λ_vae * L_vae + λ_sb * L_sb + λ_bio * L_bio
    """
    # 1) VAE loss (reconstruction + KL)
    loss_vae = compute_vae_loss(
        batch_x_original,
        batch_x_gene,
        batch_mu,
        batch_log_var,
        beta=1.0,
    )

    # 2) DSB loss (matching + velocity consistency)
    loss_dsb, loss_matching, loss_vel_consistency = (
        compute_dsb_loss_with_velocity_consistency(
            dyn, ts, xs, zs_impt, policy_opt, batch_x_original.shape[0]
        )
    )

    # 3) Biological loss (GRN + fate genes)
    #   - From t -> t+1 (train_direction == 'f'): apply GRN and death
    #   - From t+1 -> t (train_direction == 'b'): apply GRN and birth
    if train_direction == "f":
        # Forward in time: GRN(x_{t+1}) + death(x_t)
        loss_bio = compute_biological_loss_backward(
            config,
            batch_x_gene,
            batch_x_next_gene,
            grn_data,
            death_gene_indices or [],
        )
    elif train_direction == "b":
        # Backward in time: GRN(x_t) + birth(x_{t+1}) implemented by swapping
        loss_bio = compute_biological_loss_forward(
            config,
            batch_x_next_gene,
            batch_x_gene,
            grn_data,
            birth_gene_indices or [],
        )

    lambda_vae = float(config.get("lambda_vae", 1.0))
    lambda_sb = float(config.get("lambda_sb", 1.0))
    lambda_bio = float(config.get("lambda_bio", 0.5))

    total_loss = (
        lambda_vae * loss_vae
        + lambda_sb * loss_dsb
        + lambda_bio * loss_bio
    )

    loss_components = {
        "total": total_loss,
        "vae": loss_vae,
        "dsb": loss_dsb,
        "dsb_matching": loss_matching,
        "dsb_vel_consistency": loss_vel_consistency,
        "bio": loss_bio,
    }
    return total_loss, loss_components


def train_one_direction(
    *,
    direction: str,  # 'b' or 'f'
    config: Dict[str, Any],
    dyn: BaseSDE,
    ts: torch.Tensor,
    vae: torch.nn.Module,
    vae_decoder,
    z_policy: SchrodingerBridgePolicy,
    other_policy: SchrodingerBridgePolicy,
    optimizer_policy: torch.optim.Optimizer,
    optimizer_vae: torch.optim.Optimizer,
    expression_data: torch.Tensor,
    grn_data: Optional[Dict[str, Any]],
    death_gene_indices=None,
    birth_gene_indices=None,
    history: Dict[str, list],
) -> None:
    """
    One full backward or forward block (all epochs) for a single stage.
    Shared by main joint training and ablation.
    """
    device = expression_data.device

    # Extra kwargs for biology loss
    extra_kwargs = {}
    if direction == 'f' and death_gene_indices is not None:
        extra_kwargs['death_gene_indices'] = death_gene_indices
    if direction == 'b' and birth_gene_indices is not None:
        extra_kwargs['birth_gene_indices'] = birth_gene_indices

    # Sample trajectories with the *other* policy, train current policy
    if direction == 'f':
        print(f"\n  [Forward Policy Training]")
    else:
        print(f"\n  [Backward Policy Training]")

    z_policy.train()
    other_policy.eval()
    vae.train()

    with torch.no_grad():
        train_xs, train_zs = dyn.sample_traj(ts, other_policy)

    for epoch in range(config['num_epochs']):
        epoch_loss_total = 0.0
        epoch_loss_vae = 0.0
        epoch_loss_dsb = 0.0
        epoch_loss_vel = 0.0
        epoch_loss_bio = 0.0

        for _ in range(config['num_itr']):
            samp_x_idx = torch.randint(config['samp_bs'], (config['train_bs_x'],), device=device)
            samp_t_idx = torch.randint(config['interval'] - 1, (config['train_bs_t'],), device=device)

            batch_ts = ts[samp_t_idx].detach()

            batch_zs_unflat = train_xs[samp_x_idx][:, samp_t_idx, ...]
            batch_zs_impt_unflat = train_zs[samp_x_idx][:, samp_t_idx, ...]
            batch_ts_repeated = batch_ts.repeat(config['train_bs_x'])

            with torch.no_grad():
                h_k = vae.encoder(expression_data)
                mu_k, log_var_k = vae.fc_mu(h_k), vae.fc_log_var(h_k)
                rand_idx = torch.randint(0, expression_data.shape[0], (config['train_bs_x'],), device=device)
                batch_x_original = expression_data[rand_idx]
                batch_mu = mu_k[rand_idx]
                batch_log_var = log_var_k[rand_idx]

            batch_zs = batch_zs_unflat.reshape(-1, *config['data_dim']).detach().clone().requires_grad_(True)
            batch_zs_impt = batch_zs_impt_unflat.reshape(-1, *config['data_dim'])
            batch_z_next = train_xs[samp_x_idx][:, samp_t_idx + 1, ...].reshape(
                -1, *config['data_dim']
            ).detach().clone().requires_grad_(True)

            batch_x_gene = vae_decoder(batch_zs)
            batch_x_next_gene = vae_decoder(batch_z_next)

            batch_x_gene_anchor = vae_decoder(batch_mu)
            batch_x_next_original = batch_x_original

            optimizer_policy.zero_grad()
            optimizer_vae.zero_grad()

            total_loss, loss_comps = compute_loss_with_all_components(
                config, dyn, batch_ts_repeated, batch_zs, batch_z_next, batch_zs_impt, z_policy,
                batch_x_gene_anchor, batch_x_next_gene, batch_x_original, batch_x_next_original,
                batch_mu, batch_log_var,
                vae_decoder, grn_data,
                **extra_kwargs,
                train_direction=direction,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(z_policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_policy.step()
            optimizer_vae.step()

            epoch_loss_total += loss_comps['total'].item()
            epoch_loss_vae += loss_comps['vae'].item()
            epoch_loss_dsb += loss_comps['dsb'].item()
            epoch_loss_vel += loss_comps['dsb_vel_consistency'].item()
            epoch_loss_bio += loss_comps['bio'].item()

        avg_total = epoch_loss_total / config['num_itr']
        avg_vae = epoch_loss_vae / config['num_itr']
        avg_dsb = epoch_loss_dsb / config['num_itr']
        avg_vel = epoch_loss_vel / config['num_itr']
        avg_bio = epoch_loss_bio / config['num_itr']

        print(
            f"    Epoch {epoch+1:2d}/{config['num_epochs']}: "
            f"total={avg_total:.4f}, vae={avg_vae:.4f}, dsb={avg_dsb:.4f}, bio={avg_bio:.4f}"
        )

        update_training_history(
            history=history,
            direction=direction,
            avg_total=avg_total,
            avg_vae=avg_vae,
            avg_dsb=avg_dsb,
            avg_vel_consistency=avg_vel,
            avg_bio=avg_bio,
        )


def run_joint_training_loop(
    *,
    config,
    dyn,
    ts,
    vae,
    vae_decoder,
    z_f,
    z_b,
    optimizer_f,
    optimizer_b,
    optimizer_vae,
    expression_data,
    grn_data,
    death_gene_indices,
    birth_gene_indices,
    device,
) -> Dict[str, list]:
    """
    Shared multi-stage training loop for:
      - main model (with biology constraints)
      - ablation (config with λ_bio, λ_* = 0)
    Returns: training_history dict.
    """
    training_history = init_training_history()
    expression_data_dev = expression_data.to(device)

    for stage in range(config["num_stages"]):
        print(f"\n{'='*70}")
        print(f"STAGE {stage+1}/{config['num_stages']}")
        print(f"{'='*70}")

        # Forward
        train_one_direction(
            direction="f",
            config=config,
            dyn=dyn,
            ts=ts,
            vae=vae,
            vae_decoder=vae_decoder,
            z_policy=z_f,
            other_policy=z_b,
            optimizer_policy=optimizer_f,
            optimizer_vae=optimizer_vae,
            expression_data=expression_data_dev,
            grn_data=grn_data,
            death_gene_indices=death_gene_indices,
            birth_gene_indices=None,
            history=training_history,
        )
        
        # Backward
        train_one_direction(
            direction="b",
            config=config,
            dyn=dyn,
            ts=ts,
            vae=vae,
            vae_decoder=vae_decoder,
            z_policy=z_b,
            other_policy=z_f,
            optimizer_policy=optimizer_b,
            optimizer_vae=optimizer_vae,
            expression_data=expression_data_dev,
            grn_data=grn_data,
            death_gene_indices=None,
            birth_gene_indices=birth_gene_indices,
            history=training_history,
        )

    return training_history