from typing import Dict, Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------- Core biological penalties ----------------


def compute_cell_fate_score(
    x_gene_space: torch.Tensor,
    gene_indices: Iterable[int],
) -> torch.Tensor:
    """
    Cell-fate score in gene space: penalize HIGH expression of a given gene set.

    Args:
        x_gene_space:
            Decoded gene expression in HVG space, shape [batch, n_genes].
        gene_indices:
            Indices of the genes to score (e.g. death / birth genes) in HVG space.

    Returns:
        score:
            1D tensor of shape [batch], non-negative.
            Each entry is ReLU(mean expression over gene_indices) for that cell.

    Behavior:
        - If gene_indices is empty, returns scalar 0.0 (on x_gene_space.device).
        - Uses ReLU to ensure the penalty is >= 0:
              ReLU(x) = max(0, x)
          so low expression (negative or small) yields zero penalty.
    """
    gene_indices = list(gene_indices)
    if len(gene_indices) == 0:
        return torch.tensor(0.0, device=x_gene_space.device, dtype=torch.float32)

    # [batch]
    score = torch.mean(x_gene_space[:, gene_indices], dim=1)
    # Ensure non‑negative penalty
    return F.relu(score)


def compute_grn_penalty(
    x_gene_space: torch.Tensor,
    grn_data: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """
    GRN penalty: minimize squared difference between TF and target expression.

    Given a set of TF-target rules, each with a weight w_i, we penalize:

        L_grn = E_batch [ sum_i w_i * (target_i - tf_i)^2 ]

    Args:
        x_gene_space:
            Decoded gene expression in HVG space, shape [batch, n_genes].
        grn_data:
            Dictionary with the following keys (as prepared in the notebook):
                - 'tf_indices':     List[int], indices of TFs in HVG.
                - 'target_indices': List[int], indices of targets in HVG.
                - 'weights':        1D tensor [n_rules], importance weights.

    Returns:
        loss_grn:
            Scalar tensor >= 0 (on x_gene_space.device).

    Behavior:
        - If grn_data is None or has no 'tf_indices', returns scalar 0.0.
    """
    if not grn_data or not grn_data.get("tf_indices"):
        return torch.tensor(0.0, device=x_gene_space.device, dtype=torch.float32)

    tf_indices = grn_data["tf_indices"]
    target_indices = grn_data["target_indices"]
    weights = grn_data["weights"].to(x_gene_space.device)

    # [batch, n_rules]
    tf_exp = x_gene_space[:, tf_indices]
    target_exp = x_gene_space[:, target_indices]

    squared_errors = (target_exp - tf_exp) ** 2          # [batch, n_rules]
    # Broadcast weights over batch and sum over rules
    weighted_squared = torch.sum(weights * squared_errors, dim=1)  # [batch]
    return torch.mean(weighted_squared)                              # scalar >= 0


# ---------------- Combined biological loss (backward / forward) ----------------


def compute_biological_loss_backward(
    config: Dict[str, Any],
    batch_x_gene: torch.Tensor,
    batch_x_next_gene: torch.Tensor,
    grn_data: Optional[Dict[str, Any]],
    death_gene_indices: Iterable[int],
) -> torch.Tensor:
    """
    Biological penalties for a forward-in-time step t -> t+1,
    applied when enforcing death + GRN constraints.

    Interpretation in terms of (x_k, x_{k+1}):
        - GRN consistency is enforced at the next step x_{k+1}.
        - Death penalty is applied at the current step x_k.

    Components (all >= 0):
        L_bio_death_forward =
            λ_grn   * L_grn(x_{k+1})
          + λ_death * L_death(x_k)

    Args:
        config:
            Global training configuration, expected keys:
                - 'lambda_grn'   (float)
                - 'lambda_death' (float)
        batch_x_gene:
            Decoded gene expression at step k, shape [batch, n_genes].
        batch_x_next_gene:
            Decoded gene expression at step k+1, shape [batch, n_genes].
        grn_data:
            GRN dictionary as in compute_grn_penalty (may be None).
        death_gene_indices:
            Indices (in HVG) of death-related genes.

    Returns:
        loss_bio_backward:
            Scalar tensor on batch_x_gene.device.
    """
    device = batch_x_gene.device
    loss_bio = torch.tensor(0.0, device=device, dtype=torch.float32)

    lambda_grn = float(config.get("lambda_grn", 0.0))
    lambda_death = float(config.get("lambda_death", 0.0))

    # GRN penalty at step k+1
    if lambda_grn > 0.0 and grn_data:
        grn_penalty = compute_grn_penalty(batch_x_next_gene, grn_data)
        loss_bio = loss_bio + lambda_grn * grn_penalty

    # Death penalty at step k (HIGH death expression is bad)
    death_gene_indices = list(death_gene_indices)
    if lambda_death > 0.0 and len(death_gene_indices) > 0:
        death_score = compute_cell_fate_score(batch_x_gene, death_gene_indices)  # [batch]
        loss_bio = loss_bio + lambda_death * torch.mean(death_score)

    print(
        "[BIO DEBUG]",
        "lambda_grn=", lambda_grn,
        "lambda_death=", lambda_death,
        "len(death_idx)=", len(death_gene_indices),
       "loss_bio=", float(loss_bio.detach())
    )
    return loss_bio


def compute_biological_loss_forward(
    config: Dict[str, Any],
    batch_x_gene: torch.Tensor,
    batch_x_next_gene: torch.Tensor,
    grn_data: Optional[Dict[str, Any]],
    birth_gene_indices: Iterable[int],
) -> torch.Tensor:
    """
    Biological penalties for a backward-in-time step t+1 -> t,
    applied when enforcing birth + GRN constraints.

    Interpretation in terms of (x_k, x_{k+1}):
        - GRN consistency is enforced at the earlier state x_k.
        - Birth penalty is applied at the later state x_{k+1}.

    Components (all >= 0):
        L_bio_birth_backward =
            λ_grn   * L_grn(x_k)
          + λ_birth * L_birth(x_{k+1})

    Args:
        config:
            Global training configuration, expected keys:
                - 'lambda_grn'   (float)
                - 'lambda_birth' (float)
        batch_x_gene:
            Decoded gene expression at step k, shape [batch, n_genes].
        batch_x_next_gene:
            Decoded gene expression at step k+1, shape [batch, n_genes].
        grn_data:
            GRN dictionary as in compute_grn_penalty (may be None).
        birth_gene_indices:
            Indices (in HVG) of birth/proliferation-related genes.

    Returns:
        loss_bio_forward:
            Scalar tensor on batch_x_gene.device.
    """
    device = batch_x_gene.device
    loss_bio = torch.tensor(0.0, device=device, dtype=torch.float32)

    lambda_grn = float(config.get("lambda_grn", 0.0))
    lambda_birth = float(config.get("lambda_birth", 0.0))

    # GRN penalty at step k
    if lambda_grn > 0.0 and grn_data:
        grn_penalty = compute_grn_penalty(batch_x_gene, grn_data)
        loss_bio = loss_bio + lambda_grn * grn_penalty

    # Birth penalty at step k+1 (HIGH proliferation is bad)
    birth_gene_indices = list(birth_gene_indices)
    if lambda_birth > 0.0 and len(birth_gene_indices) > 0:
        birth_score = compute_cell_fate_score(batch_x_next_gene, birth_gene_indices)  # [batch]
        loss_bio = loss_bio + lambda_birth * torch.mean(birth_score)
    
    print(
        "[BIO DEBUG]",
        "lambda_grn=", lambda_grn,
        "lambda_birth=", lambda_birth,
        "len(birth_idx)=", len(birth_gene_indices),
       "loss_bio=", float(loss_bio.detach())
    )
    return loss_bio