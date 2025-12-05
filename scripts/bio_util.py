from typing import Any, Tuple

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sde import SchrodingerBridgePolicy

# ---------------- Biological constraints preparation ----------------

def load_gene_set_from_json(file_path: str) -> Tuple[str | None, list[str] | None]:
    """
    Load a gene set from a MSigDB-style JSON file.

    Returns:
        (gene_set_name, gene_symbols) or (None, None) on error.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        gene_set_name = list(data.keys())[0]
        gene_symbols = data[gene_set_name]["geneSymbols"]
        print(f"Loaded {len(gene_symbols)} genes from {gene_set_name}")
        return gene_set_name, gene_symbols
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except (KeyError, IndexError):
        print(f"Error: Could not parse gene set from {file_path}. Check JSON structure.")
    return None, None


def build_grn_with_prior(
    adj_path: str,
    prior_edges_path: str,
    hvg_names_list: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Build GRN data using pySCENIC adjacencies and a prior-knowledge edge list,
    restricted to genes in the HVG list.

    Returns:
        grn_df_hvg: DataFrame with columns ['TF', 'target', 'importance', 'EdgeType']
                    filtered to HVG genes.
        grn_data:   dict with:
                        'tf_indices': list[int]
                        'target_indices': list[int]
                        'weights': torch.FloatTensor
    """
    adj_df = pd.read_csv(adj_path)
    prior_edges_df = pd.read_csv(prior_edges_path)

    grn_pi = pd.merge(
        adj_df,
        prior_edges_df,
        left_on=["TF", "target"],
        right_on=["Source", "Target"],
        how="inner",
    )

    grn_pi = grn_pi[["TF", "target", "importance", "EdgeType"]]
    grn_pi = grn_pi.sort_values( # type: ignore[arg-type]
        by="importance", 
        ascending=False
    ).reset_index(drop=True)

    if grn_pi.empty:
        print(
            "\n[build_grn_with_prior] WARNING: No overlap between pySCENIC adjacencies "
            "and prior edges. Returning empty GRN."
        )
        return grn_pi, {"tf_indices": [], "target_indices": [], "weights": torch.tensor([])}

    print(f"\nFound {len(grn_pi)} GRN edges overlapping with prior knowledge.")

    # Restrict to HVG genes
    grn_df_hvg = grn_pi[
        grn_pi["TF"].isin(hvg_names_list)
        & grn_pi["target"].isin(hvg_names_list)
    ].copy()

    tf_indices = [hvg_names_list.index(g) for g in grn_df_hvg["TF"].values]
    target_indices = [hvg_names_list.index(g) for g in grn_df_hvg["target"].values]
    weights = torch.tensor(grn_df_hvg["importance"].values, dtype=torch.float32)

    grn_data = {"tf_indices": tf_indices, "target_indices": target_indices, "weights": weights}
    print(f"Loaded {len(tf_indices)} GRN rules where both TF and target are in the HVG set.")

    return grn_df_hvg, grn_data


def plot_grn_elbow(grn_df_hvg: pd.DataFrame, out_path: str | None = None) -> None:
    """
    Elbow plot of GRN edge importance for HVG-filtered GRN.
    """
    if grn_df_hvg.empty:
        print("[plot_grn_elbow] Empty GRN DataFrame, skipping plot.")
        return

    grn_plot_df = grn_df_hvg.reset_index(drop=True)

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x=grn_plot_df.index,
        y=grn_plot_df["importance"],
        marker="o",
        markersize=5,
        label="Edge Importance (HVG-filtered)",
    )
    plt.yscale("log")
    plt.title("Elbow Plot of GRN Edge Importance (HVG-filtered)", fontsize=16)
    plt.xlabel("Rank of Edge (Sorted by Importance)", fontsize=12)
    plt.ylabel("pySCENIC Importance Score (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--", c="0.85")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------- Drift gene analysis ----------------

@torch.no_grad()
def compute_drift_tables(
    vae: Any,
    z_f: SchrodingerBridgePolicy,
    z_b: SchrodingerBridgePolicy,
    pre_cells_expr: torch.Tensor,
    post_cells_expr: torch.Tensor,
    gene_symbols: list[str],
    epsilon: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute forward and backward drift tables (per-gene mean drift and |drift|).

    Args:
        vae: trained VAE_scRNA model.
        z_f, z_b: forward/backward SchrodingerBridgePolicy instances.
        pre_cells_expr, on_cells_expr: gene expression tensors [n_cells, n_genes] (on correct device if large).
        gene_symbols: list of gene names matching columns of expression.
        epsilon: finite-difference step in latent space.

    Returns:
        forward_drift_df, backward_drift_df (each sorted by abs_drift desc).
    """
    device = next(vae.parameters()).device

    pre_cells_expr = pre_cells_expr.to(device)
    post_cells_expr = post_cells_expr.to(device)

    # Encode to latent space
    h_pre = vae.encoder(pre_cells_expr)
    mu_pre, _ = vae.fc_mu(h_pre), vae.fc_log_var(h_pre)

    h_post = vae.encoder(post_cells_expr)
    mu_post, _ = vae.fc_mu(h_post), vae.fc_log_var(h_post)

    def decode_to_genes(z: torch.Tensor) -> torch.Tensor:
        decoded_hidden = vae.decoder(z)
        return torch.nn.functional.softplus(vae.decoder_output(decoded_hidden))

    # Forward drift (Pre → Post)
    t_mid_pre = torch.full((mu_pre.shape[0],), 0.5, device=device)
    forward_velocity = z_f.net(mu_pre, t_mid_pre)
    mu_pre_perturbed = mu_pre + epsilon * forward_velocity
    recon_original = decode_to_genes(mu_pre)
    recon_perturbed = decode_to_genes(mu_pre_perturbed)
    gene_drift_forward = (recon_perturbed - recon_original) / epsilon

    # Backward drift (Post → Pre)
    t_mid_post = torch.full((mu_post.shape[0],), 0.5, device=device)
    backward_velocity = z_b.net(mu_post, t_mid_post)
    mu_post_perturbed = mu_post + epsilon * backward_velocity
    recon_original_post = decode_to_genes(mu_post)
    recon_perturbed_post = decode_to_genes(mu_post_perturbed)
    gene_drift_backward = (recon_perturbed_post - recon_original_post) / epsilon
    
    # Aggregate to tables
    mean_drift_forward = gene_drift_forward.mean(dim=0).cpu().numpy()
    mean_drift_backward = gene_drift_backward.mean(dim=0).cpu().numpy()

    forward_drift_df = pd.DataFrame(
        {
            "gene": gene_symbols,
            "drift_velocity": mean_drift_forward,
            "abs_drift": np.abs(mean_drift_forward),
        }
    ).sort_values("abs_drift", ascending=False)

    backward_drift_df = pd.DataFrame(
        {
            "gene": gene_symbols,
            "drift_velocity": mean_drift_backward,
            "abs_drift": np.abs(mean_drift_backward),
        }
    ).sort_values("abs_drift", ascending=False)

    return forward_drift_df, backward_drift_df


def plot_top_drift_genes(
    forward_drift_df: pd.DataFrame,
    backward_drift_df: pd.DataFrame,
    k: int = 20,
    out_path: str | None = None,
) -> None:
    """
    Plot top-k drift genes (forward and backward) as two horizontal bar charts.

    Args:
        forward_drift_df, backward_drift_df: tables with 'gene' and 'abs_drift' columns.
        k: number of top genes to plot.
        out_path: optional path to save the figure (png, pdf, etc.).
    """
    top_forward = forward_drift_df.head(k)
    top_backward = backward_drift_df.head(k)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Forward
    axes[0].barh(range(len(top_forward)), top_forward["abs_drift"].values, color="steelblue")
    axes[0].set_yticks(range(len(top_forward)))
    axes[0].set_yticklabels(top_forward["gene"].values, fontsize=10)
    axes[0].set_xlabel("Absolute Drift Velocity", fontsize=12)
    axes[0].set_title("Top Drift Genes - Forward Process\n(Pre → Post)", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    # Backward
    axes[1].barh(range(len(top_backward)), top_backward["abs_drift"].values, color="coral")
    axes[1].set_yticks(range(len(top_backward)))
    axes[1].set_yticklabels(top_backward["gene"].values, fontsize=10)
    axes[1].set_xlabel("Absolute Drift Velocity", fontsize=12)
    axes[1].set_title("Top Drift Genes - Backward Process\n(Post → Pre)", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


