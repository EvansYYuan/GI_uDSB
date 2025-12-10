#!dir_to_python_when_executable 

import os
import sys
import json
import scanpy as sc
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import pandas as pd
from collections import Counter

work_dir = "/home/yyuan/ICB_TCE/"  # adjust if needed
script_dir = os.path.join(work_dir, "scripts")

if script_dir not in sys.path:
    sys.path.append(script_dir)

iter_dir = os.path.join(work_dir, "iter_results")
os.makedirs(iter_dir, exist_ok=True)

summary_dir = os.path.join(iter_dir, "summaries")
os.makedirs(summary_dir, exist_ok=True)

from vae import *
from sde import *
from bio_con import *
from bio_util import *
from training_util import *
from joint_train import *

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


adata_brca_t = sc.read_h5ad(os.path.join(work_dir, "data/brca_t_cell.h5ad"))

# QC and filter highly variable genes
sc.pp.filter_cells(adata_brca_t, min_genes = 200)
sc.pp.filter_genes(adata_brca_t, min_cells = 3)
sc.pp.highly_variable_genes(adata_brca_t, n_top_genes = 3000, subset = True)

pre_treatment_mask = adata_brca_t.obs["pre_post"] == "Pre"
post_treatment_mask = adata_brca_t.obs["pre_post"] == "Post"

# Convert the sparse matrix to a dense numpy array, then to a PyTorch tensor
if hasattr(adata_brca_t.X, "toarray"):
    expression_data = torch.tensor(adata_brca_t.X.toarray(), dtype=torch.float32)
else: # If it's already a dense array
    expression_data = torch.tensor(adata_brca_t.X, dtype=torch.float32)

input_dim = expression_data.shape[1]

LATENT_DIM = 20
NUM_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

# KL Annealing Parameters
KL_START_EPOCH = 3  # Start KL annealing earlier in light pre-training
KL_WARMUP_EPOCHS = 10


class DataSampler:
    def __init__(self, data, device):
        self.data = data.to(device)

    def sample(self, batch_size):
        idx = torch.randint(0, len(self.data), (batch_size,))
        return self.data[idx]

full_config_path = os.path.join(work_dir, "trained_models/full_config.json")
with open(full_config_path, "r") as f: 
    config = json.load(f)

config_abl = config.copy()
config_abl["lambda_bio"] = 0.0
config_abl["lambda_grn"] = 0.0
config_abl["lambda_death"] = 0.0
config_abl["lambda_birth"] = 0.0

# Prepare cell fate gene sets
cell_death_files = [
    os.path.join(work_dir, "GO_geneset/HALLMARK_APOPTOSIS.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_P53_PATHWAY.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_UNFOLDED_PROTEIN_RESPONSE.v2025.1.Hs.json"),
]

cell_birth_files = [
    os.path.join(work_dir, "GO_geneset/HALLMARK_E2F_TARGETS.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_G2M_CHECKPOINT.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_MYC_TARGETS_V1.v2025.1.Hs.json"),
    os.path.join(work_dir, "GO_geneset/HALLMARK_MYC_TARGETS_V2.v2025.1.Hs.json"),
]

all_death_genes = set()
all_birth_genes = set()

print("'Cell Death' Gene Sets:")
for file_path in cell_death_files:
    name, genes = load_gene_set_from_json(file_path)
    if genes:
        all_death_genes.update(genes)

print("\n'Cell Birth' Gene Sets:")
for file_path in cell_birth_files:
    name, genes = load_gene_set_from_json(file_path)
    if genes:
        all_birth_genes.update(genes)

all_death_genes_list = sorted(list(all_death_genes))
all_birth_genes_list = sorted(list(all_birth_genes))

# Constraint preparation using HVG subset
hvg_names_list = adata_brca_t.var['feature_name'].tolist()

# Death genes & their indices within the HVG list
death_genes_in_hvg = [g for g in all_death_genes_list if g in hvg_names_list]
death_gene_indices = [hvg_names_list.index(g) for g in death_genes_in_hvg]
print(f"\nFound {len(death_gene_indices)} matching cell death genes in the HVG set.")

# Birth genes & their indices within the HVG list
birth_genes_in_hvg = [g for g in all_birth_genes_list if g in hvg_names_list]
birth_gene_indices = [hvg_names_list.index(g) for g in birth_genes_in_hvg]
print(f"Found {len(birth_gene_indices)} matching cell birth genes in the HVG set.")

# Prepare GRN constraints
adj_file = os.path.join(work_dir, "data/brca_t_cell_adj.csv")
prior_edges_file = os.path.join(work_dir, "data/TCE_prior_edges.csv")

grn_df_hvg, grn_data = build_grn_with_prior(
    adj_path=adj_file,
    prior_edges_path=prior_edges_file,
    hvg_names_list=hvg_names_list,
)

print(f"GRN constraints (HVG-filtered): {grn_df_hvg.shape[0]} edges")

def run_single_seed(seed: int, config: dict, config_abl: dict):
    """
    Run original + ablation training pipeline for a single random seed.

    All results are stored under:
        iter_results / run_{seed:03d}
    """

    print("\n" + "=" * 80)
    print(f"[Seed {seed}] Starting running whole pipeline...")
    print("=" * 80)

    # Set paths for this seed
    run_tag = f"run_{seed:03d}"
    run_dir = os.path.join(iter_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    cfg = config.copy()
    cfg_abl = config_abl.copy()

    # -------------------------
    # VAE pretraining
    # -------------------------
    print(f"\n[Seed {seed}] Starting VAE pre-training...\n")
    set_seed(seed)
    dataset = TensorDataset(expression_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    vae_pre = VAE_scRNA(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(vae_pre.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    vae_pre.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        # Calculate beta for KL annealing
        if epoch < KL_START_EPOCH:
            beta = 0.0
        else:
            beta = min(1.0, (epoch - KL_START_EPOCH) / KL_WARMUP_EPOCHS)

        for (batch_features,) in dataloader:
            batch_features = batch_features.to(device)

            # Forward pass
            recon_x, mu, log_var = vae_pre(batch_features)

            # Compute loss
            loss = elbo_loss(batch_features, recon_x, mu, log_var, beta=beta)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)

        print(f"    Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Beta: {beta:.3f}, Average Loss: {avg_loss:.4f}")

    print(f"\n[Seed {seed}] VAE pre-training complete.")

    vae_pretrain_path = os.path.join(run_dir, f"vae_pretrain_{run_tag}.pth")
    torch.save(vae_pre.state_dict(), vae_pretrain_path)

    # Latent embeddings for bridge
    latent_mu = compute_latent_embeddings(vae_pre, expression_data, device=device)
    latent_embeddings = latent_mu.numpy()

    pre_embeddings = torch.tensor(
        latent_embeddings[pre_treatment_mask], dtype=torch.float32
    ).to(device)
    post_embeddings = torch.tensor(
        latent_embeddings[post_treatment_mask], dtype=torch.float32
    ).to(device)

    p_sampler = DataSampler(pre_embeddings, device=device)
    q_sampler = DataSampler(post_embeddings, device=device)

    # -------------------------
    # Joint training (original)
    # -------------------------
    print(f"\n[Seed {seed}] Starting joint training (original)...")
    set_seed(seed)

    # Load pretrained VAE weights
    vae = load_trained_vae(
        model_path=vae_pretrain_path,
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        device=device,
    )

    dyn = VESDE(cfg, p_sampler, q_sampler)
    ts = torch.linspace(cfg["t0"], cfg["T"], cfg["interval"]).to(device)

    net_f = MLP(input_dim=cfg["data_dim"][0], output_dim=cfg["data_dim"][0]).to(device)
    net_b = MLP(input_dim=cfg["data_dim"][0], output_dim=cfg["data_dim"][0]).to(device)

    z_f = SchrodingerBridgePolicy(cfg, "forward", dyn, net_f)
    z_b = SchrodingerBridgePolicy(cfg, "backward", dyn, net_b)

    optimizer_f = torch.optim.Adam(z_f.parameters(), lr=cfg["lr"])
    optimizer_b = torch.optim.Adam(z_b.parameters(), lr=cfg["lr"])
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=cfg.get("lr_vae", 1e-4))

    vae_decoder = lambda z: vae.decoder_output(vae.decoder(z))

    training_history = run_joint_training_loop(
        config=cfg,
        dyn=dyn,
        ts=ts,
        vae=vae,
        vae_decoder=vae_decoder,
        z_f=z_f,
        z_b=z_b,
        optimizer_f=optimizer_f,
        optimizer_b=optimizer_b,
        optimizer_vae=optimizer_vae,
        expression_data=expression_data,
        grn_data=grn_data,
        death_gene_indices=death_gene_indices,
        birth_gene_indices=birth_gene_indices,
        device=device,
    )

    # Save original models + history for this run
    torch.save(vae.state_dict(), os.path.join(run_dir, f"vae_original_{run_tag}.pth"))
    torch.save(z_f.state_dict(), os.path.join(run_dir, f"z_f_original_{run_tag}.pth"))
    torch.save(z_b.state_dict(), os.path.join(run_dir, f"z_b_original_{run_tag}.pth"))

    hist_path = os.path.join(run_dir, f"training_history_original_{run_tag}.json")
    with open(hist_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\n[Seed {seed}] Joint training (original) complete; models and history saved.")

    # -------------------------
    # Ablation (no biology constraints)
    # -------------------------
    set_seed(seed)
    print(f"\n[Seed {seed}] Starting joint training (ablation)...")

    vae_abl = load_trained_vae(
        model_path=vae_pretrain_path,
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        device=device,
    )

    z_f_abl = SchrodingerBridgePolicy(
        cfg_abl,
        "forward",
        dyn,
        MLP(cfg_abl["data_dim"][0], cfg_abl["data_dim"][0]).to(device),
    ).to(device)

    z_b_abl = SchrodingerBridgePolicy(
        cfg_abl,
        "backward",
        dyn,
        MLP(cfg_abl["data_dim"][0], cfg_abl["data_dim"][0]).to(device),
    ).to(device)

    optimizer_vae_abl = torch.optim.Adam(
        vae_abl.parameters(), lr=cfg_abl["lr"], weight_decay=1e-4
    )
    optimizer_f_abl = torch.optim.Adam(
        z_f_abl.parameters(), lr=cfg_abl["lr"], weight_decay=1e-4
    )
    optimizer_b_abl = torch.optim.Adam(
        z_b_abl.parameters(), lr=cfg_abl["lr"], weight_decay=1e-4
    )
    vae_decoder_abl = lambda z: vae_abl.decoder_output(vae_abl.decoder(z))

    ablation_history = run_joint_training_loop(
        config=cfg_abl,
        dyn=dyn,
        ts=ts,
        vae=vae_abl,
        vae_decoder=vae_decoder_abl,
        z_f=z_f_abl,
        z_b=z_b_abl,
        optimizer_f=optimizer_f_abl,
        optimizer_b=optimizer_b_abl,
        optimizer_vae=optimizer_vae_abl,
        expression_data=expression_data,
        grn_data=grn_data,
        death_gene_indices=death_gene_indices,
        birth_gene_indices=birth_gene_indices,
        device=device,
    )

    # Save ablation models + history
    torch.save(vae_abl.state_dict(), os.path.join(run_dir, f"vae_ablation_{run_tag}.pth"))
    torch.save(z_f_abl.state_dict(), os.path.join(run_dir, f"z_f_ablation_{run_tag}.pth"))
    torch.save(z_b_abl.state_dict(), os.path.join(run_dir, f"z_b_ablation_{run_tag}.pth"))

    ablation_hist_path = os.path.join(run_dir, f"training_history_ablation_{run_tag}.json")
    with open(ablation_hist_path, "w") as f:
        json.dump(ablation_history, f, indent=2)
    print(f"\n[Seed {seed}] Joint training (ablation) complete; models and history saved.")

    # Drift genes (original + ablation)
    pre_cells_expr = expression_data[pre_treatment_mask].to(device)
    post_cells_expr = expression_data[post_treatment_mask].to(device)
    gene_symbols = hvg_names_list  # HVG feature names prepared above

    drift_fwd_orig, drift_bwd_orig = compute_drift_tables(
        vae=vae,
        z_f=z_f,
        z_b=z_b,
        pre_cells_expr=pre_cells_expr,
        post_cells_expr=post_cells_expr,
        gene_symbols=gene_symbols,
    )
    drift_fwd_abl, drift_bwd_abl = compute_drift_tables(
        vae=vae_abl,
        z_f=z_f_abl,
        z_b=z_b_abl,
        pre_cells_expr=pre_cells_expr,
        post_cells_expr=post_cells_expr,
        gene_symbols=gene_symbols,
    )

    drift_fwd_orig.to_csv(os.path.join(run_dir, "drift_genes_forward_original.csv"), index=False)
    drift_bwd_orig.to_csv(os.path.join(run_dir, "drift_genes_backward_original.csv"), index=False)
    drift_fwd_abl.to_csv(os.path.join(run_dir, "drift_genes_forward_ablation.csv"), index=False)
    drift_bwd_abl.to_csv(os.path.join(run_dir, "drift_genes_backward_ablation.csv"), index=False)

    print(f"\n[Seed {seed}] Drift genes saved in {run_dir}")
    print(f"\n[Seed {seed}] run finished.")


# Loop over 100 seeds and run full pipeline
NUM_SEEDS = 100
SEEDS = list(range(1, NUM_SEEDS + 1))

for s in SEEDS:
    run_single_seed(seed=s, config=config, config_abl=config_abl)

# Select top 100 drift genes per run to generate summary files
TOP_K = 100

def aggregate_drift_counts(filename: str, top_k: int = TOP_K, gene_col: str = "gene") -> pd.DataFrame:
    """
    Scan iter_results/run_*/<filename>, collect top-K genes from each file,
    and return a DataFrame with total counts per gene.
    """
    counts = Counter()

    for run_dir in sorted(glob.glob(os.path.join(iter_dir, "run_*"))):
        path = os.path.join(run_dir, filename)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # Choose gene column: prefer gene_col, otherwise first column
        col = gene_col if gene_col in df.columns else df.columns[0]

        top_genes = df[col].head(top_k)
        counts.update(top_genes)

    summary_df = (
        pd.DataFrame({"gene": list(counts.keys()), "count": list(counts.values())})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df

# Build summaries for all four drift tables
summary_fwd_orig = aggregate_drift_counts("drift_genes_forward_original.csv")
summary_bwd_orig = aggregate_drift_counts("drift_genes_backward_original.csv")
summary_fwd_abl  = aggregate_drift_counts("drift_genes_forward_ablation.csv")
summary_bwd_abl  = aggregate_drift_counts("drift_genes_backward_ablation.csv")

# Save to iter_results/summaries/
summary_fwd_orig.to_csv(os.path.join(summary_dir, "summary_drift_forward_original.csv"), index=False)
summary_bwd_orig.to_csv(os.path.join(summary_dir, "summary_drift_backward_original.csv"), index=False)
summary_fwd_abl.to_csv(os.path.join(summary_dir, "summary_drift_forward_ablation.csv"), index=False)
summary_bwd_abl.to_csv(os.path.join(summary_dir, "summary_drift_backward_ablation.csv"), index=False)

# Summary file analysis
TOP_N = 20

summary_fwd_orig = pd.read_csv(os.path.join(summary_dir, "summary_drift_forward_original.csv"))
summary_bwd_orig = pd.read_csv(os.path.join(summary_dir, "summary_drift_backward_original.csv"))
summary_fwd_abl  = pd.read_csv(os.path.join(summary_dir, "summary_drift_forward_ablation.csv"))
summary_bwd_abl  = pd.read_csv(os.path.join(summary_dir, "summary_drift_backward_ablation.csv"))

# Add frequency columns: how often each gene appears across NUM_SEEDS runs
for df in (summary_fwd_orig, summary_bwd_orig, summary_fwd_abl, summary_bwd_abl):
    df["freq"] = (df["count"] / NUM_SEEDS).round(2)

# Take top-N by frequency
fwd_orig_top = summary_fwd_orig.sort_values("freq", ascending=False).head(TOP_N).copy()
bwd_orig_top = summary_bwd_orig.sort_values("freq", ascending=False).head(TOP_N).copy()
fwd_abl_top  = summary_fwd_abl.sort_values("freq",  ascending=False).head(TOP_N).copy()
bwd_abl_top  = summary_bwd_abl.sort_values("freq",  ascending=False).head(TOP_N).copy()

# Within each model: common fwd vs bwd (your preferred logic)

# Original model: forward vs backward
orig_common_fb = fwd_orig_top.merge(
    bwd_orig_top, on="gene", how="inner", suffixes=("_fwd", "_bwd")
)
orig_fwd_only = fwd_orig_top[~fwd_orig_top["gene"].isin(bwd_orig_top["gene"])]
orig_bwd_only = bwd_orig_top[~bwd_orig_top["gene"].isin(fwd_orig_top["gene"])]

# Ablation model: forward vs backward
abl_common_fb = fwd_abl_top.merge(
    bwd_abl_top, on="gene", how="inner", suffixes=("_fwd", "_bwd")
)
abl_fwd_only = fwd_abl_top[~fwd_abl_top["gene"].isin(bwd_abl_top["gene"])]
abl_bwd_only = bwd_abl_top[~bwd_abl_top["gene"].isin(fwd_abl_top["gene"])]

# Across models: original vs ablation for same direction

# Forward: original vs ablation (all genes, outer join)
fwd_models_genes = (
    summary_fwd_orig[["gene", "freq"]].rename(columns={"freq": "freq_orig"})
    .merge(
        summary_fwd_abl[["gene", "freq"]].rename(columns={"freq": "freq_abl"}),
        on="gene",
        how="outer",
    )
    .fillna(0.0)
).round(2)
fwd_models_genes["freq_diff"] = fwd_models_genes["freq_orig"] - fwd_models_genes["freq_abl"]

# Backward: original vs ablation
bwd_models_genes = (
    summary_bwd_orig[["gene", "freq"]].rename(columns={"freq": "freq_orig"})
    .merge(
        summary_bwd_abl[["gene", "freq"]].rename(columns={"freq": "freq_abl"}),
        on="gene",
        how="outer",
    )
    .fillna(0.0)
).round(2)
bwd_models_genes["freq_diff"] = bwd_models_genes["freq_orig"] - bwd_models_genes["freq_abl"]

print("Orig fwd/bwd: common =", len(orig_common_fb),
      "| fwd-only =", len(orig_fwd_only),
      "| bwd-only =", len(orig_bwd_only))
print("Abl fwd/bwd:  common =", len(abl_common_fb),
      "| fwd-only =", len(abl_fwd_only),
      "| bwd-only =", len(abl_bwd_only))

# Save per-gene forward/backward frequency comparisons
fwd_models_genes.to_csv(
    os.path.join(summary_dir, "summary_drift_forward_org_vs_abl.csv"),
    index=False,
    float_format="%.2f",
)
bwd_models_genes.to_csv(
    os.path.join(summary_dir, "summary_drift_backward_org_vs_abl.csv"),
    index=False,
    float_format="%.2f",
)

