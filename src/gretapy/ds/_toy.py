from __future__ import annotations

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
from decoupler._download import _log

# Default TF-celltype associations for biologically structured expression
_TF_CELLTYPE = {
    "PAX5": "B cell",
    "GATA3": "T cell",
    "SPI1": "Monocyte",
}

# Default target genes per TF (immune-relevant, with some overlap between TFs)
_TF_TARGETS = {
    "PAX5": ["CD19", "MS4A1", "CD79A", "BCL2", "IRF4"],
    "GATA3": ["CD3E", "IL7R", "TCF7", "FOXP3", "RUNX1"],
    "SPI1": ["CD14", "CD19", "MS4A1", "IRF4", "RUNX1"],  # Overlaps with PAX5/GATA3
}

# Chromosome assignments for CREs (biologically plausible loci)
_TARGET_CHR = {
    "CD19": "chr16",
    "MS4A1": "chr11",
    "CD79A": "chr19",
    "BCL2": "chr18",
    "IRF4": "chr6",
    "CD3E": "chr11",
    "IL7R": "chr5",
    "TCF7": "chr5",
    "FOXP3": "chrX",
    "RUNX1": "chr21",
    "CD14": "chr5",
    "CD34": "chr1",
}


def toy(
    n_cells: int = 60,
    n_tfs: int = 3,
    n_targets_per_tf: int = 5,
    n_peaks_per_target: int = 1,
    celltypes: list[str] | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[mu.MuData, pd.DataFrame]:
    """
    Generate synthetic MuData and GRN for testing and demonstration.

    Creates biologically structured test data with RNA and ATAC modalities,
    along with a gene regulatory network (GRN) DataFrame.

    Parameters
    ----------
    n_cells : int
        Number of cells to generate. Should be divisible by number of celltypes
        for equal distribution. Default is 60.
    n_tfs : int
        Number of transcription factors. Must be between 1 and 3. Default is 3.
    n_targets_per_tf : int
        Number of target genes per TF. Must be between 1 and 5. Default is 5.
    n_peaks_per_target : int
        Number of CREs (peaks) per target gene. Default is 1.
    celltypes : list[str] | None
        Custom celltype names. If None, uses default celltypes based on n_tfs:
        ["B cell", "T cell", "Monocyte"]. Default is None.
    seed : int
        Random seed for reproducibility. Default is 42.
    verbose : bool
        Whether to log progress messages. Default is False.

    Returns
    -------
    tuple[mu.MuData, pd.DataFrame]
        A tuple containing:
        - MuData with 'rna' and 'atac' modalities
        - GRN DataFrame with columns: source, target, cre, score
    """
    np.random.seed(seed)

    # Validate parameters
    if n_tfs < 1 or n_tfs > 3:
        raise ValueError("n_tfs must be between 1 and 3")
    if n_targets_per_tf < 1 or n_targets_per_tf > 5:
        raise ValueError("n_targets_per_tf must be between 1 and 5")
    if n_peaks_per_target < 1:
        raise ValueError("n_peaks_per_target must be at least 1")

    # Select TFs and their properties
    tf_names = list(_TF_CELLTYPE.keys())[:n_tfs]
    tf_celltypes = [_TF_CELLTYPE[tf] for tf in tf_names]

    # Set up celltypes
    if celltypes is None:
        celltypes = tf_celltypes
    n_celltypes = len(celltypes)

    if n_cells % n_celltypes != 0:
        _log(
            f"n_cells={n_cells} not divisible by {n_celltypes} celltypes, distribution will be uneven",
            level="warning",
            verbose=verbose,
        )

    _log(f"Generating toy data with {n_cells} cells, {n_tfs} TFs", level="info", verbose=verbose)

    # Build gene list: TFs + unique targets (preserve order)
    target_genes = []
    seen = set()
    for tf in tf_names:
        for target in _TF_TARGETS[tf][:n_targets_per_tf]:
            if target not in seen:
                target_genes.append(target)
                seen.add(target)
    all_genes = tf_names + target_genes

    # Build GRN DataFrame
    grn_records = []
    peak_names = []
    peak_start_base = 10000000

    for tf in tf_names:
        targets = _TF_TARGETS[tf][:n_targets_per_tf]
        for target in targets:
            chrom = _TARGET_CHR.get(target, "chr1")
            for peak_idx in range(n_peaks_per_target):
                start = peak_start_base + peak_idx * 1000
                end = start + 500
                cre = f"{chrom}-{start}-{end}"
                peak_names.append(cre)
                score = np.random.uniform(0.4, 0.95)
                grn_records.append(
                    {
                        "source": tf,
                        "target": target,
                        "cre": cre,
                        "score": round(score, 2),
                    }
                )
            peak_start_base += n_peaks_per_target * 1000

    grn = pd.DataFrame(grn_records)

    # Create cell type assignments
    cells_per_type = n_cells // n_celltypes
    remainder = n_cells % n_celltypes
    celltype_list = []
    for i, ct in enumerate(celltypes):
        count = cells_per_type + (1 if i < remainder else 0)
        celltype_list.extend([ct] * count)

    # Create RNA expression matrix with biological structure
    n_genes = len(all_genes)
    X_rna = np.random.rand(n_cells, n_genes).astype(np.float32)

    # Add TF-specific expression patterns
    for i, tf in enumerate(tf_names):
        tf_idx = all_genes.index(tf)
        ct = tf_celltypes[i] if i < len(tf_celltypes) else celltypes[i % len(celltypes)]
        # Find cells of this celltype
        cell_mask = np.array([c == ct for c in celltype_list])
        X_rna[cell_mask, tf_idx] += 2.0

    # Create RNA AnnData
    rna = ad.AnnData(X=X_rna)
    rna.var_names = all_genes
    rna.obs_names = [f"Cell{i}" for i in range(n_cells)]
    rna.obs["celltype"] = celltype_list
    rna.layers["scaled"] = (X_rna - X_rna.mean(axis=0)) / (X_rna.std(axis=0) + 1e-6)

    # Create ATAC accessibility matrix
    n_peaks = len(peak_names)
    X_atac = np.random.rand(n_cells, n_peaks).astype(np.float32)

    # Create ATAC AnnData
    atac = ad.AnnData(X=X_atac)
    atac.var_names = peak_names
    atac.obs_names = rna.obs_names.copy()
    atac.obs["celltype"] = celltype_list

    # Create MuData
    mdata = mu.MuData({"rna": rna, "atac": atac})
    mdata.obs["celltype"] = celltype_list

    _log(
        f"Created MuData with {n_cells} cells, {n_genes} genes, {n_peaks} peaks",
        level="info",
        verbose=verbose,
    )
    _log(f"Created GRN with {len(grn)} edges", level="info", verbose=verbose)

    return mdata, grn
