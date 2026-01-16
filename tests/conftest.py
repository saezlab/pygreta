import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pyranges as pr
import pytest


def get_tf_names():
    """Return common TF names used across fixtures."""
    return ["PAX5", "GATA3", "SPI1", "RUNX1", "TCF7"]


def get_target_genes():
    """Return common target gene names used across fixtures."""
    return ["CD19", "MS4A1", "CD3E", "IL7R", "CD14", "CD79A", "CD34", "BCL2", "IRF4", "FOXP3", "TCF7", "RUNX1"]


def get_all_genes():
    """Return all unique gene names (TFs + targets) used across fixtures."""
    return list(dict.fromkeys(get_tf_names() + get_target_genes()))


def get_peak_names():
    """Return common peak coordinates (near immune gene loci)."""
    return [
        "chr16-28931000-28931500",  # CD19 promoter
        "chr16-28932000-28932500",  # CD19 enhancer
        "chr11-60223000-60223500",  # MS4A1 enhancer
        "chr11-60224000-60224500",  # MS4A1 enhancer 2
        "chr11-118209000-118209500",  # CD3E promoter
        "chr5-35871000-35871500",  # IL7R enhancer
        "chr5-140013000-140013500",  # CD14 promoter
        "chr19-41879000-41879500",  # CD79A promoter
        "chr18-63318000-63318500",  # BCL2 enhancer
        "chr6-411000-411500",  # IRF4 promoter
        "chr6-412000-412500",  # IRF4 enhancer
        "chr1-207940000-207940500",  # CD34 enhancer
        "chr5-134110000-134110500",  # TCF7 promoter
        "chrX-49250000-49250500",  # FOXP3 enhancer
        "chr21-34799000-34799500",  # RUNX1 promoter
        "chr21-34800000-34800500",  # RUNX1 enhancer
    ]


@pytest.fixture
def adata():
    """Basic AnnData object with gene universe matching simple_grn plus extra genes.

    Expression is biologically structured: B cells have higher PAX5, T cells have higher GATA3.
    Includes celltype annotation.
    """
    np.random.seed(42)
    # Gene universe: all genes from simple_grn plus a few extra not in GRN
    genes = get_all_genes() + ["ACTB", "GAPDH", "B2M"]  # housekeeping genes not in GRN
    n_cells = 60  # divisible by 3 for equal cell type distribution
    n_genes = len(genes)

    # Create expression data with biological structure
    X = np.random.rand(n_cells, n_genes).astype(np.float32)

    # Cell type assignments (equal thirds)
    celltypes = ["B cell"] * (n_cells // 3) + ["T cell"] * (n_cells // 3) + ["Monocyte"] * (n_cells // 3)

    # Add cell type-specific TF expression patterns
    pax5_idx = genes.index("PAX5")
    gata3_idx = genes.index("GATA3")
    spi1_idx = genes.index("SPI1")
    # B cells: higher PAX5
    X[: n_cells // 3, pax5_idx] += 2.0
    # T cells: higher GATA3
    X[n_cells // 3 : 2 * n_cells // 3, gata3_idx] += 2.0
    # Monocytes: higher SPI1
    X[2 * n_cells // 3 :, spi1_idx] += 1.5

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f"Cell{i}" for i in range(n_cells)]
    adata.obs["celltype"] = celltypes
    adata.layers["scaled"] = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return adata


@pytest.fixture
def simple_grn():
    """Realistic GRN DataFrame with TF-target-CRE relationships.

    Contains 3 TFs (PAX5, GATA3, SPI1) each regulating 5+ target genes
    through biologically plausible CREs near target gene loci.
    """
    return pd.DataFrame(
        {
            "source": [
                # PAX5 targets (B cell lineage TF)
                "PAX5",
                "PAX5",
                "PAX5",
                "PAX5",
                "PAX5",
                "PAX5",
                # GATA3 targets (T cell lineage TF)
                "GATA3",
                "GATA3",
                "GATA3",
                "GATA3",
                "GATA3",
                # SPI1/PU.1 targets (myeloid/B cell TF)
                "SPI1",
                "SPI1",
                "SPI1",
                "SPI1",
                "SPI1",
            ],
            "target": [
                # PAX5 regulates B cell genes
                "CD19",
                "MS4A1",
                "CD79A",
                "BCL2",
                "IRF4",
                "CD34",
                # GATA3 regulates T cell genes
                "CD3E",
                "IL7R",
                "TCF7",
                "FOXP3",
                "RUNX1",
                # SPI1 regulates myeloid and B cell genes
                "CD14",
                "CD19",
                "MS4A1",
                "IRF4",
                "RUNX1",
            ],
            "cre": [
                # CREs near PAX5 target genes
                "chr16-28931000-28931500",  # CD19 promoter region
                "chr11-60223000-60223500",  # MS4A1 enhancer
                "chr19-41879000-41879500",  # CD79A promoter
                "chr18-63318000-63318500",  # BCL2 enhancer
                "chr6-411000-411500",  # IRF4 promoter
                "chr1-207940000-207940500",  # CD34 enhancer
                # CREs near GATA3 target genes
                "chr11-118209000-118209500",  # CD3E promoter
                "chr5-35871000-35871500",  # IL7R enhancer
                "chr5-134110000-134110500",  # TCF7 promoter
                "chrX-49250000-49250500",  # FOXP3 enhancer
                "chr21-34799000-34799500",  # RUNX1 promoter
                # CREs near SPI1 target genes
                "chr5-140013000-140013500",  # CD14 promoter
                "chr16-28932000-28932500",  # CD19 enhancer (different from PAX5)
                "chr11-60224000-60224500",  # MS4A1 enhancer (different from PAX5)
                "chr6-412000-412500",  # IRF4 enhancer (different from PAX5)
                "chr21-34800000-34800500",  # RUNX1 enhancer (different from GATA3)
            ],
            "score": [
                # PAX5 activation scores
                0.85,
                0.72,
                0.91,
                0.68,
                0.55,
                0.43,
                # GATA3 activation scores
                0.88,
                0.79,
                0.65,
                0.71,
                0.52,
                # SPI1 activation scores
                0.92,
                0.61,
                0.58,
                0.49,
                0.67,
            ],
        }
    )


@pytest.fixture
def reference_grn_db():
    """Reference GRN database DataFrame with shared gene names."""
    return pd.DataFrame(
        {
            "source": ["PAX5", "PAX5", "GATA3", "SPI1", "RUNX1"],
            "target": ["CD19", "MS4A1", "IL7R", "CD14", "CD34"],
        }
    )


@pytest.fixture
def tfm_db():
    """TF markers database DataFrame (two columns: TF name and cell type)."""
    return pd.DataFrame(
        {
            0: ["PAX5", "GATA3", "SPI1", "RUNX1"],
            1: ["B cell", "T cell", "Monocyte", "Stem cell"],
        }
    )


@pytest.fixture
def tfp_db():
    """TF pairs database DataFrame."""
    return pd.DataFrame(
        {
            0: ["PAX5", "PAX5", "GATA3"],
            1: ["GATA3", "SPI1", "SPI1"],
        }
    )


@pytest.fixture
def pyranges_db():
    """PyRanges database for genomic tests with coordinates overlapping peak_names."""
    return pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": [
                    "chr16",
                    "chr16",
                    "chr11",
                    "chr11",
                    "chr11",
                    "chr5",
                    "chr5",
                    "chr19",
                    "chr18",
                    "chr6",
                    "chr1",
                    "chr5",
                    "chrX",
                    "chr21",
                ],
                "Start": [
                    28931100,
                    28932100,
                    60223100,
                    60224100,
                    118209100,
                    35871100,
                    140013100,
                    41879100,
                    63318100,
                    411100,
                    207940100,
                    134110100,
                    49250100,
                    34799100,
                ],
                "End": [
                    28931400,
                    28932400,
                    60223400,
                    60224400,
                    118209400,
                    35871400,
                    140013400,
                    41879400,
                    63318400,
                    411400,
                    207940400,
                    134110400,
                    49250400,
                    34799400,
                ],
                "Name": [
                    "CD19",
                    "CD19",
                    "MS4A1",
                    "MS4A1",
                    "CD3E",
                    "IL7R",
                    "CD14",
                    "CD79A",
                    "BCL2",
                    "IRF4",
                    "CD34",
                    "TCF7",
                    "FOXP3",
                    "RUNX1",
                ],
                "Score": [
                    "B cell",
                    "B cell",
                    "B cell",
                    "B cell",
                    "T cell",
                    "T cell",
                    "Monocyte",
                    "B cell",
                    "B cell",
                    "B cell",
                    "Stem cell",
                    "T cell",
                    "T cell",
                    "Stem cell",
                ],
            }
        )
    )


@pytest.fixture
def gene_set_db():
    """Gene set database for pathway enrichment.

    Contains genes that are both present and absent in adata to test filtering.
    Each pathway has enough genes to pass decoupler's tmin threshold.
    """
    return pd.DataFrame(
        {
            "source": [
                # B cell activation pathway (7 genes: 6 present, 1 absent)
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                "B_CELL_ACTIVATION",
                # T cell activation pathway (7 genes: 6 present, 1 absent)
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                "T_CELL_ACTIVATION",
                # Myeloid pathway (6 genes: 5 present, 1 absent)
                "MYELOID_DIFFERENTIATION",
                "MYELOID_DIFFERENTIATION",
                "MYELOID_DIFFERENTIATION",
                "MYELOID_DIFFERENTIATION",
                "MYELOID_DIFFERENTIATION",
                "MYELOID_DIFFERENTIATION",
            ],
            "target": [
                # B cell genes (CD19, MS4A1, CD79A, PAX5, BCL2, IRF4 present; EBF1 absent)
                "CD19",
                "MS4A1",
                "CD79A",
                "PAX5",
                "BCL2",
                "IRF4",
                "EBF1",
                # T cell genes (CD3E, IL7R, GATA3, TCF7, FOXP3, RUNX1 present; LEF1 absent)
                "CD3E",
                "IL7R",
                "GATA3",
                "TCF7",
                "FOXP3",
                "RUNX1",
                "LEF1",
                # Myeloid genes (CD14, SPI1, IRF4, CD34, RUNX1 present; CSF1R absent)
                "CD14",
                "SPI1",
                "IRF4",
                "CD34",
                "RUNX1",
                "CSF1R",
            ],
            "weight": [1.0] * 20,
        }
    )


@pytest.fixture
def mudata_with_celltype(adata):
    """MuData object with RNA and ATAC modalities, RNA inherited from adata."""
    np.random.seed(42)
    peak_names = get_peak_names()

    # RNA modality: use adata directly
    rna = adata.copy()

    # ATAC modality with peak_names (same cells as RNA)
    n_cells = rna.n_obs
    atac = ad.AnnData(X=np.random.rand(n_cells, len(peak_names)).astype(np.float32))
    atac.var_names = peak_names
    atac.obs_names = rna.obs_names.copy()
    atac.obs["celltype"] = rna.obs["celltype"].values.copy()

    mdata = mu.MuData({"rna": rna, "atac": atac})
    mdata.obs["celltype"] = rna.obs["celltype"].values.copy()

    return mdata


@pytest.fixture
def knocktf_db():
    """Mock KnockTF database for TF scoring and forecasting tests."""
    np.random.seed(42)
    all_genes = get_all_genes()
    n_experiments = 5
    exp_names = [f"Exp{i}" for i in range(n_experiments)]

    X = np.random.randn(n_experiments, len(all_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs_names = exp_names
    adata.obs["source"] = ["PAX5", "GATA3", "PAX5", "SPI1", "GATA3"]
    adata.obs["logFC"] = [-1.0, -0.8, -1.2, -0.6, -0.9]
    adata.obs["Tissue.Type"] = ["Blood", "Blood", "Brain", "Blood", "Brain"]

    return adata
