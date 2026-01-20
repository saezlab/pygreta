"""Tests for gretapy.tl._eval module."""

import pandas as pd
import pytest

from gretapy.tl._eval import (
    _run_fileless_metric,
    _run_metric,
    _run_omics_metric,
    benchmark,
    eval_grn_dataset,
)


class TestRunMetric:
    """Tests for _run_metric function."""

    def test_reference_grn_metric(self, simple_grn, reference_grn_db):
        """Test Reference GRN metric."""
        genes = ["PAX5", "GATA3", "SPI1", "CD19", "MS4A1", "CD3E", "IL7R", "CD14"]

        result = _run_metric(
            metric_type="Reference GRN",
            db_name="CollecTRI",
            grn=simple_grn,
            db=reference_grn_db,
            genes=genes,
            peaks=[],
            cats=None,
            adata=None,
        )

        assert result is not None
        assert len(result) == 3  # prc, rcl, f01
        prc, rcl, f01 = result
        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_tf_markers_metric(self, simple_grn, tfm_db):
        """Test TF markers metric."""
        genes = ["PAX5", "GATA3", "SPI1", "RUNX1", "CD19", "MS4A1", "CD3E", "IL7R", "CD14"]

        result = _run_metric(
            metric_type="TF markers",
            db_name="Human Protein Atlas (HPA)",
            grn=simple_grn,
            db=tfm_db,
            genes=genes,
            peaks=[],
            cats=None,
            adata=None,
        )

        assert result is not None
        assert len(result) == 3

    def test_tf_pairs_metric(self, simple_grn, tfp_db):
        """Test TF pairs metric."""
        result = _run_metric(
            metric_type="TF pairs",
            db_name="Europe PMC",
            grn=simple_grn,
            db=tfp_db,
            genes=[],
            peaks=[],
            cats=None,
            adata=None,
        )

        assert result is not None
        assert len(result) == 3

    def test_gene_sets_metric(self, adata, simple_grn, gene_set_db):
        """Test Gene sets metric."""
        result = _run_metric(
            metric_type="Gene sets",
            db_name="Hallmarks",
            grn=simple_grn,
            db=gene_set_db,
            genes=[],
            peaks=[],
            cats=None,
            adata=adata,
        )

        assert result is not None
        assert len(result) == 3

    def test_unknown_metric_returns_none(self, simple_grn):
        """Test that unknown metric type returns None."""
        result = _run_metric(
            metric_type="Unknown Metric",
            db_name="Unknown DB",
            grn=simple_grn,
            db=None,
            genes=[],
            peaks=[],
            cats=None,
            adata=None,
        )

        assert result is None


class TestRunFilelessMetric:
    """Tests for _run_fileless_metric function."""

    def test_omics_metric(self, adata, simple_grn):
        """Test Omics metric."""
        result = _run_fileless_metric(
            metric_type="Omics",
            db_name="gene ~ TFs",
            dataset=adata,
            grn=simple_grn,
            adata=adata,
            is_mudata=False,
            has_cre=False,
        )

        assert result is not None
        assert len(result) == 3

    def test_steady_state_simulation(self, adata, simple_grn):
        """Test Steady state simulation metric."""
        result = _run_fileless_metric(
            metric_type="Steady state simulation",
            db_name="Boolean rules",
            dataset=adata,
            grn=simple_grn,
            adata=adata,
            is_mudata=False,
            has_cre=False,
        )

        # May return nan for small GRNs, but should not crash
        assert result is not None
        assert len(result) == 3

    def test_unknown_metric_returns_none(self, adata, simple_grn):
        """Test that unknown metric type returns None."""
        result = _run_fileless_metric(
            metric_type="Unknown Metric",
            db_name="Unknown DB",
            dataset=adata,
            grn=simple_grn,
            adata=adata,
            is_mudata=False,
            has_cre=False,
        )

        assert result is None


class TestRunOmicsMetric:
    """Tests for _run_omics_metric function."""

    def test_gene_tf_metric_anndata(self, adata, simple_grn):
        """Test gene ~ TFs metric with AnnData."""
        result = _run_omics_metric(
            db_name="gene ~ TFs",
            dataset=adata,
            grn=simple_grn,
            is_mudata=False,
            has_cre=False,
        )

        assert result is not None
        assert len(result) == 3

    def test_gene_tf_metric_mudata(self, mudata_with_celltype, simple_grn):
        """Test gene ~ TFs metric with MuData."""
        result = _run_omics_metric(
            db_name="gene ~ TFs",
            dataset=mudata_with_celltype,
            grn=simple_grn,
            is_mudata=True,
            has_cre=False,
        )

        assert result is not None
        assert len(result) == 3

    def test_gene_cre_metric_without_cre_returns_none(self, mudata_with_celltype, simple_grn):
        """Test that gene ~ CREs returns None without CRE column."""
        result = _run_omics_metric(
            db_name="gene ~ CREs",
            dataset=mudata_with_celltype,
            grn=simple_grn,
            is_mudata=True,
            has_cre=False,
        )

        assert result is None

    def test_gene_cre_metric_with_cre(self, mudata_with_celltype, simple_grn):
        """Test gene ~ CREs metric with CRE column."""
        result = _run_omics_metric(
            db_name="gene ~ CREs",
            dataset=mudata_with_celltype,
            grn=simple_grn,
            is_mudata=True,
            has_cre=True,
        )

        assert result is not None
        assert len(result) == 3


class TestEvalGrnDataset:
    """Tests for eval_grn_dataset function."""

    def test_dataframe_has_expected_columns(self, adata, simple_grn):
        """Test that DataFrame has expected columns."""
        result = eval_grn_dataset(
            organism="hg38",
            grn=simple_grn,
            dataset=adata,
            terms={},
            metrics=["CollecTRI", "Human Protein Atlas (HPA)"],
            min_edges=1,
        )

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"category", "metric", "db", "precision", "recall", "f01"}
        assert expected_cols == set(result.columns)

    def test_min_edges_threshold(self, adata):
        """Test that GRN with too few edges returns empty DataFrame."""
        small_grn = pd.DataFrame({"source": ["PAX5"], "target": ["CD19"]})

        result = eval_grn_dataset(
            organism="hg38",
            grn=small_grn,
            dataset=adata,
            terms={},
            metrics=["CollecTRI", "Human Protein Atlas (HPA)"],
            min_edges=10,  # Require more edges than we have
        )

        assert len(result) == 0

    def test_with_str(self, mudata_with_celltype, simple_grn):
        """Test evaluation with str name."""
        result = eval_grn_dataset(
            organism="hg38",
            grn=simple_grn,
            dataset="pbmc10k",
            terms={},
            metrics=["CollecTRI"],
            min_edges=1,
        )

        assert isinstance(result, pd.DataFrame)


class TestBenchmark:
    """Tests for benchmark function."""

    def test_single_grn_dataframe(self, adata, simple_grn):
        """Test benchmark with single GRN DataFrame."""
        benchmark(
            organism="hg38",
            grns=simple_grn,
            datasets=["pbmc10k"],
            terms=None,
            metrics=["CollecTRI", "Human Protein Atlas (HPA)"],
        )

    def test_invalid_grns_type_raises(self):
        """Test that invalid grns type raises ValueError."""
        with pytest.raises(ValueError, match="grns must be"):
            benchmark(
                organism="hg38",
                grns=[1, 2, 3],  # Invalid type
                datasets=None,
                terms=None,
                metrics=None,
            )

    def test_invalid_organism_raises(self, simple_grn):
        """Test that invalid organism raises ValueError."""
        with pytest.raises(ValueError, match="Invalid organism"):
            benchmark(
                organism="invalid",
                grns=simple_grn,
                datasets=None,
                terms=None,
                metrics=None,
            )

    def test_invalid_metrics_raises(self, simple_grn):
        """Test that invalid metrics raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            benchmark(
                organism="hg38",
                grns=simple_grn,
                datasets=None,
                terms=None,
                metrics=["nonexistent_metric"],
            )
