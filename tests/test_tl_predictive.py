"""Tests for pygreta.tl._predictive module."""

import numpy as np
import pandas as pd

from pygreta.tl._predictive import _extract_data, _gset, _omics, _remove_zeros, _test_predictability


class TestRemoveZeros:
    """Tests for _remove_zeros function."""

    def test_removes_zero_rows(self):
        """Test that rows with zero y values are removed."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.0, 0.0, 2.0, 0.0])

        X_out, y_out = _remove_zeros(X, y)

        assert len(y_out) == 2
        assert X_out.shape == (2, 2)
        np.testing.assert_array_equal(y_out, [1.0, 2.0])
        np.testing.assert_array_equal(X_out, [[1, 2], [5, 6]])

    def test_keeps_all_nonzero(self):
        """Test that all rows are kept when no zeros in y."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1.0, 2.0])

        X_out, y_out = _remove_zeros(X, y)

        assert len(y_out) == 2
        np.testing.assert_array_equal(y_out, y)

    def test_empty_result_when_all_zeros(self):
        """Test with all zeros in y."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0.0, 0.0])

        X_out, y_out = _remove_zeros(X, y)

        assert len(y_out) == 0
        assert X_out.shape[0] == 0


class TestExtractData:
    """Tests for _extract_data function."""

    def test_with_anndata(self, adata):
        """Test data extraction from AnnData."""
        all_obs = adata.obs_names.values
        train_obs = all_obs[:70]
        test_obs = all_obs[70:]
        sources = ["PAX5", "GATA3"]
        target = "CD19"

        train_X, train_y, test_X, test_y = _extract_data(
            data=adata,
            train_obs_names=train_obs,
            test_obs_names=test_obs,
            target=target,
            sources=sources,
            mod_source=None,
            mod_target=None,
        )

        # Check shapes (may be less than original due to zero removal)
        assert train_X.shape[1] == 2  # 2 sources
        assert test_X.shape[1] == 2

    def test_with_mudata(self, mudata_with_celltype):
        """Test data extraction from MuData."""
        all_obs = mudata_with_celltype.obs_names.values
        train_obs = all_obs[:70]
        test_obs = all_obs[70:]
        sources = ["PAX5", "GATA3"]
        target = "CD19"

        train_X, train_y, test_X, test_y = _extract_data(
            data=mudata_with_celltype,
            train_obs_names=train_obs,
            test_obs_names=test_obs,
            target=target,
            sources=sources,
            mod_source="rna",
            mod_target="rna",
        )

        assert train_X.shape[1] == 2
        assert test_X.shape[1] == 2


class TestTestPredictability:
    """Tests for _test_predictability function."""

    def test_returns_dataframe(self, adata, simple_grn):
        """Test that function returns a DataFrame."""
        all_obs = adata.obs_names.values
        train_obs = all_obs[:70]
        test_obs = all_obs[70:]

        result = _test_predictability(
            data=adata,
            train_obs_names=train_obs,
            test_obs_names=test_obs,
            grn=simple_grn,
            col_source="source",
            col_target="target",
            mod_source=None,
            mod_target=None,
            ntop=5,
        )

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"target", "n_obs", "n_vars", "coef", "pval", "padj"}
        assert expected_cols == set(result.columns)

    def test_handles_empty_grn(self, adata):
        """Test with empty GRN."""
        all_obs = adata.obs_names.values
        train_obs = all_obs[:70]
        test_obs = all_obs[70:]
        empty_grn = pd.DataFrame(columns=["source", "target", "score"])

        result = _test_predictability(
            data=adata,
            train_obs_names=train_obs,
            test_obs_names=test_obs,
            grn=empty_grn,
            col_source="source",
            col_target="target",
            mod_source=None,
            mod_target=None,
            ntop=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestOmics:
    """Tests for _omics function."""

    def test_with_anndata(self, adata, simple_grn):
        """Test omics evaluation with AnnData."""
        prc, rcl, f01 = _omics(
            data=adata,
            grn=simple_grn,
            col_source="source",
            col_target="target",
            mod_source=None,
            mod_target=None,
            test_size=0.33,
            seed=42,
            ntop=5,
        )

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_with_mudata(self, mudata_with_celltype, simple_grn):
        """Test omics evaluation with MuData."""
        prc, rcl, f01 = _omics(
            data=mudata_with_celltype,
            grn=simple_grn,
            col_source="source",
            col_target="target",
            mod_source="rna",
            mod_target="rna",
            test_size=0.33,
            seed=42,
            ntop=5,
        )

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_reproducibility_with_seed(self, adata, simple_grn):
        """Test that results are reproducible with same seed."""
        result1 = _omics(
            data=adata,
            grn=simple_grn,
            col_source="source",
            col_target="target",
            mod_source=None,
            mod_target=None,
            seed=42,
        )

        result2 = _omics(
            data=adata,
            grn=simple_grn,
            col_source="source",
            col_target="target",
            mod_source=None,
            mod_target=None,
            seed=42,
        )

        assert result1 == result2


class TestGset:
    """Tests for _gset function (gene set enrichment)."""

    def test_basic_functionality(self, adata, simple_grn, gene_set_db):
        """Test basic gene set enrichment evaluation."""
        prc, rcl, f01 = _gset(
            adata=adata,
            grn=simple_grn,
            db=gene_set_db,
            thr_pval=0.01,
            thr_prop=0.20,
        )

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_handles_duplicates(self, adata, gene_set_db):
        """Test that function handles duplicate edges in GRN."""
        grn_with_dups = pd.DataFrame(
            {
                "source": ["PAX5", "PAX5", "PAX5"],
                "target": ["CD19", "CD19", "MS4A1"],
            }
        )

        # Should not crash
        prc, rcl, f01 = _gset(
            adata=adata,
            grn=grn_with_dups,
            db=gene_set_db,
            thr_pval=0.01,
            thr_prop=0.20,
        )

        assert isinstance(prc, float)
        assert isinstance(rcl, float)
        assert isinstance(f01, float)

    def test_different_thresholds(self, adata, simple_grn, gene_set_db):
        """Test with different threshold values."""
        # More permissive thresholds
        prc1, rcl1, f01_1 = _gset(
            adata=adata,
            grn=simple_grn,
            db=gene_set_db,
            thr_pval=0.5,
            thr_prop=0.01,
        )

        # More stringent thresholds
        prc2, rcl2, f01_2 = _gset(
            adata=adata,
            grn=simple_grn,
            db=gene_set_db,
            thr_pval=0.001,
            thr_prop=0.50,
        )

        # Both should return valid results
        assert 0 <= prc1 <= 1 and 0 <= prc2 <= 1
        assert 0 <= rcl1 <= 1 and 0 <= rcl2 <= 1
        assert 0 <= f01_1 <= 1 and 0 <= f01_2 <= 1
