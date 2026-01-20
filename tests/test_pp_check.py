"""Tests for gretapy.pp._check module."""

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest

from gretapy.pp._check import (
    _check_dataset,
    _check_datasets,
    _check_dts_grn,
    _check_grn,
    _check_metrics,
    _check_organism,
    _check_terms,
)


class TestCheckOrganism:
    """Tests for _check_organism function."""

    def test_valid_organism_hg38(self):
        """Test that hg38 is accepted."""
        _check_organism("hg38")  # Should not raise

    def test_valid_organism_mm10(self):
        """Test that mm10 is accepted."""
        _check_organism("mm10")  # Should not raise

    def test_invalid_organism_raises(self):
        """Test that invalid organism raises ValueError."""
        with pytest.raises(ValueError, match="Invalid organism"):
            _check_organism("invalid_organism")

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            _check_organism("")


class TestCheckDatasets:
    """Tests for _check_datasets function."""

    def test_none_returns_all(self):
        """Test that None returns all available datasets."""
        result = _check_datasets(organism="hg38", datasets=None)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_string_returns_list(self):
        """Test that string input returns list with single element."""
        result = _check_datasets(organism="hg38", datasets="pbmc10k")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "pbmc10k"

    def test_list_returns_list(self):
        """Test that list input returns the same list."""
        result = _check_datasets(organism="hg38", datasets=["pbmc10k"])

        assert isinstance(result, list)
        assert "pbmc10k" in result

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            _check_datasets(organism="hg38", datasets="nonexistent_dataset")

    def test_invalid_dataset_in_list_raises(self):
        """Test that invalid dataset in list raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            _check_datasets(organism="hg38", datasets=["pbmc10k", "invalid"])


class TestCheckMetrics:
    """Tests for _check_metrics function."""

    def test_none_returns_all(self):
        """Test that None returns all available metrics."""
        result = _check_metrics(organism="hg38", metrics=None)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_category_expansion(self):
        """Test that category name expands to all databases in category."""
        result = _check_metrics(organism="hg38", metrics="Prior Knowledge")

        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain databases from Prior Knowledge category
        assert any("CollecTRI" in m for m in result) or "CollecTRI" in result

    def test_metric_type_expansion(self):
        """Test that metric type expands to all databases of that type."""
        result = _check_metrics(organism="hg38", metrics="TF markers")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_database_name_direct(self):
        """Test that database name is returned directly."""
        result = _check_metrics(organism="hg38", metrics="CollecTRI")

        assert "CollecTRI" in result

    def test_list_input(self):
        """Test that list input works."""
        result = _check_metrics(organism="hg38", metrics=["CollecTRI", "ChIP-Atlas"])

        assert isinstance(result, list)
        assert "CollecTRI" in result
        assert "ChIP-Atlas" in result

    def test_invalid_metric_raises(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            _check_metrics(organism="hg38", metrics="nonexistent_metric")


class TestCheckGrn:
    """Tests for _check_grn function."""

    def test_valid_simple_grn(self, simple_grn):
        """Test that valid simple GRN passes."""
        result = _check_grn(grn=simple_grn)

        assert isinstance(result, pd.DataFrame)
        assert "source" in result.columns
        assert "target" in result.columns

    def test_valid_grn_with_score_and_cre(self, simple_grn):
        """Test that GRN with score and CRE columns passes."""
        result = _check_grn(grn=simple_grn)

        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert "cre" in result.columns

    def test_missing_source_column_raises(self):
        """Test that missing source column raises AssertionError."""
        invalid_grn = pd.DataFrame({"target": ["Gene1"], "score": [0.5]})

        with pytest.raises(AssertionError, match="source"):
            _check_grn(grn=invalid_grn)

    def test_missing_target_column_raises(self):
        """Test that missing target column raises AssertionError."""
        invalid_grn = pd.DataFrame({"source": ["TF1"], "score": [0.5]})

        with pytest.raises(AssertionError, match="target"):
            _check_grn(grn=invalid_grn)

    def test_invalid_cre_format_raises(self):
        """Test that invalid CRE format raises AssertionError."""
        invalid_grn = pd.DataFrame(
            {
                "source": ["TF1"],
                "target": ["Gene1"],
                "cre": ["invalid_format"],
            }
        )

        with pytest.raises(AssertionError, match="cre"):
            _check_grn(grn=invalid_grn)

    def test_removes_duplicates(self):
        """Test that duplicates are removed."""
        grn_with_dups = pd.DataFrame(
            {
                "source": ["TF1", "TF1", "TF1"],
                "target": ["Gene1", "Gene1", "Gene2"],
            }
        )

        result = _check_grn(grn=grn_with_dups)

        assert len(result) == 2

    def test_averages_scores_for_duplicates(self):
        """Test that scores are averaged for duplicate edges."""
        grn_with_dups = pd.DataFrame(
            {
                "source": ["TF1", "TF1"],
                "target": ["Gene1", "Gene1"],
                "score": [0.8, 0.4],
            }
        )

        result = _check_grn(grn=grn_with_dups)

        assert len(result) == 1
        assert result["score"].iloc[0] == pytest.approx(0.6)

    def test_non_dataframe_raises(self):
        """Test that non-DataFrame input raises AssertionError."""
        with pytest.raises(AssertionError):
            _check_grn(grn=[{"source": "TF1", "target": "Gene1"}])


class TestCheckDataset:
    """Tests for _check_dataset function."""

    def test_valid_anndata(self, adata):
        """Test that valid AnnData passes."""
        result = _check_dataset(organism="hg38", dataset=adata)

        assert isinstance(result, ad.AnnData)

    def test_valid_mudata(self, mudata_with_celltype):
        """Test that valid MuData passes."""
        result = _check_dataset(organism="hg38", dataset=mudata_with_celltype)

        assert isinstance(result, mu.MuData)

    def test_anndata_missing_celltype_raises(self):
        """Test that AnnData without celltype raises AssertionError."""
        adata = ad.AnnData(X=np.random.rand(10, 5))

        with pytest.raises(AssertionError, match="celltype"):
            _check_dataset(organism="hg38", dataset=adata)

    def test_mudata_missing_modalities_raises(self):
        """Test that MuData without rna/atac raises AssertionError."""
        rna = ad.AnnData(X=np.random.rand(10, 5))
        rna.obs["celltype"] = ["A"] * 10
        mdata = mu.MuData({"rna": rna})

        with pytest.raises(AssertionError, match="atac"):
            _check_dataset(organism="hg38", dataset=mdata)

    def test_invalid_type_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid type"):
            _check_dataset(organism="hg38", dataset={"key": "value"})


class TestCheckDtsGrn:
    """Tests for _check_dts_grn function."""

    def test_valid_overlap_anndata(self, adata, simple_grn):
        """Test that valid gene overlap passes for AnnData."""
        _check_dts_grn(dataset=adata, grn=simple_grn)  # Should not raise

    def test_valid_overlap_mudata(self, mudata_with_celltype, simple_grn):
        """Test that valid gene overlap passes for MuData."""
        _check_dts_grn(dataset=mudata_with_celltype, grn=simple_grn)  # Should not raise

    def test_no_overlap_raises(self, adata):
        """Test that no gene overlap raises AssertionError."""
        grn_no_overlap = pd.DataFrame(
            {
                "source": ["UnknownTF1", "UnknownTF2"],
                "target": ["UnknownGene1", "UnknownGene2"],
            }
        )

        with pytest.raises(AssertionError, match="genes from grn do not exist"):
            _check_dts_grn(dataset=adata, grn=grn_no_overlap)


class TestCheckTerms:
    """Tests for _check_terms function."""

    def test_none_with_string_dataset(self):
        """Test that None terms with string dataset loads from config."""
        result = _check_terms(organism="hg38", dataset="pbmc10k", terms=None)

        assert isinstance(result, dict)

    def test_none_with_object_dataset_raises(self, adata):
        """Test that None terms with object dataset raises ValueError."""
        with pytest.raises(ValueError, match="terms cannot be None"):
            _check_terms(organism="hg38", dataset=adata, terms=None)

    def test_invalid_db_in_terms_raises(self, adata):
        """Test that invalid database in terms raises AssertionError."""
        invalid_terms = {"NonexistentDB": ["term1", "term2"]}

        with pytest.raises(AssertionError, match="not found"):
            _check_terms(organism="hg38", dataset=adata, terms=invalid_terms)
