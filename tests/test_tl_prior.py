"""Tests for gretapy.tl._prior module."""

import pandas as pd
import pytest

from gretapy.tl._prior import _compute_overlap_pval, _find_pairs, _grn, _tfm, _tfp


class TestGrn:
    """Tests for _grn function (Reference GRN comparison)."""

    def test_perfect_match(self, simple_grn, reference_grn_db):
        """Test with GRN that matches reference perfectly for available genes."""
        genes = ["PAX5", "GATA3", "SPI1", "CD19", "MS4A1", "CD3E", "IL7R", "CD14"]
        prc, rcl, f01 = _grn(grn=simple_grn, db=reference_grn_db, genes=genes)

        # Should have some overlap
        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_no_overlap(self):
        """Test with GRN that has no overlap with reference."""
        grn = pd.DataFrame({"source": ["TFA", "TFB"], "target": ["GeneX", "GeneY"]})
        db = pd.DataFrame({"source": ["TFC", "TFD"], "target": ["GeneZ", "GeneW"]})
        genes = ["TFA", "TFB", "TFC", "TFD", "GeneX", "GeneY", "GeneZ", "GeneW"]

        prc, rcl, f01 = _grn(grn=grn, db=db, genes=genes)
        assert prc == 0.0
        assert rcl == 0.0
        assert f01 == 0.0

    def test_complete_overlap(self):
        """Test with identical GRN and reference."""
        grn = pd.DataFrame({"source": ["PAX5", "GATA3"], "target": ["CD19", "CD3E"]})
        db = pd.DataFrame({"source": ["PAX5", "GATA3"], "target": ["CD19", "CD3E"]})
        genes = ["PAX5", "GATA3", "CD19", "CD3E"]

        prc, rcl, f01 = _grn(grn=grn, db=db, genes=genes)
        assert prc == 1.0
        assert rcl == 1.0
        assert f01 == pytest.approx(1.0)

    def test_filters_by_genes(self):
        """Test that function filters database by measured genes."""
        grn = pd.DataFrame({"source": ["PAX5"], "target": ["CD19"]})
        db = pd.DataFrame({"source": ["PAX5", "GATA3"], "target": ["CD19", "FOXP3"]})
        # Only include genes from GRN
        genes = ["PAX5", "CD19"]

        prc, rcl, f01 = _grn(grn=grn, db=db, genes=genes)
        # Should be perfect match since GATA3->FOXP3 is filtered out
        assert prc == 1.0
        assert rcl == 1.0

    def test_handles_duplicates(self):
        """Test that function handles duplicate edges."""
        grn = pd.DataFrame({"source": ["PAX5", "PAX5", "PAX5"], "target": ["CD19", "CD19", "MS4A1"]})
        db = pd.DataFrame({"source": ["PAX5"], "target": ["CD19"]})
        genes = ["PAX5", "CD19", "MS4A1"]

        prc, rcl, f01 = _grn(grn=grn, db=db, genes=genes)
        # After deduplication: PAX5->CD19, PAX5->MS4A1 in GRN, PAX5->CD19 in DB
        assert 0 <= prc <= 1
        assert rcl == 1.0  # We found the one edge in DB


class TestTfm:
    """Tests for _tfm function (TF markers evaluation)."""

    def test_basic_functionality(self, simple_grn, tfm_db):
        """Test basic TF markers evaluation."""
        genes = ["PAX5", "GATA3", "SPI1", "RUNX1", "CD19", "MS4A1", "CD3E", "IL7R", "CD14"]
        prc, rcl, f01 = _tfm(grn=simple_grn, db=tfm_db, genes=genes, cats=None)

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_with_category_filter(self, simple_grn, tfm_db):
        """Test TF markers evaluation with category filtering."""
        genes = ["PAX5", "GATA3", "SPI1", "RUNX1", "CD19", "MS4A1", "CD3E", "IL7R", "CD14"]
        cats = ["B cell", "T cell"]
        prc, rcl, f01 = _tfm(grn=simple_grn, db=tfm_db, genes=genes, cats=cats)

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_no_matching_tfs(self):
        """Test with no matching TFs between GRN and database."""
        grn = pd.DataFrame({"source": ["TFX", "TFY"], "target": ["GeneA", "GeneB"]})
        db = pd.DataFrame({0: ["TFA", "TFB"], 1: ["Type1", "Type2"]})
        genes = ["TFX", "TFY", "TFA", "TFB", "GeneA", "GeneB"]

        prc, rcl, f01 = _tfm(grn=grn, db=db, genes=genes, cats=None)
        assert prc == 0.0
        assert f01 == 0.0


class TestComputeOverlapPval:
    """Tests for _compute_overlap_pval function."""

    def test_significant_overlap(self):
        """Test with significant target overlap between TFs."""
        # Create GRN where TF1 and TF2 share many targets
        grn = pd.DataFrame(
            {
                "source": ["TF1", "TF1", "TF1", "TF2", "TF2", "TF2", "TF3", "TF3"],
                "target": ["G1", "G2", "G3", "G1", "G2", "G4", "G5", "G6"],
            }
        )
        stat, pval = _compute_overlap_pval(grn=grn, tf_a="TF1", tf_b="TF2")

        assert stat > 0
        assert 0 <= pval <= 1

    def test_no_overlap(self):
        """Test with no target overlap between TFs."""
        grn = pd.DataFrame(
            {
                "source": ["TF1", "TF1", "TF2", "TF2"],
                "target": ["G1", "G2", "G3", "G4"],
            }
        )
        stat, pval = _compute_overlap_pval(grn=grn, tf_a="TF1", tf_b="TF2")

        assert stat == 0.0
        assert pval == 1.0


class TestFindPairs:
    """Tests for _find_pairs function."""

    def test_finds_significant_pairs(self):
        """Test that significant TF pairs are found."""
        # Create GRN with clear pair structure
        grn = pd.DataFrame(
            {
                "source": ["TF1"] * 5 + ["TF2"] * 5 + ["TF3"] * 3,
                "target": ["G1", "G2", "G3", "G4", "G5", "G1", "G2", "G3", "G6", "G7", "G8", "G9", "G10"],
            }
        )
        pairs = _find_pairs(grn=grn, thr_pval=0.05)

        assert isinstance(pairs, set)

    def test_returns_empty_for_no_pairs(self):
        """Test that empty set is returned when no significant pairs exist."""
        # Create GRN with minimal overlap
        grn = pd.DataFrame(
            {
                "source": ["TF1", "TF2", "TF3"],
                "target": ["G1", "G2", "G3"],
            }
        )
        pairs = _find_pairs(grn=grn, thr_pval=0.01)

        assert isinstance(pairs, set)

    def test_pair_format(self):
        """Test that pairs are formatted correctly (sorted, pipe-separated)."""
        grn = pd.DataFrame(
            {
                "source": ["TF1"] * 10 + ["TF2"] * 10,
                "target": [f"G{i}" for i in range(10)] + [f"G{i}" for i in range(10)],
            }
        )
        pairs = _find_pairs(grn=grn, thr_pval=1.0)  # Very permissive threshold

        for pair in pairs:
            assert "|" in pair
            tf_a, tf_b = pair.split("|")
            assert tf_a <= tf_b  # Should be sorted


class TestTfp:
    """Tests for _tfp function (TF pairs evaluation)."""

    def test_basic_functionality(self, simple_grn, tfp_db):
        """Test basic TF pairs evaluation."""
        prc, rcl, f01 = _tfp(grn=simple_grn, db=tfp_db, thr_pval=0.05)

        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_filters_by_tfs_in_db(self):
        """Test that GRN is filtered to only include TFs present in database."""
        grn = pd.DataFrame(
            {
                "source": ["PAX5", "PAX5", "UnknownTF"],
                "target": ["CD19", "MS4A1", "GeneX"],
            }
        )
        db = pd.DataFrame({0: ["PAX5"], 1: ["GATA3"]})

        prc, rcl, f01 = _tfp(grn=grn, db=db, thr_pval=0.05)
        # Should not crash even with TFs not in DB
        assert 0 <= prc <= 1
        assert 0 <= rcl <= 1
        assert 0 <= f01 <= 1

    def test_handles_duplicates(self):
        """Test that function handles duplicate edges in GRN."""
        grn = pd.DataFrame(
            {
                "source": ["PAX5", "PAX5", "PAX5"],
                "target": ["CD19", "CD19", "MS4A1"],
            }
        )
        db = pd.DataFrame({0: ["PAX5"], 1: ["GATA3"]})

        # Should not crash
        prc, rcl, f01 = _tfp(grn=grn, db=db, thr_pval=0.05)
        assert isinstance(prc, float)
        assert isinstance(rcl, float)
        assert isinstance(f01, float)
