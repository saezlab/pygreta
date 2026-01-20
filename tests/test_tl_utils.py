"""Tests for gretapy.tl._utils module."""

import pytest

from gretapy.tl._utils import _f_beta_score, _prc_rcl_f01


class TestFBetaScore:
    """Tests for _f_beta_score function."""

    def test_perfect_scores(self):
        """Test with perfect precision and recall."""
        result = _f_beta_score(prc=1.0, rcl=1.0, beta=0.1)
        assert result == pytest.approx(1.0)

    def test_zero_scores(self):
        """Test with zero precision and recall."""
        result = _f_beta_score(prc=0.0, rcl=0.0, beta=0.1)
        assert result == 0.0

    def test_zero_precision(self):
        """Test with zero precision."""
        result = _f_beta_score(prc=0.0, rcl=0.5, beta=0.1)
        assert result == 0.0

    def test_zero_recall(self):
        """Test with zero recall."""
        result = _f_beta_score(prc=0.5, rcl=0.0, beta=0.1)
        assert result == 0.0

    def test_different_beta_values(self):
        """Test with different beta values."""
        prc, rcl = 0.8, 0.6
        # Lower beta weights precision more
        f01 = _f_beta_score(prc=prc, rcl=rcl, beta=0.1)
        f05 = _f_beta_score(prc=prc, rcl=rcl, beta=0.5)
        f1 = _f_beta_score(prc=prc, rcl=rcl, beta=1.0)

        # With high precision and lower recall, lower beta should give higher score
        assert f01 > f05 > f1

    def test_formula_correctness(self):
        """Test that the formula is correctly implemented."""
        prc, rcl, beta = 0.8, 0.6, 0.1
        expected = (1 + beta**2) * (prc * rcl) / ((prc * beta**2) + rcl)
        result = _f_beta_score(prc=prc, rcl=rcl, beta=beta)
        assert result == pytest.approx(expected)


class TestPrcRclF01:
    """Tests for _prc_rcl_f01 function."""

    def test_all_zeros(self):
        """Test with no true positives."""
        prc, rcl, f01 = _prc_rcl_f01(tps=0, fps=10, fns=5)
        assert prc == 0.0
        assert rcl == 0.0
        assert f01 == 0.0

    def test_perfect_classification(self):
        """Test with perfect classification (no FP or FN)."""
        prc, rcl, f01 = _prc_rcl_f01(tps=10, fps=0, fns=0)
        assert prc == 1.0
        assert rcl == 1.0
        assert f01 == pytest.approx(1.0)

    def test_precision_calculation(self):
        """Test precision calculation: TP / (TP + FP)."""
        prc, rcl, f01 = _prc_rcl_f01(tps=8, fps=2, fns=5)
        assert prc == pytest.approx(8 / 10)

    def test_recall_calculation(self):
        """Test recall calculation: TP / (TP + FN)."""
        prc, rcl, f01 = _prc_rcl_f01(tps=8, fps=2, fns=2)
        assert rcl == pytest.approx(8 / 10)

    def test_f01_uses_default_beta(self):
        """Test that F0.1 uses beta=0.1 by default."""
        tps, fps, fns = 8, 2, 2
        prc, rcl, f01 = _prc_rcl_f01(tps=tps, fps=fps, fns=fns)
        expected_f01 = _f_beta_score(prc=prc, rcl=rcl, beta=0.1)
        assert f01 == pytest.approx(expected_f01)

    def test_custom_beta(self):
        """Test with custom beta value."""
        tps, fps, fns = 8, 2, 2
        prc, rcl, f01 = _prc_rcl_f01(tps=tps, fps=fps, fns=fns, beta=0.5)
        expected_f01 = _f_beta_score(prc=prc, rcl=rcl, beta=0.5)
        assert f01 == pytest.approx(expected_f01)
