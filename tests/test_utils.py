"""Tests for pygreta._utils module (shared utility functions)."""

import pandas as pd
import pytest

import pygreta as pg


class TestShowOrganisms:
    """Tests for show_organisms function."""

    def test_returns_list(self):
        """Test that show_organisms returns a list."""
        result = pg.show_organisms()
        assert isinstance(result, list)

    def test_contains_expected_organisms(self):
        """Test that result contains expected organisms."""
        result = pg.show_organisms()
        assert "hg38" in result
        assert "mm10" in result


class TestShowDatasets:
    """Tests for show_datasets function."""

    def test_returns_dataframe(self):
        """Test that show_datasets returns a DataFrame."""
        result = pg.show_datasets(organism="hg38")
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        result = pg.show_datasets(organism="hg38")
        expected_cols = {"name", "pubmed", "geo"}
        assert expected_cols == set(result.columns)

    def test_invalid_organism_raises(self):
        """Test that invalid organism raises assertion error."""
        with pytest.raises(AssertionError):
            pg.show_datasets(organism="invalid_organism")


class TestShowTerms:
    """Tests for show_terms function."""

    def test_returns_dataframe(self):
        """Test that show_terms returns a DataFrame."""
        result = pg.show_terms(organism="hg38")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_organism_raises(self):
        """Test that invalid organism raises assertion error."""
        with pytest.raises(AssertionError):
            pg.show_terms(organism="invalid_organism")


class TestShowMetrics:
    """Tests for show_metrics function."""

    def test_returns_dataframe(self):
        """Test that show_metrics returns a DataFrame."""
        result = pg.show_metrics()
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        result = pg.show_metrics()
        expected_cols = {"organism", "category", "metric", "db"}
        assert expected_cols == set(result.columns)

    def test_filter_by_organism_hg38(self):
        """Test filtering by hg38 organism."""
        result = pg.show_metrics(organism="hg38")
        assert isinstance(result, pd.DataFrame)
        # When filtered, organism column is dropped
        assert "organism" not in result.columns
        assert len(result) > 0

    def test_filter_by_organism_mm10(self):
        """Test filtering by mm10 organism."""
        result = pg.show_metrics(organism="mm10")
        assert isinstance(result, pd.DataFrame)
        assert "organism" not in result.columns
        assert len(result) > 0

    def test_invalid_organism_raises(self):
        """Test that invalid organism raises assertion error."""
        with pytest.raises(AssertionError):
            pg.show_metrics(organism="invalid_organism")

    def test_contains_expected_categories(self):
        """Test that result contains expected metric categories."""
        result = pg.show_metrics()
        categories = result["category"].unique()
        expected_categories = {"Prior Knowledge", "Genomic", "Predictive", "Mechanistic"}
        assert expected_categories.issubset(set(categories))
