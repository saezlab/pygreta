import mudata as mu
import pandas as pd
import pytest

import gretapy as gp


class TestToyReturnTypes:
    """Test that toy() returns correct types."""

    def test_returns_tuple(self):
        result = gp.ds.toy()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_mudata(self):
        mdata, _ = gp.ds.toy()
        assert isinstance(mdata, mu.MuData)

    def test_returns_dataframe(self):
        _, grn = gp.ds.toy()
        assert isinstance(grn, pd.DataFrame)


class TestMuDataModalities:
    """Test MuData structure."""

    def test_has_rna_modality(self):
        mdata, _ = gp.ds.toy()
        assert "rna" in mdata.mod

    def test_has_atac_modality(self):
        mdata, _ = gp.ds.toy()
        assert "atac" in mdata.mod

    def test_rna_atac_same_observations(self):
        mdata, _ = gp.ds.toy()
        assert list(mdata.mod["rna"].obs_names) == list(mdata.mod["atac"].obs_names)

    def test_celltype_annotation_exists(self):
        mdata, _ = gp.ds.toy()
        assert "celltype" in mdata.obs.columns
        assert "celltype" in mdata.mod["rna"].obs.columns
        assert "celltype" in mdata.mod["atac"].obs.columns


class TestGRNColumns:
    """Test GRN DataFrame structure."""

    def test_has_source_column(self):
        _, grn = gp.ds.toy()
        assert "source" in grn.columns

    def test_has_target_column(self):
        _, grn = gp.ds.toy()
        assert "target" in grn.columns

    def test_has_cre_column(self):
        _, grn = gp.ds.toy()
        assert "cre" in grn.columns

    def test_has_score_column(self):
        _, grn = gp.ds.toy()
        assert "score" in grn.columns

    def test_cre_format(self):
        _, grn = gp.ds.toy()
        for cre in grn["cre"]:
            parts = cre.split("-")
            assert len(parts) == 3
            assert parts[0].startswith("chr")
            assert parts[1].isdigit()
            assert parts[2].isdigit()


class TestParameters:
    """Test parameter variations."""

    def test_n_cells(self):
        mdata, _ = gp.ds.toy(n_cells=30)
        assert mdata.n_obs == 30

    def test_n_tfs(self):
        _, grn = gp.ds.toy(n_tfs=2)
        assert grn["source"].nunique() == 2

    def test_n_targets_per_tf(self):
        _, grn = gp.ds.toy(n_tfs=1, n_targets_per_tf=3, n_peaks_per_target=1)
        assert len(grn) == 3

    def test_n_peaks_per_target(self):
        _, grn = gp.ds.toy(n_tfs=1, n_targets_per_tf=2, n_peaks_per_target=2)
        assert len(grn) == 4

    def test_custom_celltypes(self):
        custom_cts = ["TypeA", "TypeB"]
        mdata, _ = gp.ds.toy(n_cells=20, celltypes=custom_cts)
        assert set(mdata.obs["celltype"]) == set(custom_cts)

    def test_invalid_n_tfs(self):
        with pytest.raises(ValueError, match="n_tfs must be between 1 and 3"):
            gp.ds.toy(n_tfs=0)
        with pytest.raises(ValueError, match="n_tfs must be between 1 and 3"):
            gp.ds.toy(n_tfs=4)

    def test_invalid_n_targets(self):
        with pytest.raises(ValueError, match="n_targets_per_tf must be between 1 and 5"):
            gp.ds.toy(n_targets_per_tf=0)
        with pytest.raises(ValueError, match="n_targets_per_tf must be between 1 and 5"):
            gp.ds.toy(n_targets_per_tf=6)


class TestSeedReproducibility:
    """Test that seed parameter ensures reproducibility."""

    def test_same_seed_same_data(self):
        mdata1, grn1 = gp.ds.toy(seed=123)
        mdata2, grn2 = gp.ds.toy(seed=123)

        # Check RNA expression is identical
        assert (mdata1.mod["rna"].X == mdata2.mod["rna"].X).all()

        # Check ATAC accessibility is identical
        assert (mdata1.mod["atac"].X == mdata2.mod["atac"].X).all()

        # Check GRN scores are identical
        assert (grn1["score"] == grn2["score"]).all()

    def test_different_seed_different_data(self):
        mdata1, grn1 = gp.ds.toy(seed=42)
        mdata2, grn2 = gp.ds.toy(seed=99)

        # Expression should differ
        assert not (mdata1.mod["rna"].X == mdata2.mod["rna"].X).all()


class TestGRNEntitiesInMuData:
    """Test that GRN entities are present in MuData."""

    def test_tfs_in_rna_var(self):
        mdata, grn = gp.ds.toy()
        tfs = grn["source"].unique()
        rna_genes = set(mdata.mod["rna"].var_names)
        for tf in tfs:
            assert tf in rna_genes, f"TF {tf} not found in RNA var_names"

    def test_targets_in_rna_var(self):
        mdata, grn = gp.ds.toy()
        targets = grn["target"].unique()
        rna_genes = set(mdata.mod["rna"].var_names)
        for target in targets:
            assert target in rna_genes, f"Target {target} not found in RNA var_names"

    def test_cres_in_atac_var(self):
        mdata, grn = gp.ds.toy()
        cres = grn["cre"].unique()
        atac_peaks = set(mdata.mod["atac"].var_names)
        for cre in cres:
            assert cre in atac_peaks, f"CRE {cre} not found in ATAC var_names"


class TestCelltypeDistribution:
    """Test celltype distribution in generated data."""

    def test_default_celltypes(self):
        mdata, _ = gp.ds.toy(n_cells=60)
        expected = {"B cell", "T cell", "Monocyte"}
        assert set(mdata.obs["celltype"]) == expected

    def test_equal_distribution(self):
        mdata, _ = gp.ds.toy(n_cells=60)
        counts = mdata.obs["celltype"].value_counts()
        assert all(c == 20 for c in counts.values)
