import os

PATH_DATA = os.path.join(".", "pygreta_data")
ID_ZENODO = 17872739
URL_STR = f"https://zenodo.org/records/{ID_ZENODO}/files/"
URL_END = "?download=1"

DATA = {
    "hg38": {
        # Prior knowledge
        "Human Protein Atlas (HPA)": {
            "fname": "hg38_tfm_hpa.tsv.gz",
            "metric": "TF markers",
        },
        "TF-Marker": {
            "fname": "hg38_tfm_tfmdb.tsv.gz",
            "metric": "TF markers",
        },
        "Europe PMC": {
            "fname": "hg38_tfp_europmc.tsv.gz",
            "metric": "TF pairs",
        },
        "IntAct": {
            "fname": "hg38_tfp_intact.tsv.gz",
            "metric": "TF pairs",
        },
        "CollecTRI": {
            "fname": "hg38_gst_collectri.csv.gz",
            "metric": "Reference GRN",
        },
        # Genomic
        "ChIP-Atlas": {
            "fname": "hg38_tfb_chipatlas.bed.gz",
            "metric": "TF binding",
        },
        "ReMap 2022": {
            "fname": "hg38_tfb_remap2022.bed.gz",
            "metric": "TF binding",
        },
        "UniBind": {
            "fname": "hg38_tfb_unibind.bed.gz",
            "metric": "TF binding",
        },
        "ENCODE Blacklist": {
            "fname": "hg38_cre_blacklist.bed.gz",
            "metric": "CREs",
        },
        "ENCODE CREs": {
            "fname": "hg38_cre_encode.bed.gz",
            "metric": "CREs",
        },
        "GWAS Catalog": {
            "fname": "hg38_cre_gwascatalogue.bed.gz",
            "metric": "CREs",
        },
        "phastCons": {
            "fname": "hg38_cre_phastcons.bed.gz",
            "metric": "CREs",
        },
        "Promoters": {
            "fname": "hg38_cre_promoters.bed.gz",
            "metric": "CREs",
        },
        "Zhang21": {
            "fname": "hg38_cre_zhang21.bed.gz",
            "metric": "CREs",
        },
        "eQTL Catalogue": {
            "fname": "hg38_c2g_eqtlcatalogue.bed.gz",
            "metric": "CRE to gene links",
        },
        # Predictive
        "Hallmarks": {
            "fname": "hg38_gst_hall.csv.gz",
            "metric": "Gene sets",
        },
        "KEGG": {
            "fname": "hg38_gst_kegg.csv.gz",
            "metric": "Gene sets",
        },
        "Reactome": {
            "fname": "hg38_gst_reac.csv.gz",
            "metric": "Gene sets",
        },
        "PROGENy": {
            "fname": "hg38_gst_prog.csv.gz",
            "metric": "Gene sets",
        },
        "gene ~ TFs": {
            "fname": None,
            "metric": "Omics",
        },
        "gene ~ CREs": {
            "fname": None,
            "metric": "Omics",
        },
        "CRE ~ TFs": {
            "fname": None,
            "metric": "Omics",
        },
        # Mechanistic
        "KnockTF (scoring)": {
            "fname": "hg38_prt_knocktf.h5ad",
            "metric": "TF scoring",
        },
        "KnockTF (forecasting)": {
            "fname": "hg38_prt_knocktf.h5ad",
            "metric": "Perturbation forecasting",
        },
        "Boolean rules": {
            "fname": None,
            "metric": "Steady state simulation",
        },
    },
    "mm10": {
        # Prior knowledge
        "CollecTRI": {
            "fname": "mm10_gst_collectri.csv.gz",
            "metric": "Reference GRN",
        },
        # Genomic
        "ChIP-Atlas": {
            "fname": "mm10_tfb_chipatlas.bed.gz",
            "metric": "TF binding",
        },
        "ReMap 2022": {
            "fname": "mm10_tfb_remap2022.bed.gz",
            "metric": "TF binding",
        },
        "UniBind": {
            "fname": "mm10_tfb_unibind.bed.gz",
            "metric": "TF binding",
        },
        "ENCODE Blacklist": {
            "fname": "hg38_cre_blacklist.bed.gz",
            "metric": "CREs",
        },
        "ENCODE CREs": {
            "fname": "hg38_cre_encode.bed.gz",
            "metric": "CREs",
        },
        "phastCons": {
            "fname": "hg38_cre_phastcons.bed.gz",
            "metric": "CREs",
        },
        "Promoters": {
            "fname": "hg38_cre_promoters.bed.gz",
            "metric": "CREs",
        },
        # Predictive
        "Hallmarks": {
            "fname": "mm10_gst_hall.csv.gz",
            "metric": "Gene sets",
        },
        "Reactome": {
            "fname": "mm10_gst_reac.csv.gz",
            "metric": "Gene sets",
        },
        "PROGENy": {
            "fname": "mm10_gst_prog.csv.gz",
            "metric": "Gene sets",
        },
        "gene ~ TFs": {
            "fname": None,
            "metric": "Omics",
        },
        "gene ~ CREs": {
            "fname": None,
            "metric": "Omics",
        },
        "CRE ~ TFs": {
            "fname": None,
            "metric": "Omics",
        },
        # Mechanistic
        "KnockTF (scoring)": {
            "fname": "m10_prt_knocktf.h5ad.gz",
            "metric": "TF scoring",
        },
        "KnockTF (forecasting)": {
            "fname": "m10_prt_knocktf.h5ad.gz",
            "metric": "Perturbation forecasting",
        },
        "Boolean rules": {
            "fname": None,
            "metric": "Steady state simulation",
        },
    },
}

METRIC_CATS = {
    "TF markers": "Prior Knowledge",
    "TF pairs": "Prior Knowledge",
    "Reference GRN": "Prior Knowledge",
    "TF binding": "Genomic",
    "CREs": "Genomic",
    "CRE to gene links": "Genomic",
    "Gene sets": "Predictive",
    "Omics": "Predictive",
    "TF scoring": "Mechanistic",
    "Perturbation forecasting": "Mechanistic",
    "Steady state simulation": "Mechanistic",
}
