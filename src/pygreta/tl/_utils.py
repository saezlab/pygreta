import pandas as pd

from pygreta.config import DATA, METRIC_CATS
from pygreta.ds._utils import show_organisms


def show_metrics(organism: str | None = None) -> None:
    """
    Show all available metrics.

    Parameters
    ----------
    organism
        Which organism to use.

    Returns
    -------
    Dataframe listing all available metrics per organism.

    Example
    -------
    .. code-block:: python

        import pygreta as pg

        pg.show_metrics(organism="hg38")
    """
    assert isinstance(organism, str) or organism is None
    df = []
    for org in DATA:
        org_dbs = DATA[org]["dbs"]
        for db in org_dbs:
            metric = org_dbs[db]["metric"]
            metric_cat = METRIC_CATS[metric]
            df.append([org, metric_cat, metric, db])
    cols = ["organism", "category", "metric", "db"]
    df = pd.DataFrame(df, columns=cols)
    df = df.sort_values(cols)
    if organism:
        organisms = show_organisms()
        assert organism in organisms, f"organism={organism} not available ({organisms})"
        df = df[df["organism"] == organism].drop(columns="organism")
    else:
        df = df.sort_values(cols)
    df = df.reset_index(drop=True)
    return df


def _f_beta_score(
    prc: float,
    rcl: float,
    beta: float = 0.1,
):
    if prc + rcl == 0:
        return 0
    return (1 + beta**2) * (prc * rcl) / ((prc * beta**2) + rcl)


def _prc_rcl_f01(tps: float, fps: float, fns: float, beta: float = 0.1):
    if tps > 0:
        prc = tps / (tps + fps)
        rcl = tps / (tps + fns)
        f01 = _f_beta_score(prc, rcl, beta=beta)
    else:
        prc, rcl, f01 = (
            0.0,
            0.0,
            0.0,
        )
    return prc, rcl, f01
