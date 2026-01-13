import os
import shutil

import decoupler as dc
from decoupler._download import _download, _log

from pygreta.config import DATA, PATH_DATA, URL_END, URL_STR


def show_organisms() -> None:
    """
    Show all available organisms.

    Returns
    -------
    List of available organisms.

    Example
    -------
    .. code-block:: python

        import pygreta as pg

        pg.show_organisms(organism="hg38")
    """
    return list(DATA.keys())


def _download_data(
    organism: str,
    db_name: str,
    verbose: bool = False,
    retries: int = 5,
    wait_time: int = 20,
) -> str:
    os.makedirs(PATH_DATA, exist_ok=True)
    assert organism in DATA, f"organism={organism} not available:\n{DATA.keys()}"
    assert db_name in DATA[organism], f"db_name={db_name} not available as a database:\n{DATA[organism]}"
    fname = DATA[organism][db_name]["fname"]
    path_fname = os.path.join(PATH_DATA, fname)
    if not os.path.isfile(path_fname):
        if fname != "hg38_prt_knocktf.h5ad":
            url = URL_STR + fname + URL_END
            data = _download(url, verbose=verbose)
            data.seek(0)  # Move pointer to beginning
            with open(path_fname, "wb") as f:
                shutil.copyfileobj(data, f)
            m = f"Database {db_name} saved in {path_fname}"
            _log(m, level="info", verbose=verbose)
        else:
            adata = dc.ds.knocktf(thr_fc=100_000, verbose=verbose)  # Do not filter here
            adata.write(path_fname)
    else:
        m = f"Database {db_name} found in {path_fname}"
        _log(m, level="info", verbose=verbose)
    return path_fname
