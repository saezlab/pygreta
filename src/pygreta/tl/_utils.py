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
