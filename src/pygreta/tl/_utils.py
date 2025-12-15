def f_beta_score(
    prc: float,
    rcl: float,
    beta: float = 0.1,
):
    if prc + rcl == 0:
        return 0
    return (1 + beta**2) * (prc * rcl) / ((prc * beta**2) + rcl)
