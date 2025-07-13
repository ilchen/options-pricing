import numpy as np
from scipy.interpolate import CubicSpline


# class ParYieldConverter:
#     """
#     Converter of par yields to spot rates (aka zero rates) via bootstrapping.
#     """

PAR_VALUE = 100.

#@staticmethod
def par_yields_to_spot(par_yields, maturities, coupon_frequency):
    """
    Converts par yields to spot rates using cubic spline interpolation and bootstrapping.

    Parameters:
    - par_yields (list or array): Par yields (e.g. 0.024 for 2.4%) at corresponding maturities, compounding
                                  assumed to correspond to the coupon_frequency
    - maturities (list or array): Maturities in years
    - coupon_frequency (int): Number of coupon payments per year (e.g. 2 for semiannual)

    Returns:
    - dict: Mapping of maturity (years) to annualized spot rate (as decimal)
    """
    par_yields = np.asarray(par_yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if len(par_yields) != len(maturities):
        raise ValueError("par_yields and maturities must have the same length")

    if coupon_frequency == 0:
        return dict(zip(maturities, par_yields))

    if len(par_yields) == 0:
        raise ValueError("par_yields must be a non-empty sequence")

    cashflow_interval = 1 / coupon_frequency

    if not np.all(maturities[:-1] <= maturities[1:]):
        raise ValueError("maturities must be in ascending order")

    if not np.all((maturities <= cashflow_interval) | (maturities % cashflow_interval == 0)):
        raise ValueError("Each maturity must align with coupon payment frequency")

    # Build full schedule based on cash flow intervals
    all_maturities = np.arange(cashflow_interval, maturities[-1] + cashflow_interval, cashflow_interval)

    # Interpolate par yields using cubic splines
    spline = CubicSpline(maturities, par_yields)
    interpolated_par_yields = spline(all_maturities)

    # Bootstrapping
    spot_rates = [interpolated_par_yields[0]]

    for i in range(1, len(all_maturities)):
        coupon = interpolated_par_yields[i] / coupon_frequency * PAR_VALUE
        pv_coupons = 0.0

        for j in range(i):
            r = spot_rates[j]
            discount = (1 + r / coupon_frequency) ** (j + 1)
            pv_coupons += coupon / discount

        final_cf = coupon + PAR_VALUE
        remaining_value = PAR_VALUE - pv_coupons
        spot_period = (final_cf / remaining_value) ** (1 / (i + 1)) - 1
        spot_rates.append(spot_period * coupon_frequency)

    output = dict(zip(all_maturities, spot_rates))

    # Patch short-term maturities (zero-coupon case)
    output.update({
        m: y for m, y in zip(maturities, par_yields)
        if m < cashflow_interval
    })

    return output

