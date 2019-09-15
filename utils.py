# ====================================================================================== #
# Utilities for manna module.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from misc.utils import hist_log


def log_binned_p(x, bins):
    """Integer x-axis histogram.

    Parameters
    ----------
    x : ndarray
    bins : int

    Returns
    -------
    ndarray
        Normalized probability distribution.
    ndarray
        Centers of bins.
    ndarray
        Edges of bins.
    """
    
    assert (x>0).all()
    pt, xt, xbins = hist_log(x, bins)
    nint = np.diff(xbins)  # number of integers in each interval
    nint[np.mod(xbins[:-1],1)==0] += 1
    pt = pt / nint
    pt = pt / pt.sum()
    
    return pt, xt, xbins
