# ====================================================================================== #
# Utilities for manna module.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from misc.utils import hist_log


def log_binned_p(x, bins):
    """
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

    pt, xt, xbins = hist_log(x, bins)
    pt = pt/np.diff(xbins)
    pt = pt/pt.sum()
    
    return pt, xt, xbins
