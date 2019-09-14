# ====================================================================================== #
# Testing module for activated random walkers models.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
from .arw import *


def test_random_transfer_periodic_1d():
    np.random.seed(0)
    lattice = np.random.poisson(2, size=10)
    n = lattice.sum()
    lattice, _ = random_transfer_periodic_1d(lattice, 2)
    assert lattice.sum()==n
    print("Test passed: Number of particles conserved.")

def test_random_transfer_periodic_2d():
    np.random.seed(0)
    lattice = np.random.poisson(2, size=(10,10))
    n = lattice.sum()
    lattice, _ = random_transfer_periodic_2d(lattice, 2)
    assert lattice.sum()==n
    print("Test passed: Number of particles conserved.")

def test_ARW1D():
    arw = ARW1D(.9, 1000)
    n = arw.lattice.sum()
    arw.relax(conserved=True, max_iters=0)
    assert arw.lattice.sum()==n
    print("Test passed: Number of particles conserved.")
