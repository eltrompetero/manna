# ====================================================================================== #
# Testing module for activated random walkers models.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
from .arw import *


def test_random_transfer_periodic_1d():
    lattice = np.zeros(10)
    lattice[0] = 2
    lattice[-1] = 2
    n = lattice.sum()
    lattice, _ = random_transfer_periodic_1d(lattice, 2)
    assert lattice.sum()==n
    print("Test passed: Number of particles conserved.")

def test_random_transfer_periodic_2d():
    lattice = np.zeros((10,10))
    lattice[0,0] = 2
    lattice[9,0] = 2
    lattice[0,9] = 2
    lattice[9,9] = 2
    n = lattice.sum()
    lattice, _ = random_transfer_periodic_2d(lattice, 2, 0)
    assert lattice.sum()==n
    print("Test passed: Number of particles conserved.")

def test_ARW1D():
    arw = ARW1D(.9, 1000)
    n = arw.lattice.sum()
    arw.relax(conserved=True, max_iters=0)
    assert arw.lattice.sum()==n
    print("Test passed: Number of particles conserved.")
