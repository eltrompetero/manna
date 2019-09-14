# ====================================================================================== #
# Activated random walkers models.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from numba import njit
from numba import boolean, uint64


class ARW1D():
    def __init__(self, density, L,
                 rng=np.random,
                 density_as_particle_number=False):
        """
        Parameters
        ----------
        density : float
        L : int
            System lattice size.
        rng : np.random.RandomState
        density_as_particle_number : bool, False
            If True, take density parameter as the total number of particles instead. Must
            be an integer in this case.
        """
        
        assert 0<density
        assert L>1
        
        if density_as_particle_number:
            assert density%1==0, "If specifying particle number this must be an integer."
        self.density = density
        self.density_is_density = not density_as_particle_number
        self.L = L
        self.rng = rng

        self.initialize()

    def initialize(self):
        self.lattice = np .zeros(self.L, dtype=int)
        if self.density_is_density:
            self.lattice += self.rng.poisson(self.density, size=self.L)
        else:
            ix = self.rng.randint(self.L, size=self.density)
            for i in ix:
                self.lattice[i] += 1

    def _relax_conserved(self, max_iters=10_000):
        """Relax the system to an absorbing state synchronously while conserving particles.
        
        Parameters
        ----------
        max_iters : int, 10_000

        Returns
        -------
        int
            Lifetime of avalanche.
        """

        counter = 0
        cascadeSize = 0
        ix = self.lattice>=2
        cascadeSize += ix.sum()
        while ix.any() and counter<max_iters:
            self.lattice, ix = random_transfer_periodic_1d(self.lattice)
            #self.lattice[ix] -= 2
            #self.lattice[np.roll(ix,1)] += 1
            #self.lattice[np.roll(ix,-1)] += 1
            cascadeSize += ix.sum()

            #ix = self.lattice>=2
            counter += 1

        return counter, cascadeSize

    def _relax(self, tracer_ix=None, max_iters=10_000):
        """Relax the system to an absorbing state synchronously while losing particles at
        endpoints. 
        
        Keep track of single special tracer particle which is uniformly and randomly
        selected to be one of the two particles that moves from each lattice site. If it
        is chosen, it moves in either direction. If it leaves the lattice, it does not
        move.
        
        Parameters
        ----------
        tracer_ix : int, None
        max_iters : int, 10_000

        Returns
        -------
        int
            Lifetime of avalanche.
        int
            Size of cascade as the number of affected sites per turn summed.
        int 
            Location of tracer particle.
        """
        
        tracer_ix = tracer_ix or self.L//2

        counter = 0
        cascadeSize = 0  # should be related to the number of particles lost on the side
        ix = self.lattice>=2
        while ix.any() and counter<max_iters:
            #if 0<=tracer_ix<self.L and self.lattice[tracer_ix]>=2:
            #    # tracer is randomly selected
            #    if self.rng.rand()<(2/self.lattice[tracer_ix]):
            #        if self.rng.rand()<.5:
            #            tracer_ix -= 1
            #        else:
            #            tracer_ix += 1
            
            self.lattice, ix = random_transfer_1d(self.lattice, 2)
            # move one particle to the left and one to the right for each overloaded site
            #self.lattice[ix] -= 2
            cascadeSize += ix.sum()

            #shiftplus = np.roll(ix, 1)
            #shiftplus[0] = False
            #self.lattice[shiftplus] += 1

            #shiftminus = np.roll(ix, -1)
            #shiftminus[-1] = False
            #self.lattice[shiftminus] += 1

            #ix = self.lattice>=2
            counter += 1
        return counter, cascadeSize

    def relax(self, conserved=False, **kwargs):
        if conserved:
            return self._relax_conserved(**kwargs)
        return self._relax(**kwargs)
    
    def add(self, ix=None):
        """Add a particle to a spot on the lattice.

        Parameters
        ----------
        ix : int
        """

        ix = ix or self.rng.randint(self.L)
        self.lattice[ix] += 1

    def remove(self, ix=None):
        """Remove a particle from a site on the lattice.

        Parameters
        ----------
        ix : int, None
        """
        
        if ix:
            assert self.lattice[ix]
            self.lattice[ix] -= 1
            return

        ix = self.rng.choice(np.where(self.lattice)[0])
        self.lattice[ix] -= 1

    def sample(self, n_samples,
               max_iters=1_000_000,
               conserved=False):
        """Sample cascades. Likely, you will want to relax the system to a stable
        configuration first by calling relax().

        Parameters
        ----------
        n_samples : int
        max_iters : int, 1_000_000
        conserved : bool, False
            If True, conserve particle number.

        Returns
        -------
        ndarray
            Durations.
        ndarray
            Sizes measured by the number of active sites. The total number of particles
            that move is a constant multiplied by this.
        """

        t = np.zeros(n_samples, dtype=int)
        s = np.zeros(n_samples, dtype=int)

        i = 0
        while i<n_samples:
            self.add()
            if conserved:
                self.remove()
            
            t[i], s[i] = self.relax(conserved=conserved, max_iters=max_iters)
            # ignore moves that don't cause any cascade
            if t[i]==0:
                i -= 1
            i += 1

        return t, s

@njit
def random_transfer_1d(lattice, k):
    """Randomly transfer k particles from each site to adjacent sites when there are at
    least two particles per site.

    Parameters
    ----------
    lattice : ndarray
    k : int

    Returns
    -------
    ndarray
        Copy of new lattice.
    ndarray
        Index array of where there was toppling.
    """
    
    newlattice = np.zeros(lattice.size, dtype=uint64)
    ix = np.zeros(lattice.size, dtype=boolean)

    for i in range(lattice.size):
        if lattice[i]>=2:
            ix[i] = True
            newlattice[i] += lattice[i]-k
            for particleix in range(k):
                if np.random.rand()<.5:
                    if i>0:
                        newlattice[i-1] += 1
                elif i<(lattice.size-1):
                    newlattice[i+1] += 1
        else:
            newlattice[i] += lattice[i]
    return newlattice, ix

@njit
def random_transfer_periodic_1d(lattice):
    """Randomly transfer 2 particles from each site to adjacent sites when there are at
    least two particles per site.

    Parameters
    ----------
    lattice : ndarray

    Returns
    -------
    ndarray
        Copy of new lattice.
    ndarray
        Index array of where there was toppling.
    """
    
    newlattice = np.zeros(lattice.size, dtype=uint64)
    ix = np.zeros(lattice.size, dtype=boolean)

    for i in range(lattice.size):
        if lattice[i]>=2:
            ix[i] = True
            newlattice[i] += lattice[i]-2
            if np.random.rand()<.5:
                newlattice[(i-1)%lattice.size] += 1
            else:
                newlattice[(i+1)%lattice.size] += 1
            if np.random.rand()<.5:
                newlattice[(i-1)%lattice.size] += 1
            else:
                newlattice[(i+1)%lattice.size] += 1
        else:
            newlattice[i] += lattice[i]
    return newlattice, ix
#end ARW1D


class ARW2D(ARW1D):
    def __init__(self, density, L,
                 rng=np.random,
                 density_as_particle_number=False):
        """
        Parameters
        ----------
        density : float
        L : int
            System lattice length on each size.
        rng : np.random.RandomState
        density_as_particle_number : bool, False
            If True, take density parameter as the total number of particles instead. Must
            be an integer in this case.
        """
        
        assert 0<density
        assert L>1
        
        if density_as_particle_number:
            assert density%1==0, "If specifying particle number this must be an integer."
        self.density = density
        self.density_is_density = not density_as_particle_number
        self.L = L
        self.rng = rng

        self.initialize()

    def initialize(self):
        self.lattice = np.zeros((self.L,self.L), dtype=int)
        if self.density_is_density:
            self.lattice += self.rng.poisson(self.density, size=(self.L,self.L))
        else:
            ix = self.rng.randint(self.L, size=(self.density,2))
            for i in ix:
                self.lattice[i[0],i[1]] += 1

    def _relax_conserved(self, max_iters=10_000):
        """Relax the system to an absorbing state synchronously while conserving particles.
        
        Parameters
        ----------
        max_iters : int, 10_000

        Returns
        -------
        int
            Lifetime of avalanche.
        """

        counter = 0
        cascadeSize = 0
        ix = self.lattice>=2
        cascadeSize += ix.sum()
        while ix.any() and counter<max_iters:
            self.lattice, ix = random_transfer_periodic_2d(self.lattice, 2)
            cascadeSize += ix.sum()

            counter += 1

        return counter, cascadeSize

    def _relax(self, max_iters=10_000):
        """Relax the system to an absorbing state synchronously while losing particles at
        endpoints. 
        
        Keep track of single special tracer particle which is uniformly and randomly
        selected to be one of the two particles that moves from each lattice site. If it
        is chosen, it moves in either direction. If it leaves the lattice, it does not
        move.
        
        Parameters
        ----------
        max_iters : int, 10_000

        Returns
        -------
        int
            Lifetime of avalanche.
        int
            Size of cascade as the number of affected sites per turn summed.
        int 
            Location of tracer particle.
        """

        counter = 0
        cascadeSize = 0  # should be related to the number of particles lost on the side
        ix = self.lattice>=2
        while ix.any() and counter<max_iters:
            # move one particle to the left and one to the right for each overloaded site
            self.lattice, ix = random_transfer_2d(self.lattice, 2)
            cascadeSize += ix.sum()
            counter += 1

        return counter, cascadeSize

    def relax(self, conserved=False, **kwargs):
        if conserved:
            return self._relax_conserved(**kwargs)
        return self._relax(**kwargs)
    
    def add(self, ix=None):
        """Add a particle to a spot on the lattice.

        Parameters
        ----------
        ix : twople, None
        """

        ix = ix or self.rng.randint(self.L, size=2)
        self.lattice[ix[0],ix[1]] += 1

    def remove(self, ix=None):
        """Remove a particle from a site on the lattice.

        Parameters
        ----------
        ix : twople, None
        """
        
        if ix:
            assert self.lattice[ix[0],ix[1]]
            self.lattice[ix[0],ix[1]] -= 1
            return
        
        ix = np.where(self.lattice)
        ixofix = self.rng.randint(ix[0].size)
        self.lattice[ix[0][ixofix],ix[1][ixofix]] -= 1

@njit
def random_transfer_2d(lattice, k):
    """Randomly transfer k particles from each site to adjacent sites when there are at
    least two particles per site.

    Parameters
    ----------
    lattice : ndarray
    k : int

    Returns
    -------
    ndarray
        Copy of new lattice.
    ndarray
        Index array of where there was toppling.
    """
    
    newlattice = np.zeros(lattice.shape, dtype=uint64)
    ix = np.zeros(lattice.shape, dtype=boolean)
    L = newlattice.shape[0]

    for i in range(L):
        for j in range(L):
            if lattice[i,j]>=k:
                ix[i,j] = True
                newlattice[i,j] += lattice[i,j]-k
                for particleix in range(k):
                    if np.random.rand()<.5:
                        if i>0:
                            if np.random.rand()<.5:
                                if j>0:
                                    newlattice[i-1,j-1] += 1
                            elif (j+1)<L:
                                newlattice[i-1,j+1] += 1
                    elif i<(L-1):
                        if np.random.rand()<.5:
                            if j>0:
                                newlattice[i+1,j-1] += 1
                        elif j<(L-1):
                            newlattice[i+1,j+1] += 1
            else:
                newlattice[i,j] += lattice[i,j]
    return newlattice, ix

@njit
def random_transfer_periodic_2d(lattice, k):
    """Randomly transfer 2 particles from each site to adjacent sites when there are at
    least two particles per site.

    Parameters
    ----------
    lattice : ndarray
    k : int

    Returns
    -------
    ndarray
        Copy of new lattice.
    ndarray
        Index array of where there was toppling.
    """
    
    newlattice = np.zeros(lattice.shape, dtype=uint64)
    ix = np.zeros(lattice.shape, dtype=boolean)
    L = lattice.shape[0]

    for i in range(L):
        for j in range(L):
            if lattice[i,j]>=k:
                ix[i,j] = True
                newlattice[i,j] += lattice[i,j]-k
                for particleix in range(k):
                    if np.random.rand()<.5:
                        if np.random.rand()<.5:
                            newlattice[(i-1)%L,(j-1)%L] += 1
                        else:
                            newlattice[(i-1)%L,(j+1)%L] += 1
                    else:
                        if np.random.rand()<.5:
                            newlattice[(i+1)%L,(j-1)%L] += 1
                        else:
                            newlattice[(i+1)%L,(j+1)%L] += 1
            else:
                newlattice[i,j] += lattice[i,j]
    return newlattice, ix
#end ARW2D


class ARW1DNetwork():
    """Linked network of 1D ARW simulations. "Instigator" particles moves in between them.
    """
    def __init__(self, density, L, adj, rng=np.random):
        """
        Parameters
        ----------
        density : float
        L : int
        adj : ndarray
        rng : np.random.RandomSTate
        """

        self.density = density
        self.L = L
        self.adj = adj
        self.rng = rng

        self.initialize()

    def initialize(self):
        arw = []
        for i in range(len(self.adj)):
            arw.append(ARW1D(self.density, self.L, rng=self.rng))
            arw[-1].relax()
        self.arw = arw
        
    def instigate(self, arw_ix,
                  continuation_insert_threshold=10,
                  continuation_size_threshold=10,
                  max_iters=10_000):
        """Start a cascade by inserting an instigator into one of the ARW chains at random
        (at a point where there is at least one particle such that a cascade is guaranteed
        to start).

        Parameters
        ----------
        arw_ix : int
            Starting index of ARW.
        continuation_insert_threshold : int, 10
        continuation_size_threshold : int, 10
        max_iters : int, 10_000

        Returns
        -------
        list
            Duration of each cascade.
        list
            Indices of visited ARW chains.
        """
        
        dt = []
        visitedix = []
        cascadeSize = []
        
        # first iteration
        thisix = arw_ix
        insertpts = self.arw[thisix].lattice>=1
        insertix = self.rng.choice(np.where(insertpts)[0])

        while ((len(cascadeSize)==0 or cascadeSize[-1]>=continuation_size_threshold) and
               insertpts.sum()>continuation_insert_threshold and
               len(dt)<max_iters):
            self.arw[thisix].add(insertix)
            dt_, s, locix = self.arw[thisix].relax()
            dt.append(dt_)
            cascadeSize.append(s)
            
            # randomly move to another part of the network
            visitedix.append(thisix)
            thisix = self.rng.choice(np.where(self.adj[thisix])[0])
            insertpts = self.arw[thisix].lattice>=1
            insertix = self.rng.choice(np.where(insertpts)[0])
        
        assert all([t>0 for t in dt])
        return dt, cascadeSize, visitedix 
#end ARW1DNetwork
