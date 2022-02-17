import numpy as np
from ase import Atoms
from typing import List
from numpy.random import Generator


_meV_to_kBK = 11.604518
_kBK_to_meV = 1.0 / _meV_to_kBK


class Particle:
    """ Stores a classical particle."""
    def __init__(self, state: str, ekin: float, epot: float):
        """
        Initializes a Particle.

        Args:
            state: str
            The label for the state or pore the particle is in.

            ekin: float, unit=milli-electronvolt
            The initial kinetic energy.

            epot: float, unit=milli-electronvolt
            The initial potential energy.
        """
        self._epot = epot
        self._ekin = ekin
        self._state = state
        self._history = []

    def __repr__(self):
        return "Particle( {0:7.5f}, {1:7.5f} )".format(self.ekin, self.epot)

    @property
    def epot(self) -> float:
        """Get the potential energy of the particle."""
        return self._epot

    @epot.setter
    def epot(self, new_epot: float):
        self._epot = new_epot

    @property
    def ekin(self) -> float:  # unit: milli-electronvolt
        """Get the kinetic energy of the particle."""
        return self._ekin

    @ekin.setter
    def ekin(self, new_ekin: float):
        self._ekin = new_ekin

    @property
    def etot(self) -> float:  # unit: milli-electronvolt
        """Get the total energy of the particle."""
        return self.ekin + self.epot

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, new_state: str):
        self._state = new_state

    @property
    def history(self):
        return self._history


class PoreHoppingModel:
    r""" Stores a pore hopping model.

    Such a model comprises the potential energy at the center of each pore with
    one unit cell of a porous crystal and the barriers between pairs of
    connected pores.

    What is simulated is a random walk of one or more particles within the pore
    network. The propagation is done using the Metropolis algorithm, where a
    step is accepted if the total energy of the (classical) particle is larger
    than the barrier between two connected pores or if the Metropolis criterion
    is fulfilled with the acceptance probability :math:`P(f\leftarrow i)`,
        .. math::
            P(f\leftarrow i) = \exp[
                                  -\frac{\Delta E_{f\leftarrow i}}
                                        {k_\mathrm{B} T}
                                   ]\,,
    where :math:`\Delta E_{f\leftarrow i}` is the energy barrier between the
    pore :math:`i` and the pore :math:`f`, and :math:`T` is the temperature of
    the system. :math:`k_\mathrm{B}` is the Boltzmann constant.

    Attributes:

    """
    def __init__(self, pore_potentials: dict, pore_barriers: dict, temperature: float,
                 initial_pores: List[str]):
        """Initialises a PoreHoppingModel.

        Args:
            pore_potentials: dict, unit=milli-electronvolt
            Specifies the potential energy at the center of each pore. Keys are
            the pore labels, values are the potential energies.

            pore_barriers: dict, unit=milli-electronvolt
            Specifies the potential energy barriers for each pair of connected
            pores. Keys must have the format "{i}>{f}", where {i} is the label
            of the initial pore and {f} the label of the final pore.

            temperature: float, unit=kelvin
            Specifies the temperature of the system. Specifies the initial
            kinetic energy of each particle in the model according to the
            equipartition theorem, i. e. :math:`3 k_\mathrm{B} T / 2`.

            initial_pores: List[str]
            Specifies, using the pore labels, the initial potential energy of
            each particle in the pore network.

        """
        self._pore_potentials = pore_potentials
        self._pore_barriers = pore_barriers
        self._temperature = temperature

        self._connectivity = {label: [] for label in iter(self._pore_potentials.keys())}
        for key in iter(self._pore_barriers.keys()):
            _ini, _fin = key.split(">")
            self._connectivity[_ini].append(_fin)
            self._connectivity[_fin].append(_ini)
        for key, value in iter(self._connectivity.items()):
            self._connectivity[key] = list(set(value))

        # Initial kinetic energy for each particle according to the classical
        # equipartition theorem.
        _ekin0 = 1.5 * temperature * _kBK_to_meV  # unit: milli-electronvolt
        self._particles = [Particle(label, _ekin0, self._pore_potentials[label])
                           for label in initial_pores
                           ]

    @property
    def pore_potentials(self) -> dict:  # unit: milli-electronvolt
        """Get the potential energy at the center of each pore."""
        return self._pore_potentials

    @property
    def pore_barriers(self) -> dict:  # unit: milli-electronvolt
        """Get the potential energy barriers between connected pore pairs."""
        return self._pore_barriers

    @property
    def connectivity(self) -> dict:
        return self._connectivity

    @property
    def temperature(self) -> float:  # unit: kelvin
        """Get the temperature of the system."""
        return self._temperature

    @property
    def particles(self):
        return self._particles

    @property
    def n_particles(self):
        return len(self._particles)

    def _step_propagate(self, rng: Generator):
        """
        Propagate by one step.

        Args:
            rng: Generator
            A random number generator.

        """
        for i_particle in range(self.n_particles):
            _particle = self._particles[i_particle]
            _accessible = self.connectivity[_particle.state]
            #_accessible.append(_particle.state)
            _accessible_items = {key: self.pore_potentials[key] for key in _accessible}
            _accessible_items = tuple(map(list, _accessible_items.items()))
            _random_items = rng.choice(_accessible_items, (self.n_particles,))
            _prop_pores = [label for label in _random_items[:, 0]]
            _prop_epots = [float(epot) for epot in _random_items[:, 1]]
            _prop_pore = _prop_pores[i_particle]
            _prop_epot = _prop_epots[i_particle]
            _hop = "{}>{}".format(_particle.state, _prop_pore)
            _barrier = self._pore_barriers[_hop]
            _accept = (_particle.etot > _barrier or
                       np.exp(- _meV_to_kBK * _barrier / self.temperature) > rng.random())
            if _accept:
                _particle.ekin += _particle.epot - _prop_epot
                _particle.state = _prop_pore
            # Add the current state to the particle history.
            _particle.history.append(_particle.state)


if __name__ == "__main__":
    from numpy.random import default_rng
    from band_transport.utilities.progress import AdvancedProgressBar as ProgressBar

    epot_large = 0.0
    epot_small = 15.0
    barrier_large_small = 45.0
    barrier_small_large = 30.0

    temperature = 1000.0
    initial_pores = ["1"]

    pore_potentials = {"1": epot_large,
                       "2": epot_large,
                       "3a": epot_small,
                       "3b": epot_small,
                       "3c": epot_small,
                       "3d": epot_small
                       }

    pore_barriers = {"1>1": 0.0,
                     "2>2": 0.0,
                     "3a>3a": 0.0,
                     "3b>3b": 0.0,
                     "3c>3c": 0.0,
                     "3d>3d": 0.0,
                     "1>3a": barrier_large_small,
                     "1>3b": barrier_large_small,
                     "1>3c": barrier_large_small,
                     "1>3d": barrier_large_small,
                     "2>3a": barrier_large_small,
                     "2>3b": barrier_large_small,
                     "2>3c": barrier_large_small,
                     "2>3d": barrier_large_small,
                     "3a>1": barrier_small_large,
                     "3a>2": barrier_small_large,
                     "3b>1": barrier_small_large,
                     "3b>2": barrier_small_large,
                     "3c>1": barrier_small_large,
                     "3c>2": barrier_small_large,
                     "3d>1": barrier_small_large,
                     "3d>2": barrier_small_large
                     }

    model = PoreHoppingModel(pore_potentials, pore_barriers, temperature, initial_pores)
    rng = default_rng()

    n_trials = 100000
    progress = ProgressBar("Sampling ", n_trials)
    for i_step in range(n_trials):
        progress()
        model._step_propagate_(rng)

    hist = {label: (np.array(model.particles[0].history) == label).sum() / n_trials
            for label in iter(model.pore_potentials.keys())}
    print(hist)

    import matplotlib.pyplot as plt
    plt.bar(hist.keys(), hist.values())
    plt.show()