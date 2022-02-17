import datetime
import os
from copy import deepcopy
from typing import Tuple, Optional

import itertools
import numpy as np
import sys
from tqdm import tqdm
from ase import Atoms
from ase.data import covalent_radii, vdw_alvarez, atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list
from numpy.random import Generator, default_rng

from mercury_sampler.maths import random_quaternions, rotate_and_shift


def _timestamp():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))


def _line_format(length, format_str=" 7.5f", separator="\t"):
    line = ""
    for i in range(length - 1):
        line += "{" + str(i) + ":" + format_str + "}" + separator
    line += "{" + str(length - 1) + ":" + format_str + "}" + "\n"
    return line


def _radius(sym: str, 
            kind: str = 'vdw', 
            override: float = 0.0) -> float:
    """
    Returns an atomic radius

    Args:
        sym: str
        The element symbol.

        kind: str, allowed=['vdw', 'cov']
        The kind of radius to be used, van der Waals ('vdw')
        or covalent ('cov').

        override: float, default=0.0
        Setting this to a float value greater 0.0 overrides the rest of the 
        function and simply returns this value.

    Returns:
        rad: float
    """
    if override > 0.0:
        rad = override
        return rad
    if kind == 'cov':
        rad = covalent_radii[atomic_numbers[sym]]
    elif kind == 'vdw':
        rad = vdw_alvarez.vdw_radii[atomic_numbers[sym]]
    elif isinstance(kind, list):
        rad = kind[atomic_numbers[sym]]
    else:
        raise ValueError("`kind` must have values 'cov' or 'vdw' or be a list. "
                         "Found {}.".format(kind))
    return rad


def _check_position(guest_atoms: Atoms,
                    host_atoms: Atoms,
                    kind: str = "vdw",
                    override: float = 0.0,
                    counter: Optional[int] = None,
                    debug: bool = False,
                    print_accessible: bool = False):
    """Checks if the any atomic sphere from `guest_atoms` touches or penetrates
    any of the atomic spheres from `host_atoms`.

    Args:
        guest_atoms: Atoms
        The guest structure.

        host_atoms: Atoms
        The porous host structure.

        kind: str, default="vdw"
        Which kind of atomic radius is to be used. Per default, this is the
        van der Waals radius.

        override: float, default=0.0
        When set to a float value larger than zero for a single-atom guest,
        this value is set as the guest/probe radius.

        counter: int, optional
        The number of the tested structure. Only for printing purposes.

        debug: bool, default=False
        Print debug messages.

        print_accessible: bool, default=False
        Print a message if an accessible position and orientation has been
        found. Also print a message every 1000 trials.

    Returns:
        is_inside: bool

    Author:
        Christoph Muschielok (c.muschielok@tum.de)

    """
    _host_guest_atoms = deepcopy(host_atoms)
    _host_guest_atoms.extend(guest_atoms)
    _symbols = _host_guest_atoms.symbols
    pairs = [p for
             p in itertools.combinations(_symbols, 2)
             ]
    pairs = [p
             for s in guest_atoms.symbols
             for p in pairs
             if s in p
             ]
    if override > 0.0 and len(guest_atoms) == 1:
        cutoffs = []
        for p, q in pairs:
            _r_cut = _radius(p, kind=kind, override=override) \
                     + _radius(q, kind=kind, override=override)
            cutoffs.append(_r_cut)
    else:
        cutoffs = [_radius(p, kind=kind) + _radius(q, kind=kind)
                   for p, q in pairs
                   ]
    cutoff_dict = {p: c
                   for p, c in zip(pairs, cutoffs)
                   }
    i_neigh_list, j_neigh_list = neighbor_list("ij", _host_guest_atoms, cutoff_dict)
    neighbors = [(i, j)
                 for i, j in zip(i_neigh_list, j_neigh_list)
                 if i >= len(host_atoms) > j
                 ]
    if debug:
        print("DEBUG: Printing cutoff radii.")
        print(cutoff_dict)
        print("DEBUG: Printing neighbors.")
        print(neighbors)
    is_inside = False
    if neighbors:
        pass
    else:
        is_inside = True
    if print_accessible:
        if is_inside or counter % 1000 == 0:
            print("|  {}, Structure {:6d}, {}".format(_timestamp(), counter, is_inside))
    return is_inside


class Sampler:
    r"""Samples a host unit cell for pores into which a guest structure fits.

    The functionality is currently restricted to atomic guests and linear
    guests structures. If a position is accessible is determined by an overlap
    criterion for hard spheres using the van der Waals radii.
    For an atomic guest the proposed position in fractional coordinates of the
    cell vectors is acquired by drawing from a uniform distribution
    :math:`[0, 1)^3`. For a linear guest, the same is done to get a center of
    mass of the molecule. Then, the molecule is rotated randomly. After that,
    the overlap criterion is applied and the center of mass and orientation is
    accepted or rejected.

    The random rotations are facilitated using random quaternions. This ensures
    that the rotations lead to a uniform distribution of a point on the unit
    sphere. The trick is to draw random 4-vectors from a normal distribution
    which ensures that there is no angular preference. These vectors are
    normalized and used as random quaternions :math:`q=(q_0, q_1, q_2, q_3)`.
    A vector which shall be rotated, e. g. the molecular axis of a linear guest,
    can be then interpreted as a pure quaternion, i. e. one with
    :math:`q_0 = 0`, and rotated by :math:`q \cdot (0, P) \cdot q^{-1}`.

    Author:
        Christoph Muschielok (c.muschielok@tum.de)

    """
    def __init__(self, host_atoms: Atoms, guest_atoms: Atoms):
        """Initialize a Sampler instance.

        Args:
            host_atoms: Atoms
            The host structure.

            guest_atoms: Atoms
            The guest structure.
        """
        self._host_atoms = host_atoms
        self._guest_atoms = guest_atoms

    def __call__(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the host structure with the guest structure.

        This is just a wrapper for the `sample` method. See its documentation
        for details.

        Args:
            **kwargs: Additional keyword arguments
            Anything that goes for the `sample` method.

        Returns:
            accessible: (N, 3) or (N, 7) ndarray

            inaccessible: (M, 3) or (M, 7) ndarray

        """
        accessible, inaccessible = self.sample(**kwargs)
        return accessible, inaccessible

    def sample(self,
               acc_filename: str = "accessible.dat",
               rej_filename: str = "inaccessible.dat",
               sample_size: int = 1000,
               pos_sample_size: Optional[int] = None,
               quat_sample_size: Optional[int] = None,
               kind: str = "vdw",
               override: float = 0.0,
               rng: Optional[Generator] = None,
               rng_params: Optional[dict] = None,
               positions: Optional[np.ndarray] = None,
               quaternions: Optional[np.ndarray] = None,
               guest_atoms_filename: str = "guest-atoms.json",
               ignore_guest_atoms_file: bool = False,
               debug: bool = False,
               print_accessible: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the unit cell of the host structure.

        Args:
            acc_filename: str, default="accessible.dat"
            The name of the file to which the accepted positions (and rotations)
            are saved.

            rej_filename: str, default="inaccessible.dat"
            The name of the file to which the rejected positions (and rotations)
            are saved.

            sample_size: int, default=1000
            The number of random positions (and rotations) which are drawn. This
            value is used instead of `pos_sample_size` or `quat_sample_size` if
            one or both of these parameters are None.

            pos_sample_size: int, optional
            The number of random positions to be drawn. If this value is given
            explicitly, the value of `sample size` is not used for this purpose.

            quat_sample_size: int, optional
            The number of random rotations, i. e. quaternions, to be drawn. If
            this value if given explicitly, the value of `sample_size` is not
            used for this purpose.

            kind: str, default="vdw"
            The kind of atomic radii to be used. Per default, these are van der
            Waals radii from reference [1].

            override: float, default=0.0
            When set to a float value greater 0.0 for a single-atom guest, this
            as the guest/probe radius instead of, for example, its 
            van der Waals radius.

            rng: Generator, optional
            The random number generator to be used. This defaults to the
            standard RNG of numpy. See `numpy.random.Generator` for details.

            positions: (N, 3) ndarray, optional
            The array of positions, usually from a pre-sampling with a small
            radius probe body, e. g. a hydrogen atom, to be used instead of
            random sampling. The usage of this greatly increases the yield of
            accessible positions for larger guest structures.

            quaternions: (M, 3) ndarray, optional
            The array of quaternions to be used instead of purely random
            sampling.

            guest_atoms_filename: str, default="guest-atoms.in"
            The name of the file to which the guest structure used for sampling
            is saved to.

            ignore_guest_atoms_file: bool, default=False
            Ignore an existing guest structure file with the same requested
            name as stored by `guest_atoms_filename`, thereby possibly making
            a mistake.

            debug: bool, default=False
            Print additional information during the sampling process.

            print_accessible: bool, default=False
            If a new accessible guest position and orientation is found, print
            it to stdout.

        Returns:
            accessible: (P, 3) or (P, 7) ndarray
            The array of accessible shifts and rotations for the host-guest
            pair.

            inaccessible: (Q, 3) or (Q, 7) ndarray
            The array of inaccessible shifts and rotations for the host-guest
            pair.

        References:
            [1] Alvarez, S.; Dalton Trans. 2013, 42, 8617-8636.

        """
        _guest_atoms_isfile = os.path.isfile(os.path.join(os.getcwd(),
                                                          guest_atoms_filename))
        if not ignore_guest_atoms_file and _guest_atoms_isfile:
            _msg = "Found guest atoms file `{}` in the working directory. Testing for " \
                   "differences."
            print(_msg.format(guest_atoms_filename))
            try:
                _test_atoms = read(guest_atoms_filename)
                if _test_atoms != self.guest_atoms:
                    _msg = "The structure in `{}` differs from the structure in the " \
                           "given `guest_atoms` instance (self.guest_atoms)."
                    raise ValueError(_msg.format(guest_atoms_filename))
            except ValueError:
                raise
        elif not ignore_guest_atoms_file and not _guest_atoms_isfile:
            _msg = "Writing guest structure to file `{}`"
            print(_msg.format(guest_atoms_filename))
            self.guest_atoms.write(guest_atoms_filename)
        if override > 0.0:
            print(f"Overriding atomic radius: {override: 7.3f} angstrom")
        if rng_params is None:
            rng_params = dict()
        if rng is None:
            rng = default_rng(**rng_params)
        if pos_sample_size is None:
            pos_sample_size = sample_size
        if quat_sample_size is None:
            quat_sample_size = sample_size
        # Get the actual sample size.
        if len(self._guest_atoms) > 1:
            sample_size = pos_sample_size * quat_sample_size
        else:
            sample_size = pos_sample_size
        if positions is None:
            print("Generating random positions.")
            positions = rng.random((pos_sample_size, 3))
            positions = self.host_atoms.cell.cartesian_positions(positions)
        if quaternions is None:
            print("Generating random quaternions.")
            quaternions = random_quaternions(quat_sample_size, rng=rng)
        elif quaternions is not None and len(self.guest_atoms) > 1:
            print("Using given quaternions.")
        # Set up the translations and rotations.
        print("[{}] Setting up test structures.".format(_timestamp()))
        trans_quat_list = []  # List of translation vectors and quaternion components.
        if len(self.guest_atoms) == 1:
            # There is no sense in rotating a single atom.
            trans_quat_list = deepcopy(positions)
        else:
            for trans, quat in itertools.product(positions, quaternions):
                trans_quat = list(trans)
                trans_quat.extend(quat.components)
                trans_quat_list.append(trans_quat)
        trans_quat_list = np.array(trans_quat_list, dtype=float)
        # Test the translations and rotations for overlap.
        print("[{}] Begin testing of {} structures.".format(_timestamp(),
                                                            len(trans_quat_list)))
        flags = []
        # progress_bar = ProgressBar("Sampling", sample_size, print_iter=False)
        with open(acc_filename, "a") as acc_file, open(rej_filename, "a") as rej_file:
            with tqdm(enumerate(trans_quat_list),
                      desc="Sampling",
                      total=len(trans_quat_list),
                      bar_format="{l_bar}{bar:20}{r_bar}",
                      ncols=120) as pbar:
                for i_transform, trans_quat in pbar:
                    if len(flags) > 0:
                        _acc_percent = round(100 * sum(flags) / len(flags), 1)
                        _rej_percent = round(100 * (1.0 - sum(flags) / len(flags)), 1)
                        pbar.set_postfix_str(f"acc={_acc_percent:3.1f}, rej={_rej_percent:3.1f}")
                    else:
                        _acc_percent = "n.d."
                        _rej_percent = "n.d."
                        pbar.set_postfix_str(f"acc={_acc_percent:4s}, rej={_rej_percent:4s}")
                    _guest_atoms = deepcopy(self.guest_atoms)
                    if len(_guest_atoms) == 1:
                        trans = trans_quat
                        _guest_atoms.positions = rotate_and_shift(None,
                                                                  -trans,
                                                                  self.guest_atoms.positions)
                        line = _line_format(3)
                    else:
                        trans = trans_quat[:3]
                        quat = np.quaternion(*trans_quat[3:])
                        _guest_atoms.positions = rotate_and_shift(quat,
                                                                  -trans,
                                                                  self.guest_atoms.positions)
                        line = _line_format(7)
                    _is_in_pore = _check_position(_guest_atoms,
                                                  self.host_atoms,
                                                  counter=i_transform,
                                                  kind=kind,
                                                  override=override,
                                                  debug=debug,
                                                  print_accessible=print_accessible)
                    flags.append(_is_in_pore)
                    if _is_in_pore:
                        acc_file.write(line.format(*trans_quat))
                    else:
                        rej_file.write(line.format(*trans_quat))
        flags = np.array(flags)
        accessible = trans_quat_list[flags]
        inaccessible = trans_quat_list[flags == False]
        print("Number of non-overlapping positions found: {}".format(len(accessible)))
        print("Number of overlapping positions found: {}".format(len(inaccessible)))
        print("Finished sampling.\n")
        return accessible, inaccessible

    @property
    def host_atoms(self) -> Atoms:
        """Get the host structure."""
        return self._host_atoms

    @property
    def guest_atoms(self) -> Atoms:
        """Get the guest structure."""
        return self._guest_atoms
