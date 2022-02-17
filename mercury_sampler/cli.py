#!/usr/bin/env python
import click
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read

from mercury_sampler.sampler import Sampler

# Banner made with the `art` package. All credits to its authors. It is amazing!
# https://pypi.org/project/art/
_silver_sampler = r"""
******************************************************************************
               _  _         ___                     _                        
              | || | __ _  / __| __ _  _ __   _ __ | | ___  _ _ 
              | __ |/ _` | \__ \/ _` || '  \ | '_ \| |/ -_)| '_|
              |_||_|\__, | |___/\__,_||_|_|_|| .__/|_|\___||_|  
                    |___/                    |_|                
******************************************************************************
"""


@click.command(help="Sample the pores of a host with some guest structure.")
@click.option("--host_filename", "-H", default="geometry.in", type=str,
              help="File containing the host structure.")
@click.option("--guest_filename", "-G", default="guest-atoms.json", type=str,
              help="File containing the guest structure.")
@click.option("--guest", "-g", default="none", type=str,
              help="Chemical formula of the guest structure. Use ASE to get "
                   "the structure of common molecules. If this is given, "
                   "ignore --guest_filename.")
@click.option("--override", "-r", default=-1.0, type=float,
              help="Overrides the radius for a mono-atomic guest species. This "
                   "is similar to having an artificial hard sphere probe with "
                   "which the host cell is sampled.")
@click.option("--acc_filename", "-acc", default="accessible.dat", type=str,
              help="File to which the accessible and thus accepted shifts "
                   "(and rotations) are saved.")
@click.option("--rej_filename", "-rej", default="inaccessible.dat", type=str,
              help="File to which the inaccessible and thus rejected shifts "
                   "(and rotations) are saved.")
@click.option("--sample_size", "-S", default=1000, type=int,
              help="Number of random samples to be drawn. This number is used "
                   "for both, random shifts and random rotations, if "
                   "--pos_sample_size or --quat_sample_size are not specified.")
@click.option("--pos_sample_size", "-Sp", type=int,
              help="Number of random shifts to be drawn. Per default --sample_size "
                   "is used if this is not specified.")
@click.option("--quat_sample_size", "-Sq", type=int,
              help="Number of random rotations to be drawn. Per default --sample_size "
                   "is used if this is not specified.")
@click.option("--kind", "-K", default="vdw", type=str,
              help="Kind of atomic radii to be used for the acceptance criterion. Per "
                   "default, van der Waals radii are used (Alvarez, S.; Dalton Trans. "
                   "2013, 42, 8617-8636).")
@click.option("--data", "-D", default="none", type=str,
              help="File in which a pore sampling (usually with a small probe body, "
                   "like an H atom) is stored. Using this, can greatly increase the "
                   "yield of accessible shifts and rotations.")
def sample(host_filename: str,
           guest_filename: str,
           guest: str,
           acc_filename: str,
           rej_filename: str,
           sample_size: int,
           pos_sample_size: int,
           quat_sample_size: int,
           kind: str,
           override: float,
           data: str):
    print(_silver_sampler)
    print("Reading host structure from file: {}".format(host_filename))
    host_atoms = read(host_filename)
    if guest == "none":
        print("Reading guest structure from file: {}".format(host_filename))
        guest_atoms = read(guest_filename)
    else:
        try:
            guest_atoms = molecule(guest)
            print("Build guest structure with ASE: {}".format(guest))
        except KeyError:
            try:
                guest_atoms = Atoms(guest)
                print("Atomic guest structure: {}".format(guest))
            except KeyError:
                raise
    print("Using {} as guest.".format(guest_atoms.get_chemical_formula()))
    positions = None
    quaternions = None
    if data != "none":
        print("Using data from file {}".format(data))
        data = np.loadtxt(data)
        positions = data[:, :3]
        pos_sample_size = len(positions)
        if data.shape[1] > 3:
            quaternions = data[:, 3:7]
            quat_sample_size = len(quaternions)
    sampler = Sampler(host_atoms=host_atoms,
                      guest_atoms=guest_atoms)
    sampler(acc_filename=acc_filename,
            rej_filename=rej_filename,
            sample_size=sample_size,
            pos_sample_size=pos_sample_size,
            quat_sample_size=quat_sample_size,
            kind=kind,
            override=override,
            guest_atoms_filename=guest_filename,
            positions=positions,
            quaternions=quaternions)


if __name__ == "__main__":
    sample()
