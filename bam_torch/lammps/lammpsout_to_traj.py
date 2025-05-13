import numpy as np
import re
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator


atoms_list = read("dump.lammpstrj", index=slice(None))

LOG = open("log.lammps")
log = LOG.readlines()
energies = []
collecting = False

for i, line in enumerate(log):
    line_ls = line.split()

    if 'Step' in line_ls:
        collecting = True
        continue

    if 'Loop' in line_ls:
        collecting = False
        continue

    if collecting:
        enr = float(line.split()[2])
        energies.append(enr)

traj = Trajectory('lammps_out.traj', 'w')
for i, atoms in enumerate(atoms_list):
    # Add energy values to atoms object
    # ASE's SinglePointCalculator only accepts specific property names
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energies[i],    # Total energy
        forces=atoms.get_forces(),
        stress=np.zeros(6)        # Placeholder stress
    )
    # Store potential energy as a custom info attribute
    atoms.info['potential_energy'] = energies[i]
    traj.write(atoms)
traj.close()
