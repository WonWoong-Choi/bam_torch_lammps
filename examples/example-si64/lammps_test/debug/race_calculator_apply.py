from bam_torch.tase.base_calculator import BaseCalculator
from ase.io import read, write
import json


with open('input.json') as f:
	json_data = json.load(f)

calc = BaseCalculator(json_data)


atoms = read("si64_test_xyz")
#write("si64_test.data", atoms, format="lammps-data")
atoms.set_calculator(calc)
enr = atoms.get_potential_energy()

