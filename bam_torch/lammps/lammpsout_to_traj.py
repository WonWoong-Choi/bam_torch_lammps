import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

# 궤적 파일 읽기
atoms_list = read("dump.lammpstrj", index=slice(None))

# 로그 파일에서 에너지 추출
energies = []
with open("log.lammps") as log_file:
    collecting = False
    for line in log_file:
        line_split = line.split()
        if not line_split:
            continue
        
        if 'Step' in line_split:
            collecting = True
            continue
        elif 'Loop' in line_split:
            collecting = False
            continue
            
        if collecting and len(line_split) > 2:
            try:
                energies.append(float(line_split[2]))
            except (ValueError, IndexError):
                pass

# 에너지 수와 프레임 수가 맞는지 확인
if len(atoms_list) != len(energies):
    print(f"경고: 프레임 수({len(atoms_list)})와 에너지 수({len(energies)})가 일치하지 않습니다.")
    # 더 짧은 길이에 맞춤
    min_length = min(len(atoms_list), len(energies))
    atoms_list = atoms_list[:min_length]
    energies = energies[:min_length]

# ASE 궤적 파일로 저장
with Trajectory('lammps_out.traj', 'w') as traj:
    for atoms, energy in zip(atoms_list, energies):
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=atoms.get_forces(),
            stress=np.zeros(6)
        )
        atoms.info['potential_energy'] = energy
        traj.write(atoms)
