import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from bam_torch.utils.utils import find_input_json, extract_species
import json
import torch
from e3nn.util import jit
from bam_torch.model import models as bam_models
from bam_torch.model.models import get_edge_relative_vectors_with_pbc_lammps
from lammps_bam import LAMMPS_BAM

def main():
    input_json_path = find_input_json()
    with open(input_json_path) as f:
         json_data = json.load(f)

    # JSON에서 필요한 정보 추출
    model_path = json_data.get("model_path", "model.pt")
    pckl = torch.load('model.pkl', weights_only=False)
    nlayers = pckl['input.json']['nlayers']
    cutoff = pckl['input.json']['cutoff']

    model = torch.load("model.pt", weights_only=False)
    model.eval()
    
    species = extract_species("train_si.xyz")
    model.atomic_numbers = species.clone().detach()
    # model.num_species = len(species)
    model.num_interactions = torch.tensor(nlayers)
    model.r_max = torch.tensor(cutoff)

    model = model.float().to("cpu")

    print("\n=== [DEBUG] Dumping model wrapping info ===")
    print(f"atomic_numbers = {model.atomic_numbers}")
    print(f"r_max = {model.r_max}")
    print(f"num_interactions = {model.num_interactions}")
    print("Model heads = ", model.heads if hasattr(model, "heads") else "None")
    print("=== [DEBUG] Wrapping info done ===\n")

    bam_models.get_edge_relative_vectors_with_pbc = get_edge_relative_vectors_with_pbc_lammps
    model.training_mode_for_lammps = True
    

    lammps_model = LAMMPS_BAM(model)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
