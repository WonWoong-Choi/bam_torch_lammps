# pylint: disable=wrong-import-position
import os
import json

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from e3nn.util import jit

from lammps_bam import LAMMPS_BAM
from bam_torch.utils.utils import extract_species, find_input_json


def main():
    # JSON 파일에서 설정 로드
    input_json_path = find_input_json()  # bam_torch의 유틸리티 함수로 JSON 파일 경로 찾기
    with open(input_json_path) as f:
        json_data = json.load(f)

    # JSON에서 필요한 정보 추출
    model_path = json_data.get("model_path", "model.pt")  # 기본값 설정
    dtype = json_data.get("dtype", "float64")  # 데이터 타입
    head = json_data.get("head", None)  # 헤드 선택 (기본값 None)

    # bam_torch 방식으로 모델 로드 및 속성 설정
    pckl = torch.load('model.pkl', weights_only=True)  # 설정 정보 로드
    nlayers = pckl['input.json']['nlayers']
    cutoff = pckl['input.json']['cutoff']

    model = torch.load(model_path, weights_only=True)  # JSON에서 지정된 모델 경로 사용
    model.eval()

    # 속성 설정
    species = extract_species("train_300K.xyz")  # atomic_numbers 추출
    model.atomic_numbers = species.clone().detach()
    model.num_species = max(species) + 1
    model.num_interactions = torch.tensor(nlayers, dtype=torch.long)
    model.r_max = torch.tensor(cutoff, dtype=torch.float)
    print(f"atomic_numbers: {model.atomic_numbers}")
    print(f"num_interactions: {model.num_interactions}")
    print(f"r_max: {model.r_max}")

    # 데이터 타입 설정
    if dtype == "float64":
        model = model.double().to("cpu")
    elif dtype == "float32":
        print("Converting model to float32, this may cause loss of precision.")
        model = model.float().to("cpu")

    # 헤드 처리 (JSON에서 지정되지 않은 경우 기본값 사용)
    if head is None and hasattr(model, "heads") and len(model.heads) > 1:
        print("Multiple heads detected, but no head specified in JSON. Using the last head.")
        head = model.heads[-1]
    if head is not None:
        print(f"Using head: {head}")

    # LAMMPS_BAM으로 래핑
    lammps_model = (
        LAMMPS_BAM(model, head=head) if head is not None else LAMMPS_BAM(model)
    )
    
    # JIT 컴파일 및 저장
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
