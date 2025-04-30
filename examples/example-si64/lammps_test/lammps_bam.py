from typing import Dict, List, Optional, Tuple

import torch
from e3nn.util.jit import compile_mode

from bam_torch.utils.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_BAM(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        for param in self.model.parameters():
            param.requires_grad = False
            

    def forward(self, data: Dict[str, torch.Tensor], local_or_ghost: torch.Tensor, compute_virials: bool = False) -> Dict[str, Optional[torch.Tensor]]:
            num_graphs = data["ptr"].numel() - 1
            data["head"] = self.head
            data["num_nodes"] = torch.tensor(data["positions"].shape[0], dtype=torch.long, device=data["positions"].device)

            # BAM 모델 호출 (forces는 모델에서 직접 가져옴)
            out = self.model(data, backprop=True)


            if "energy" not in out or out["energy"] is None:
                return {
                    "total_energy_local": None,
                    "node_energy": None,
                    "forces": None,
                    "virials": None,
                }

            # output 받기
            energy = out["energy"]  # (1,)
            node_energy = out["node_energy"]  # (n_atoms,)
            positions = data["positions"]  # (n_atoms, 3)

            # === Force 계산: autograd 사용 ===
            forces = torch.autograd.grad(
                outputs=[energy.sum()],
                inputs=[positions],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if forces is not None:
                forces = -forces
            else:
                forces = torch.zeros_like(positions)

            # 에너지 로컬화
            node_energy_local = node_energy * local_or_ghost
            total_energy_local = scatter_sum(
                src=node_energy_local,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )

            # virials는 BAM forward()에서 따로 계산하지 않으므로 빈 텐서로 대체
            virials = torch.zeros((1, 3, 3), dtype=energy.dtype, device=energy.device)

            return {
                "total_energy_local": total_energy_local,
                "node_energy": node_energy,
                "forces": forces,
                "virials": virials,
            }

'''
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = False
        if compute_virials:
            compute_displacement = True
        data["head"] = self.head
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        positions = data["positions"]
        displacement = out["displacement"]
        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        # accumulate energies of local atoms
        # node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]
        if compute_virials and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros_like(displacement)
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }
'''