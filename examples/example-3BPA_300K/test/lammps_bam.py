@compile_mode("script")
class LAMMPS_BAM(torch.nn.Module):
    def __init__(self, model, head: str = None):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers.clone().detach())
        self.register_buffer("r_max", torch.tensor(model.r_max).clone().detach())
        self.register_buffer("num_interactions", torch.tensor(model.num_interactions).clone().detach())
        
        # heads 처리
        if not hasattr(model, "heads") or model.heads is None:
            model.heads = ["default"]
        head_idx = model.heads.index(head if head is not None else model.heads[-1])
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))
        
        # 모델 파라미터 고정
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if "cell" in data:
            print(f"data['cell'].shape: {data['cell'].shape}")
        
        # edge_index가 없으면 동적으로 생성
        if "edge_index" not in data:
            n_atoms = data["positions"].shape[0]
            edge_i: List[int] = []
            edge_j: List[int] = []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        edge_i.append(i)
                        edge_j.append(j)
            data["edge_index"] = torch.tensor([edge_i, edge_j], dtype=torch.long, device=data["positions"].device)

        # 기본값 설정
        num_graphs = data["ptr"].numel() - 1
        data["head"] = self.head
        data["batch"] = data.get("batch", torch.zeros(data["positions"].shape[0], dtype=torch.long, device=data["positions"].device))

        # species 전처리: LAMMPS 타입 번호(1, 2, 3, 4)를 모델 인덱스(0, 1, 2, 3)로 변환
        species = data.get("species", torch.ones(data["positions"].shape[0], dtype=torch.long, device=data["positions"].device))
        # LAMMPS 타입 번호가 1부터 시작한다고 가정하고 0-based 인덱스로 변환
        data["species"] = species - 1  # 1,2,3,4 -> 0,1,2,3

        # 디버깅 출력 추가
        print("Processed species:", data["species"])
        print("Unique species:", torch.unique(data["species"]))

        # RACE 모델 호출
        out = self.model(data, backprop=True)  # backprop=True로 forces 계산 보장
        total_energy = out["energy"]  # RACE는 "energy"를 반환
        forces = out.get("forces", torch.zeros_like(data["positions"]))  # forces가 없는 경우 대비

        # node_energy 계산
        node_energy = scatter_sum(
            src=total_energy,
            index=data["batch"],
            dim=-1,
            dim_size=data["positions"].shape[0],
            out=torch.zeros(data["positions"].shape[0], device=total_energy.device)
        )

        # 로컬 에너지 합산
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs
        )

        # virials 계산 (compute_virials가 True일 때만)
        virials = torch.zeros((num_graphs, 3, 3), device=total_energy.device)
        if compute_virials:
            displacement = torch.zeros((num_graphs, 3, 3), device=total_energy.device)
            displacement.requires_grad_(True)
            total_energy_with_disp = total_energy + torch.sum(displacement * torch.zeros_like(displacement))
            forces_grad, virials_grad = torch.autograd.grad(
                outputs=[total_energy_with_disp],
                inputs=[data["positions"], displacement],
                grad_outputs=torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(total_energy)]),
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            forces = -forces_grad if forces_grad is not None else torch.zeros_like(data["positions"])
            virials = -virials_grad if virials_grad is not None else torch.zeros_like(displacement)
        else:
            if forces is None:
                forces = torch.autograd.grad(
                    outputs=[total_energy_local],
                    inputs=[data["positions"]],
                    grad_outputs=torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(total_energy_local)]),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
                forces = -forces if forces is not None else torch.zeros_like(data["positions"])

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }