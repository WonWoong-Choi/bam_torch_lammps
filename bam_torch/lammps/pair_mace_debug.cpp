/* pair_bam.cpp : BAM-torch(RACE) 모델용 LAMMPS pair_style */

#include "pair_mace.h" // 기존 pair_mace를 복사해오되 이름만 수정
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

PairMACE::PairMACE(LAMMPS *lmp) : Pair(lmp) {
  no_virial_fdotr_compute = 1;
}

PairMACE::~PairMACE() {}

void PairMACE::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);

  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
  if (domain_decomposition) {
    if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");
  }

  int n_nodes = domain_decomposition ? atom->nlocal + atom->nghost : atom->nlocal;

  std::cout << "[DEBUG] atom->nlocal = " << atom->nlocal << std::endl;
  std::cout << "[DEBUG] atom->nghost = " << atom->nghost << std::endl;
  std::cout << "[DEBUG] list->inum = " << list->inum << std::endl;
  std::cout << "[DEBUG] n_nodes = " << n_nodes << std::endl;

  // positions
  auto positions = torch::empty({n_nodes, 3}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii = 0; ii < n_nodes; ++ii) {
    int i = list->ilist[ii];
    positions[i][0] = atom->x[i][0];
    positions[i][1] = atom->x[i][1];
    positions[i][2] = atom->x[i][2];
  }
  positions.set_requires_grad(true);

  // cell
  auto cell = torch::zeros({3, 3}, torch_float_dtype);
  cell[0][0] = domain->h[0];
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = domain->h[5];
  cell[1][1] = domain->h[1];
  cell[1][2] = 0.0;
  cell[2][0] = domain->h[4];
  cell[2][1] = domain->h[3];
  cell[2][2] = domain->h[2];

  // ========== 여기부터 edge 계산 ==========
  int n_edges = 0;
  std::vector<int> n_edges_vec(n_nodes, 0);

  #pragma omp parallel for reduction(+:n_edges)
  for (int ii = 0; ii < n_nodes; ++ii) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj] & NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        n_edges++;
        n_edges_vec[ii]++;
      }
    }
  }

  std::vector<int> first_edge(n_nodes);
  first_edge[0] = 0;
  for (int ii = 0; ii < n_nodes - 1; ++ii) {
    first_edge[ii + 1] = first_edge[ii] + n_edges_vec[ii];
  }

  auto edge_index = torch::empty({2, n_edges}, torch::kInt64);
  auto unit_shifts = torch::zeros({n_edges, 3}, torch_float_dtype);
  auto shifts = torch::zeros({n_edges, 3}, torch_float_dtype);
  auto edges = torch::zeros({n_edges,3}, torch_float_dtype); // ★ 추가
  

  #pragma omp parallel for
  for (int ii = 0; ii < n_nodes; ++ii) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    int k = first_edge[ii];
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj] & NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        edge_index[0][k] = i;
        if (domain_decomposition) {
          edge_index[1][k] = j;
        } else {
          int j_local = atom->map(atom->tag[j]);
          edge_index[1][k] = j_local;

          double shiftx = atom->x[j][0] - atom->x[j_local][0];
          double shifty = atom->x[j][1] - atom->x[j_local][1];
          double shiftz = atom->x[j][2] - atom->x[j_local][2];

          double shiftxs = std::round(domain->h_inv[0] * shiftx + domain->h_inv[5] * shifty + domain->h_inv[4] * shiftz);
          double shiftys = std::round(domain->h_inv[1] * shifty + domain->h_inv[3] * shiftz);
          double shiftzs = std::round(domain->h_inv[2] * shiftz);

          unit_shifts[k][0] = shiftxs;
          unit_shifts[k][1] = shiftys;
          unit_shifts[k][2] = shiftzs;

          shifts[k][0] = domain->h[0] * shiftxs + domain->h[5] * shiftys + domain->h[4] * shiftzs;
          shifts[k][1] = domain->h[1] * shiftys + domain->h[3] * shiftzs;
          shifts[k][2] = domain->h[2] * shiftzs;
        }
        k++;
      }
    }
  }
  auto species = torch::zeros({n_nodes}, torch::kInt64);
  #pragma omp parallel for
  for (int ii = 0; ii < n_nodes; ++ii) {
    int i = list->ilist[ii];
    int atomic_num = lammps_atomic_numbers[atom->type[i] - 1];
    for (int j = 0; j < mace_atomic_numbers.size(); ++j) {
      if (atomic_num == mace_atomic_numbers[j]) species[i] = j;
    }
  }

  auto batch = torch::zeros({n_nodes}, torch::kInt64);
  auto ptr = torch::tensor({0, n_nodes}, torch::kInt64);
  auto weight = torch::tensor({1.0}, torch_float_dtype);

  auto input = c10::Dict<std::string, torch::Tensor>();
  input.insert("positions", positions.to(device));
  input.insert("batch", batch.to(device));
  input.insert("ptr", ptr.to(device));
  input.insert("cell", cell.to(device));
  input.insert("edge_index", edge_index.to(device));
  input.insert("unit_shifts", unit_shifts.to(device));
  input.insert("shifts", shifts.to(device));
  input.insert("edges", edges.to(device)); // ★ 추가
  input.insert("species", species.to(device));
  input.insert("weight", weight.to(device));

  auto mask = torch::zeros({n_nodes}, torch::kBool).to(device);
  for (int i = 0; i < atom->nlocal; ++i) mask[list->ilist[i]] = true;

  std::cout << "[Debug] Start model.forward()" << std::endl;
  std::cout << "\n=== [DEBUG] Dumping model input tensors ===" << std::endl;

  // positions
  std::cout << "[positions] shape = (" << positions.size(0) << ", " << positions.size(1) << ")" << std::endl;
  std::cout << positions.slice(0, 0, 5) << std::endl; // 처음 5개만 출력

  // edges
  std::cout << "[edges] shape = (" << edges.size(0) << ", " << edges.size(1) << ")" << std::endl;
  std::cout << edges << std::endl;

  // species
  std::cout << "[species] shape = (" << species.size(0) << ")" << std::endl;
  std::cout << species.slice(0, 0, 5) << std::endl;

  std::cout << "=== [DEBUG] Dumping done ===\n" << std::endl;

  auto output = model.forward({input, mask.to(device), bool(vflag_global)}).toGenericDict();
  std::cout << "[Debug] Finished model.forward()" << std::endl;

  // model output 검사
  if (!output.contains("forces")) {
      error->all(FLERR, "Error: Model output does not contain 'forces' key!");
  }
  if (!output.contains("total_energy_local")) {
      error->all(FLERR, "Error: Model output does not contain 'total_energy_local' key!");
  }
  if (!output.contains("node_energy")) {
      error->all(FLERR, "Error: Model output does not contain 'node_energy' key!");
  }

    // === 추가 디버깅: 에너지 출력 ===
    if (output.contains("total_energy_local")) {
      auto total_energy_local = output.at("total_energy_local").toTensor().cpu();
      std::cout << "[DEBUG] total_energy_local = " << total_energy_local.item<double>() << std::endl;
  }

  if (output.contains("node_energy")) {
      auto node_energy = output.at("node_energy").toTensor().cpu();
      std::cout << "[DEBUG] node_energy ndim = " << node_energy.dim() << std::endl;
      std::cout << "[DEBUG] node_energy shape[0] = " << node_energy.size(0) << std::endl;
      if (node_energy.dim() > 1) {
          std::cout << "[DEBUG] node_energy shape[1] = " << node_energy.size(1) << std::endl;
      }
      std::cout << "[DEBUG] node_energy shape = (" << node_energy.size(0) << ")" << std::endl;

      if (node_energy.dim() == 0) {
          error->all(FLERR, "Error: node_energy is a scalar, expected per-atom vector!");
      }
      if (node_energy.size(0) != n_nodes) {
          error->all(FLERR, "Error: node_energy size mismatch with n_nodes!");
      }

      for (int i = 0; i < std::min<int64_t>(node_energy.size(0), 10); ++i) {
          std::cout << "  node_energy[" << i << "] = " << node_energy[i].item<double>() << std::endl;
      }

      if (eflag_atom) {
          #pragma omp parallel for
          for (int i = 0; i < list->inum; ++i) {
              eatom[list->ilist[i]] = node_energy[i].item<double>();
          }
      }
  } else {
      error->all(FLERR, "Error: model output does not contain 'node_energy'!");
  }

  std::cout << "[Debug] Output contains all expected keys" << std::endl;

  // === 에너지 처리 ===
  if (eflag_global) {
      auto total_energy_local = output.at("total_energy_local").toTensor().cpu();
      eng_vdwl += total_energy_local.item<double>();
  }


  // === Forces 처리 ===
  auto forces = output.at("forces").toTensor().cpu();
  std::cout << "[Debug] Forces shape: (" << forces.size(0) << ", " << forces.size(1) << ")" << std::endl;

  if (forces.size(0) != n_nodes || forces.size(1) != 3) {
      error->all(FLERR, "Error: Forces tensor shape mismatch!");
  }

  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    if (ii >= list->inum) continue;
    int i = list->ilist[ii];
    atom->f[i][0] += forces[i][0].item<double>();
    atom->f[i][1] += forces[i][1].item<double>();
    atom->f[i][2] += forces[i][2].item<double>();
  }
  std::cout << "[unit_shifts] shape = (" << unit_shifts.size(0) << ", " << unit_shifts.size(1) << ")" << std::endl;
  std::cout << unit_shifts.slice(0, 0, 5) << std::endl;

  std::cout << "[shifts] shape = (" << shifts.size(0) << ", " << shifts.size(1) << ")" << std::endl;
  std::cout << shifts.slice(0, 0, 5) << std::endl;

  std::cout << "[Debug] Finished applying forces" << std::endl;
}

/* ---------------------------------------------------------------------- */

void PairMACE::settings(int narg, char **arg)
{
  if (narg > 1) {
    error->all(FLERR, "Too many pair_style arguments for pair_style mace.");
  }

  if (narg == 1) {
    if (strcmp(arg[0], "no_domain_decomposition") == 0) {
      domain_decomposition = false;
      // TODO: add check against MPI rank
    } else {
      error->all(FLERR, "Unrecognized argument for pair_style mace.");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMACE::coeff(int narg, char **arg)
{
  // TODO: remove print statements from this routine, or have a single proc print

  if (!allocated) allocate();

  if (!torch::cuda::is_available()) {
    std::cout << "CUDA unavailable, setting device type to torch::kCPU." << std::endl;
    device = c10::Device(torch::kCPU);
  } else {
    std::cout << "CUDA found, setting device type to torch::kCUDA." << std::endl;
    MPI_Comm local;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localrank;
    MPI_Comm_rank(local, &localrank);
    device = c10::Device(torch::kCUDA,localrank);
  }

  std::cout << "Loading MACE model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2], device);
  std::cout << " finished." << std::endl;

  // extract default dtype from mace model
  for (auto p: model.named_attributes()) {
      // this is a somewhat random choice of variable to check. could it be improved?
      if (p.name == "model.node_embedding.linear.weight") {
          if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<float>()) {
            torch_float_dtype = torch::kFloat32;
          } else if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<double>()) {
            torch_float_dtype = torch::kFloat64;
          }
      }
  }
  std::cout << "  - The torch_float_dtype is: " << torch_float_dtype << std::endl;

  // extract r_max from mace model
  r_max = model.attr("r_max").toTensor().item<double>();
  r_max_squared = r_max*r_max;
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  num_interactions = model.attr("num_interactions").toTensor().item<int64_t>();
  std::cout << "  - The model has: " << num_interactions << " layers." << std::endl;

  // extract atomic numbers from mace model
  auto a_n = model.attr("atomic_numbers").toTensor();
  for (int i=0; i<a_n.size(0); ++i) {
    mace_atomic_numbers.push_back(a_n[i].item<int64_t>());
  }
  std::cout << "  - The MACE model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;

  // extract atomic numbers from pair_coeff
  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<=lammps_atomic_numbers.size(); ++i) {
    std::cout << "  - Mapping LAMMPS type " << i
      << " (" << periodic_table[lammps_atomic_numbers[i-1]-1]
      << ") to MACE type " << mace_type(i) << "." << std::endl;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

void PairMACE::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");

  /*
    MACE requires the full neighbor list AND neighbors of ghost atoms
    it appears that:
      * without REQ_GHOST
           list->gnum == 0
           list->ilist does not include ghost atoms, but the jlists do
      * with REQ_GHOST
           list->gnum == atom->nghost
           list->ilist includes ghost atoms
  */
  if (domain_decomposition) {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
}

double PairMACE::init_one(int i, int j)
{
  // to account for message passing, require cutoff of n_layers * r_max
  return num_interactions*model.attr("r_max").toTensor().item<double>();
}

void PairMACE::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
  memory->create(eatom, atom->nmax, "pair:eatom");
}

int PairMACE::mace_type(int lammps_type)
{
    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
        return i+1;
      }
    }
    error->all(FLERR, "Problem converting lammps_type to mace_type.");
    return -1;
 }
