#include "MyLoop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_host_helpers.h"
#include <cub/device/device_scan.cuh>
#include <cub/cub.cuh>
#include <cusparse.h>
#include <numeric>
#include <iostream>

using namespace std;

#define pi 3.1415926535897932384626434f

__global__ void calculate_vertices_valency(
	const int* M_ids, 
	int* offset, 
	int* valences, 
	const int nnz)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= nnz)
		return;

	offset[i] = atomicAdd(&valences[M_ids[i]], 1);

}

__global__ void calculate_adj_matrix(
	const int* M_ids, 
	const int* offset, 
	const int* E_ptr, 
	int* E_ids, 
	int* E_vals, 
	int* E_v,
	const int nnz,
	const int nf)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= nf)
		return;

	int v0 = M_ids[i];
	int v1 = M_ids[i + nf];
	int v2 = M_ids[i + 2 * nf];

	int adj_id0 = offset[i] + E_ptr[v0];
	E_ids[adj_id0] = v1;
	E_vals[adj_id0] = v0 < v1;
	
	E_v[adj_id0] = v0;
	E_v[adj_id0 + nnz] = v2;

	int adj_id1 = offset[i + nf] + E_ptr[v1];
	E_ids[adj_id1] = v2;
	E_vals[adj_id1] = v1 < v2;

	E_v[adj_id1] = v1;
	E_v[adj_id1 + nnz] = v0;

	int adj_id2 = offset[i + 2 * nf] + E_ptr[v2];
	E_ids[adj_id2] = v0;
	E_vals[adj_id2] = v2 < v0;

	E_v[adj_id2] = v2;
	E_v[adj_id2 + nnz] = v1;
}

__global__ void refine_topology(
	const int* E_ptr, 
	const int* E_ids, 
	const int* E_num, 
	const int* c_ids, 
	int* r_ids, 
	const int nv, 
	const int nf) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= nf)
		return;

	// rmeh.ids
	int v0 = c_ids[i];
	int v1 = c_ids[i + nf];
	int v2 = c_ids[i + 2 * nf];

	// edge 0 1
	int e2 = 0;
	for (int j = E_ptr[min(v0,v1)]; j < E_ptr[min(v0, v1) + 1]; ++j) {
		if (E_ids[j] == max(v0, v1)) 
		{
			e2 = E_num[j];
			break;
		}
	}
	// edge 1 2
	int e0 = 0;
	for (int j = E_ptr[min(v1, v2)]; j < E_ptr[min(v1, v2) + 1]; ++j) {
		if (E_ids[j] == max(v1, v2))
		{
			e0 = E_num[j];
			break;
		}
	}
	// edge 2 0
	int e1 = 0;
	for (int j = E_ptr[min(v2, v0)]; j < E_ptr[min(v2, v0) + 1]; ++j) {
		if (E_ids[j] == max(v2, v0))
		{
			e1 = E_num[j];
			break;
		}
	}
	reinterpret_cast<int4*>(r_ids)[i] = { v0, v1, v2, e0 + nv -1 };
	reinterpret_cast<int4*>(r_ids)[i + nf] = { e2 + nv - 1, e0 + nv - 1, e1 + nv - 1, e1 + nv - 1 };
	reinterpret_cast<int4*>(r_ids)[i + 2 * nf] = { e1 + nv - 1, e2 + nv - 1, e0 + nv - 1, e2 + nv - 1 };
}


__global__ void calculate_original_points(
	const int* E_ptr, 
	const int* E_ids, 
	float* verts_ref, 
	const float* verts, 
	const int nv)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= nv)
		return;

	float vals_x = 0.0f;
	float vals_y = 0.0f;
	float vals_z = 0.0f;
	int n = 0;

	int rid = 0;
	for (int j = E_ptr[i]; j < E_ptr[i + 1]; ++j) {
		rid = E_ids[j];
		vals_x += verts[4 * rid];
		vals_y += verts[4 * rid + 1];
		vals_z += verts[4 * rid + 2];
		++n;
	}

	float c = cos(2 * pi / n);
	float beta = (0.625f - (0.375f + c / 4.0f) * (0.375f + c / 4.0f)) / n;

	verts_ref[4 * i] = beta * vals_x + (1 - n * beta) * verts[4 * i];
	verts_ref[4 * i + 1] = beta * vals_y + (1 - n * beta) * verts[4 * i + 1];
	verts_ref[4 * i + 2] = beta * vals_z + (1 - n * beta) * verts[4 * i + 2];
	verts_ref[4 * i + 3] = 1.0f;
}

__global__ void calculate_edge_verts_v2(
	const int* M_ids,
	const int* offset,
	const float* verts,
	const int* E_ptr,
	const int* E_ids,
	const int* E_num,
	const int* E_v,
	float* verts_ref,
	const int nv,
	const int nf,
	const int nnz) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= nnz)
		return;

	int v0 = E_ids[i];
	int v2 = E_v[i];

	if (v2 < v0) {
		for (int index = E_ptr[v0]; index < E_ptr[v0 + 1]; ++index) {
			if (E_ids[index] == v2) {
				int v3 = E_v[index + nnz];
				int v1 = E_v[i + nnz];
				int e = E_num[i];
				verts_ref[4 * (e + nv - 1)] = 3.0f * (verts[4 * v2] + verts[4 * v0]) / 8.0f + (verts[4 * v1] + verts[4 * v3]) / 8.0f;
				verts_ref[4 * (e + nv - 1) + 1] = 3.0f * (verts[4 * v2 + 1] + verts[4 * v0 + 1]) / 8.0f + (verts[4 * v1 + 1] + verts[4 * v3 + 1]) / 8.0f;
				verts_ref[4 * (e + nv - 1) + 2] = 3.0f * (verts[4 * v2 + 2] + verts[4 * v0 + 2]) / 8.0f + (verts[4 * v1 + 2] + verts[4 * v3 + 2]) / 8.0f;
				verts_ref[4 * (e + nv - 1) + 3] = 1.0f;
				break;
			}
		}
	}
}


__global__ void mesh_vertex_refine(
	const int* id, 
	const float* dsts, 
	float* vert, 
	const int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= n)
		return;

	int j = id[i];

	vert[4 * j] = dsts[3 * i];
	vert[4 * j + 1] = dsts[3 * i + 1];
	vert[4 * j + 2] = dsts[3 * i + 2];
}


void MyLoop::cmesh_subdivison() {
	tmesh = cmesh;
	mat_init();
	time = 0;
	std::vector<cudaEvent_t> evts(10 * subdiv_level);
	std::vector<float> usedtime(5 * subdiv_level, 0);
	for (auto& tt : evts) cudaEventCreate(&tt);

	for (int current_level = 1; current_level <= subdiv_level; ++current_level) {

		mat_disposition(tmesh, evts, current_level);

		mesh_subdivide(tmesh, rmesh, evts, current_level);

		tmesh.free();

		if (current_level < subdiv_level) {
			tmesh = rmesh;
			rmesh.free();
		}
	}
	tmesh.free();
	mat.free();
	cudaEventSynchronize(evts.back());

	for (int i = 0; i < 5 * subdiv_level; ++i) {
		cudaEventElapsedTime(&usedtime[i], evts[2 * i], evts[2 * i + 1]);
		time += usedtime[i];
	}

	for (auto& tt : evts) cudaEventDestroy(tt);
}

void MyLoop::cmesh_refine(std::vector<int> const& P2PVtxIds, std::vector<float> const& p2pDsts) {
	int* id = 0;
	float* dsts = 0;
	cudaMalloc(reinterpret_cast<void**>(&id), P2PVtxIds.size() * sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&dsts), p2pDsts.size() * sizeof(float));

	cudaMemcpy(id, &P2PVtxIds[0], P2PVtxIds.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dsts, &p2pDsts[0], p2pDsts.size() * sizeof(float), cudaMemcpyHostToDevice);

	mesh_vertex_refine << <P2PVtxIds.size() / 32 + 1, 32 >> > (id, dsts, cmesh.verts, P2PVtxIds.size());

	succeed(cudaFree(id));
	succeed(cudaFree(dsts));
}

void MyLoop::mat_init() {
	int nv = cmesh.nverts;
	int ne = cmesh.nfaces + cmesh.nverts - 2;
	int nf = cmesh.nfaces;
	int nnz = cmesh.nnz;
	int tmp_nf = nf, tmp_nv = nv, tmp_nnz = nnz;

	for (int current_level = 1; current_level <= subdiv_level; ++current_level) {
		nf = 4 * tmp_nf;
		nv = 2 * tmp_nv + tmp_nf - 2;
		nnz = 4 * tmp_nnz;
		tmp_nf = nf, tmp_nv = nv, tmp_nnz = nnz;
	}
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.offset), nnz * sizeof(int)));
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.ptr), (nv + 1) * sizeof(int)));
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.ids), nnz * sizeof(int)));
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.vals), nnz * sizeof(int)));
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.vert), 2 * nnz * sizeof(int)));
	succeed(cudaMalloc(reinterpret_cast<void**>(&mat.edge_number), nnz * sizeof(int)));
}


void MyLoop::mat_disposition(MeshInfo<int, float>& mesh, std::vector<cudaEvent_t>& evts, int current_level) {
	int nv = mesh.nverts;
	int ne = mesh.nfaces + tmesh.nverts - 2;
	int nf = mesh.nfaces;
	int nnz = mesh.nnz;

	int* valences = nullptr;

	succeed(cudaMalloc((void**)&valences, nv * sizeof(int)));

	cudaMemsetAsync(valences, 0, nv * sizeof(int));

	cudaEventRecord(evts[10 * (current_level - 1) + 0]);
	calculate_vertices_valency << <nnz / 256 + 1, 256 >> > (mesh.ids, mat.offset, valences, nnz);
	cudaEventRecord(evts[10 * (current_level - 1) + 1]);

	void* ptr_temp_storage = NULL;
	size_t ptr_temp_storage_bytes = 0;

	cub::DeviceScan::ExclusiveSum(ptr_temp_storage, ptr_temp_storage_bytes, valences, mat.ptr, nv + 1);
	cudaMalloc(&ptr_temp_storage, ptr_temp_storage_bytes);
	cub::DeviceScan::ExclusiveSum(ptr_temp_storage, ptr_temp_storage_bytes, valences, mat.ptr, nv + 1);

	cudaEventRecord(evts[10 * (current_level - 1) + 2]);
	calculate_adj_matrix << <nf / 256 + 1, 256 >> > (mesh.ids, mat.offset, mat.ptr, mat.ids, mat.vals, mat.vert, nnz, nf);
	cudaEventRecord(evts[10 * (current_level - 1) + 3]);

	void* num_temp_storage = NULL;
	size_t num_temp_storage_bytes = 0;

	cub::DeviceScan::InclusiveSum(num_temp_storage, num_temp_storage_bytes, mat.vals, mat.edge_number, nnz);
	cudaMalloc(&num_temp_storage, num_temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(num_temp_storage, num_temp_storage_bytes, mat.vals, mat.edge_number, nnz);

	succeed(cudaFree(ptr_temp_storage));
	succeed(cudaFree(num_temp_storage));
	succeed(cudaFree(valences));
}

void MyLoop::mesh_subdivide(MeshInfo<int, float>& inmesh, MeshInfo<int, float>& outmesh, std::vector<cudaEvent_t>& evts, int current_level) {
	outmesh.nfaces = 4 * inmesh.nfaces;
	outmesh.nverts = inmesh.nverts + (inmesh.nfaces + inmesh.nverts - 2);
	outmesh.nnz = 4 * inmesh.nnz;

	int nv = inmesh.nverts;
	int ne = inmesh.nfaces + inmesh.nverts - 2;
	int nf = inmesh.nfaces;
	int nnz = inmesh.nnz;
	int sub_nv = outmesh.nverts;
	int sub_nf = outmesh.nfaces;

	cudaMalloc(reinterpret_cast<void**>(&outmesh.ids), outmesh.nnz * sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&outmesh.verts), 4 * sub_nv * sizeof(float));

	cudaEventRecord(evts[10 * (current_level - 1) + 4]);
	refine_topology << < nf / 256 + 1, 256 >> > (mat.ptr, mat.ids, mat.edge_number, inmesh.ids, outmesh.ids, nv, nf);
	cudaEventRecord(evts[10 * (current_level - 1) + 5]);

	cudaMemsetAsync(outmesh.verts, 0, 4 * sub_nv * sizeof(float));

	cudaEventRecord(evts[10 * (current_level - 1) + 6]);
	calculate_original_points << < nv / 256 + 1, 256 >> > (mat.ptr, mat.ids, outmesh.verts, inmesh.verts, nv);
	cudaEventRecord(evts[10 * (current_level - 1) + 7]);

	cudaEventRecord(evts[10 * (current_level - 1) + 8]);
	calculate_edge_verts_v2 << < nnz / 256 + 1, 265 >> > (inmesh.ids, mat.offset, inmesh.verts, mat.ptr, mat.ids, mat.edge_number, mat.vert, outmesh.verts, nv, nf, nnz);
	cudaEventRecord(evts[10 * (current_level - 1) + 9]);

}
