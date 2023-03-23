#pragma once
#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"
#include <algorithm>
#include <vector>

#include "Loop/obj.h"
#include "Loop/MeshInfo.h"
#include "Loop/MatInfo.h"


class MyLoop
{

public:

	MyLoop() = default;

	~MyLoop() {
		cmesh.free();
		rmesh.free();
		mat.free();
	}

	float get_time() {
		return time;
	}

	int get_level() {
		return subdiv_level;
	}

	void set_level(int target_level) {
		subdiv_level = target_level;
	}

	void set_cmesh(const std::string& path) {
		cmesh.set_mesh(path);
	}
	
	void cmesh_refine(std::vector<int> const& P2PVtxIds, std::vector<float> const& p2pDsts);
	void cmesh_subdivison();

	MeshInfo<int, float> rmesh;

private:

	MeshInfo<int, float> cmesh;
	MeshInfo<int, float> tmesh;

	MatInfo<int> mat;

	int subdiv_level = 0;

	float time = 0;
	
	void mat_init();

	void mat_disposition(MeshInfo<int, float> &mesh, std::vector<cudaEvent_t> &evts, int current_level);

	void mesh_subdivide(MeshInfo<int, float>& inmesh, MeshInfo<int, float>& outmesh, std::vector<cudaEvent_t>& evts, int current_level);
};

