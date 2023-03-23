#pragma once
#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"
#include <algorithm>
#include <vector>

#include "Loop/obj.h"

template<typename INDEX_TYPE, typename VERTEX_TYPE>
struct MeshInfo
{

	size_t nfaces{ 0 };
	size_t nverts{ 0 };
	size_t nnz{ 0 };

	INDEX_TYPE* ids{ nullptr };

	VERTEX_TYPE* verts{ nullptr };

	MeshInfo() = default;


	~MeshInfo()
	{
		free();
	}

	MeshInfo& operator=(const MeshInfo& rhs)
	{
		if (&rhs == this)
			return *this;

		free();
		MeshInfo::copy(*this, rhs);

		return *this;
	}

	void set_mesh(const std::string& path) {
		OBJ::Data obj;
		OBJ::read(obj, path.c_str());

		nnz = obj.f_vi.size();
		nfaces = obj.f_offs.size() - 1;
		nverts = obj.v.size();

		std::vector<math::float4> h_verts(nverts);
		for (size_t i = 0; i < nverts; ++i)
			h_verts[i] = math::float4(obj.v[i], 1.0f);

		std::vector<INDEX_TYPE> f_vi_trans(nnz);
		for (size_t j = 0; j < nfaces; ++j) {
			f_vi_trans[j] = obj.f_vi[3 * j];
			f_vi_trans[nfaces + j] = obj.f_vi[3 * j + 1];
			f_vi_trans[2 * nfaces + j] = obj.f_vi[3 * j + 2];
		}
		setFromCPU(f_vi_trans, h_verts);
	}

	void alloc(size_t nfaces, size_t nverts, size_t nnz)
	{
		succeed(cudaMalloc(reinterpret_cast<void**>(&ids), nnz * sizeof(INDEX_TYPE)));
		succeed(cudaMalloc(reinterpret_cast<void**>(&verts), nverts * 4 * sizeof(VERTEX_TYPE)));
	}

	void free()
	{

		if (ids != nullptr)
			succeed(cudaFree(ids));

		if (verts != nullptr)
			succeed(cudaFree(verts));

		ids = nullptr;
		verts = nullptr;

		nfaces = 0, nverts = 0, nnz = 0;
	}

	void setFromCPU(std::vector<INDEX_TYPE>& cpu_ids, std::vector<math::float4>& cpu_verts)
	{
		free();

		nfaces = cpu_ids.size() / 3;
		nverts = *std::max_element(cpu_ids.begin(), cpu_ids.end()) + 1;
		nnz = cpu_ids.size();

		alloc(nfaces, nverts, nnz);

		succeed(cudaMemcpy(this->ids, &cpu_ids[0], nnz * sizeof(INDEX_TYPE), cudaMemcpyHostToDevice));
		succeed(cudaMemcpy(this->verts, &cpu_verts[0], nverts * sizeof(math::float4), cudaMemcpyHostToDevice));
	}

	static void copy(MeshInfo& dst, const MeshInfo& src)
	{
		if (&dst == &src)
			return;

		dst.nfaces = src.nfaces, dst.nverts = src.nverts, dst.nnz = src.nnz;

		if (src.ids != nullptr) {
			succeed(cudaMalloc(reinterpret_cast<void**>(&dst.ids), src.nnz * sizeof(INDEX_TYPE)));
			succeed(cudaMemcpy(dst.ids, src.ids, src.nnz * sizeof(INDEX_TYPE), cudaMemcpyDeviceToDevice));
		}

		if (src.verts != nullptr) {
			succeed(cudaMalloc(reinterpret_cast<void**>(&dst.verts), src.nverts * 4 * sizeof(VERTEX_TYPE)));
			succeed(cudaMemcpy(dst.verts, src.verts, src.nverts * 4 * sizeof(VERTEX_TYPE), cudaMemcpyDeviceToDevice));
		}
	}
};