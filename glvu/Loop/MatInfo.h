#pragma once
#include <cuda_runtime_api.h>
#include "cuda_host_helpers.h"
#include <algorithm>
#include <vector>

#include "Loop/obj.h"

template<typename INDEX_TYPE>
struct MatInfo
{
	size_t nverts{ 0 };

	INDEX_TYPE* ptr{ nullptr };

	INDEX_TYPE* ids{ nullptr };

	INDEX_TYPE* vals{ nullptr };

	INDEX_TYPE* edge_number{ nullptr };

	INDEX_TYPE* vert{ nullptr };

	INDEX_TYPE* vert2{ nullptr };

	INDEX_TYPE* offset{ nullptr };

	MatInfo() = default;

	~MatInfo()
	{
		free();
	}

	void free()
	{
		if (ptr != nullptr)
			succeed(cudaFree(ptr));

		if (ids != nullptr)
			succeed(cudaFree(ids));

		if (vals != nullptr)
			succeed(cudaFree(vals));

		if (edge_number != nullptr)
			succeed(cudaFree(edge_number));

		if (offset != nullptr)
			succeed(cudaFree(offset));

		if (vert != nullptr)
			succeed(cudaFree(vert));

		ptr = nullptr, ids = nullptr, vals = nullptr, edge_number = nullptr, offset = nullptr, vert = nullptr;
		nverts = 0;
	}
};
