#pragma once
#include <stdio.h>
#include <string>
#include <cusparse.h>

#include "cuda_runtime_api.h"

inline void succeed(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		printf("CUDA ERROR: %s", cudaGetErrorString(e));
		throw(cudaGetErrorString(e));
	}
}

inline void cuSparseSucceed(cusparseStatus_t status, std::string errorMsg = "CusparseError")
{
	if (status != CUSPARSE_STATUS_SUCCESS)
		throw (errorMsg.c_str());
}


template <typename T>
inline constexpr std::enable_if_t<std::is_unsigned<T>::value, T> divup(T a, T b)
{
	return (a + b - 1) / b;
}