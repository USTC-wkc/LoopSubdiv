#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void launch_array(float3* pos, float* verts, const int nv) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= nv)
        return;

    float a = verts[4 * i];
    float b = verts[4 * i + 1];
    float c = verts[4 * i + 2];

    pos[i] = make_float3(a, b, c);
}

__global__ void launch_faces(int3* pos, int* ids, const int nf) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= nf)
        return;

    int a = ids[3 * i];
    int b = ids[3 * i + 1];
    int c = ids[3 * i + 2];

    pos[i] = make_int3(a, b, c);
}

__global__ void launch_faces_trans(int3* pos, int* ids, const int nf) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= nf)
        return;

    int a = ids[i];
    int b = ids[i + nf];
    int c = ids[i + 2 * nf];

    pos[i] = make_int3(a, b, c);
}