#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "general_struct_gpu.h"

namespace cuda_filters
{
__global__ void isFiniteFilter(CudaPointXYZAER* points_cu_,
                                std::uint32_t point_cloud_size);

__global__ void distanceFilter(CudaPointXYZAER* points_cu_,
                                std::uint32_t point_cloud_size,
                                float minDistanceSq);

}