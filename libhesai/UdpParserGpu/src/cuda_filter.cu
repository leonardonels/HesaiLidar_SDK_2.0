#include "cuda_filter.h"

/**
 * struct CudaPointXYZAER {
 *     float x;
 *     float y;
 *     float z;
 *     float azimuthCalib;
 *     float elevationCalib;
 *     uint8_t reserved[8];
 * };
 */

namespace cuda_filters{

__global__ void isFiniteFilter(
    CudaPointXYZAER* points_cu_,
    std::uint32_t point_cloud_size)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= point_cloud_size) return;

    if (!isfinite(points_cu_[id].x) ||
        !isfinite(points_cu_[id].y) ||
        !isfinite(points_cu_[id].z))
    {
        points_cu_[id].x = 0;
        points_cu_[id].y = 0;
        points_cu_[id].z = 0;
    }
}

__global__ void distanceFilter(
    CudaPointXYZAER* points_cu_,
    std::uint32_t point_cloud_size,
    float minDistanceSq)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= point_cloud_size) return;

    if ((points_cu_[id].x * points_cu_[id].x +
        points_cu_[id].y * points_cu_[id].y +
        points_cu_[id].z * points_cu_[id].z) <
        minDistanceSq)
    {
        points_cu_[id].x = 0;
        points_cu_[id].y = 0;
        points_cu_[id].z = 0;   
    }
}
}