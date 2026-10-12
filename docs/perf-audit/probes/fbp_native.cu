#include <cuda_runtime.h>
#include <astra/ParallelProjectionGeometry2D.h>
#include <astra/VolumeGeometry2D.h>
#include <astra/cuda/2d/par_bp.h>
#include <cmath>
#include <cstddef>
#include <exception>
#include <vector>

__constant__ float4 batch_coefficients[2560];

namespace astraCUDA {
bool BP_internal(float*, unsigned int, float*, unsigned int, const SDimensions&,
                 const SProjectorParams2D&, const SParProjection*, cudaStream_t);
}

struct Context {
    astra::Geometry2DParameters geometry;
    astraCUDA::SProjectorParams2D parameters;
    float* coefficients = nullptr;

    ~Context() { if (coefficients) cudaFree(coefficients); }
};

__global__ void batch_backproject(float* output, cudaTextureObject_t texture,
                                 unsigned int columns,
                                 unsigned int angles, unsigned int start_angle,
                                 float output_scale) {
    unsigned int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int row = blockIdx.z;
    if (pixel_x >= columns || pixel_y >= columns) return;
    float coordinate_x = pixel_x - 0.5f * columns + 0.5f;
    float coordinate_y = pixel_y - 0.5f * columns + 0.5f;
    float subtotal = 0.0f;
    unsigned int end_angle = min(start_angle + 16, angles);
    for (unsigned int angle = start_angle; angle < end_angle; ++angle) {
        float4 coefficient = batch_coefficients[angle];
        float detector = coordinate_x * coefficient.x
                       - coordinate_y * coefficient.y + coefficient.z;
        subtotal += tex2D<float>(texture, detector, row * angles + angle + 0.5f)
                  * coefficient.w;
    }
    output[(static_cast<size_t>(row) * columns + pixel_y) * columns + pixel_x]
        += subtotal * output_scale;
}

extern "C" void* fbp_create(unsigned int columns, unsigned int angles, const float* theta) {
    try {
        if (angles > 2560) return nullptr;
        std::vector<float> negative_theta(angles);
        for (unsigned int angle = 0; angle < angles; ++angle) negative_theta[angle] = -theta[angle];
        astra::CParallelProjectionGeometry2D projections(angles, columns, 1.0f, std::move(negative_theta));
        astra::CVolumeGeometry2D volume(columns, columns);
        Context* context = new Context;
        context->geometry = astra::convertAstraGeometry(&volume, &projections);
        context->parameters.fOutputScale = context->geometry.getOutputScale()
            * (static_cast<float>(M_PI) / 2.0f) / angles;
        std::vector<float> coefficients(angles * 4);
        const auto* vectors = context->geometry.getParallel();
        for (unsigned int angle = 0; angle < angles; ++angle) {
            const auto& vector = vectors[angle];
            double determinant = vector.fDetUX * vector.fRayY - vector.fDetUY * vector.fRayX;
            coefficients[angle * 4] = vector.fRayY / determinant;
            coefficients[angle * 4 + 1] = -vector.fRayX / determinant;
            coefficients[angle * 4 + 2] = (vector.fDetSY * vector.fRayX - vector.fDetSX * vector.fRayY) / determinant;
            coefficients[angle * 4 + 3] = std::sqrt(vector.fRayX * vector.fRayX + vector.fRayY * vector.fRayY) / std::abs(determinant);
        }
        if (cudaMalloc(&context->coefficients, coefficients.size() * sizeof(float)) != cudaSuccess) {
            delete context;
            return nullptr;
        }
        if (cudaMemcpy(context->coefficients, coefficients.data(), coefficients.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            delete context;
            return nullptr;
        }
        return context;
    } catch (const std::exception&) { return nullptr; }
}

extern "C" int fbp_run(void* handle, float* input, float* output, unsigned int rows,
                       unsigned int projection_pitch, void* stream_handle, unsigned int mode) {
    Context* context = static_cast<Context*>(handle);
    auto stream = static_cast<cudaStream_t>(stream_handle);
    const auto& dimensions = context->geometry.getDims();
    size_t output_bytes = static_cast<size_t>(rows) * dimensions.iVolWidth * dimensions.iVolHeight * sizeof(float);
    cudaError_t error = cudaMemsetAsync(output, 0, output_bytes, stream);
    if (error != cudaSuccess) return error;
    if (mode == 0 || mode == 2) {
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) return error;
        for (unsigned int row = 0; row < rows; ++row) {
            float* volume = output + static_cast<size_t>(row) * dimensions.iVolWidth * dimensions.iVolHeight;
            float* projections = input + static_cast<size_t>(row) * dimensions.iProjAngles * projection_pitch;
            bool success = mode == 2 && row > 0
                ? astraCUDA::BP_internal(volume, dimensions.iVolWidth, projections, projection_pitch,
                    dimensions, context->parameters, context->geometry.getParallel(), stream)
                : astraCUDA::BP(volume, dimensions.iVolWidth, projections, projection_pitch,
                    dimensions, context->parameters, context->geometry.getParallel());
            if (!success) return -1;
        }
        return 0;
    }
    error = cudaMemcpyToSymbolAsync(batch_coefficients, context->coefficients,
        dimensions.iProjAngles * sizeof(float4), 0, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return error;
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = input;
    resource.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resource.res.pitch2D.width = dimensions.iProjDets;
    resource.res.pitch2D.height = rows * dimensions.iProjAngles;
    resource.res.pitch2D.pitchInBytes = projection_pitch * sizeof(float);
    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeBorder;
    description.addressMode[1] = cudaAddressModeBorder;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    cudaTextureObject_t texture;
    error = cudaCreateTextureObject(&texture, &resource, &description, nullptr);
    if (error != cudaSuccess) return error;
    dim3 threads(16, 32);
    dim3 blocks((dimensions.iVolWidth + 15) / 16, (dimensions.iVolHeight + 31) / 32, rows);
    for (unsigned int angle = 0; angle < dimensions.iProjAngles; angle += 16) {
        batch_backproject<<<blocks, threads, 0, stream>>>(output, texture,
            dimensions.iVolWidth, dimensions.iProjAngles, angle, context->parameters.fOutputScale);
    }
    error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    cudaError_t cleanup_error = cudaDestroyTextureObject(texture);
    return error != cudaSuccess ? error : cleanup_error;
}

extern "C" void fbp_destroy(void* handle) { delete static_cast<Context*>(handle); }
