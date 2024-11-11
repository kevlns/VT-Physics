#include "PBFCudaApi.cuh"

#include "Modules/SPH/KernelFunctions.cuh"

namespace VT_Physics::pbf {

#define CHECK_THREAD() \
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_data->particle_num)                     \
        return;                                         \
    auto p_i = d_nsParams->particleIndices_cuData[i];

#define FOR_EACH_NEIGHBOR_Pj() \
       auto neib_ind = p_i * d_nsConfig->maxNeighborNum;                        \
       for (unsigned int p_j = d_nsParams->neighbors_cuData[neib_ind], t = 0;   \
            p_j != UINT_MAX && t < d_nsConfig->maxNeighborNum;                  \
            ++t, p_j = d_nsParams->neighbors_cuData[neib_ind + t])

#define CONST_VALUE(name) \
        d_data->name

#define DATA_VALUE(name, index) \
        d_data->name[index]

#define CUBIC_KERNEL_VALUE() \
        sph::cubic_value(pos_i - pos_j, d_data->h)

#define CUBIC_KERNEL_GRAD() \
        sph::cubic_gradient(pos_i - pos_j, d_data->h)

}

namespace VT_Physics::pbf { // cuda kernels
    __global__ void
    init_cuda(Data *d_data) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= d_data->particle_num)
            return;

        DATA_VALUE(volume, i) = CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(dx, i) *= 0;
        DATA_VALUE(mass, i) = CONST_VALUE(fPart_rest_density) * CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(error, i) = 0;
        DATA_VALUE(error_grad, i) *= 0;
    }

    __global__ void
    compute_rigid_particle_volume_cuda(Data *d_data,
                                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_BOUNDARY)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        float delta = 0;
        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);

            if (DATA_VALUE(mat, p_j) == DATA_VALUE(mat, p_i))
                delta += CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(volume, p_i) = 1.f / delta;

        DATA_VALUE(mass, p_i) = CONST_VALUE(bPart_rest_density) * DATA_VALUE(volume, p_i);
    }

    __global__ void
    compute_sph_density_and_error_cuda(Data *d_data,
                                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(density_sph, p_i) *= 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto m_j = DATA_VALUE(mass, p_j);

            DATA_VALUE(density_sph, p_i) += m_j * CUBIC_KERNEL_VALUE();
        }

        if (DATA_VALUE(density_sph, p_i) < CONST_VALUE(fPart_rest_density))
            DATA_VALUE(density_sph, p_i) = CONST_VALUE(fPart_rest_density);

        DATA_VALUE(error, p_i) = DATA_VALUE(density_sph, p_i) / CONST_VALUE(fPart_rest_density) - 1.f;
    }

    __global__ void
    update_lamb_cuda(Data *d_data,
                     UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                     UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(error_grad, p_i) *= 0;
        DATA_VALUE(lamb, p_i) = 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            auto error_grad_j = -1 / CONST_VALUE(fPart_rest_density) * wGrad;
            DATA_VALUE(error_grad, p_i) += wGrad;
            DATA_VALUE(lamb, p_i) += dot(error_grad_j, error_grad_j);
        }

        DATA_VALUE(error_grad, p_i) /= CONST_VALUE(fPart_rest_density);

        DATA_VALUE(lamb, p_i) += dot(DATA_VALUE(error_grad, p_i), DATA_VALUE(error_grad, p_i));

        DATA_VALUE(lamb, p_i) = -DATA_VALUE(error, p_i) / (DATA_VALUE(lamb, p_i) + 1e-6f);
    }

    __global__ void
    compute_dx_cuda(Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(dx, p_i) *= 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            DATA_VALUE(dx, p_i) += (DATA_VALUE(lamb, p_i) + DATA_VALUE(lamb, p_j)) * wGrad;
        }

        DATA_VALUE(dx, p_i) /= CONST_VALUE(fPart_rest_density);
    }

    __global__ void
    apply_ext_force_cuda(Data *d_data,
                         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(vel, p_i) += CONST_VALUE(gravity) * CONST_VALUE(dt);
        DATA_VALUE(pos, p_i) += DATA_VALUE(vel, p_i) * CONST_VALUE(dt);
    }

    __global__ void
    apply_dx_cuda(Data *d_data,
                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(pos, p_i) += DATA_VALUE(dx, p_i);
        DATA_VALUE(vel, p_i) += DATA_VALUE(dx, p_i) / CONST_VALUE(dt);
    }

    __global__ void
    XSPH_cuda(Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        auto vel_i = DATA_VALUE(vel, p_i);
        DATA_VALUE(dx, p_i) *= 0;
        float3 dv = {0, 0, 0};

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto vel_j = DATA_VALUE(vel, p_j);

            dv += CONST_VALUE(XSPH_k) * DATA_VALUE(volume, p_j) * (vel_j - vel_i) * CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(dx, p_i) += dv * CONST_VALUE(dt);
    }
}

namespace VT_Physics::pbf { // host invoke api
    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // init_data_cuda
        init_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data);

        // compute_rigid_particle_volume_cuda
        compute_rigid_particle_volume_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                      d_nsConfig,
                                                                                      d_nsParams);
    }

    __host__ void
    compute_sph_density_and_error(Data *h_data,
                                  Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // compute_sph_density_and_lamb_cuda
        compute_sph_density_and_error_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                      d_nsConfig,
                                                                                      d_nsParams);
    }

    __host__ void
    update_lamb(Data *h_data,
                Data *d_data,
                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // update_lamb_cuda
        update_lamb_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                    d_nsConfig,
                                                                    d_nsParams);
    }

    __host__ void
    compute_dx(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // compute_dx_cuda
        compute_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                   d_nsConfig,
                                                                   d_nsParams);
    }

    __host__ void
    apply_ext_force(Data *h_data,
                    Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // apply_ext_force_cuda
        apply_ext_force_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                        d_nsConfig,
                                                                        d_nsParams);
    }

    __host__ void
    apply_dx(Data *h_data,
             Data *d_data,
             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // apply_dx_cuda
        apply_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                 d_nsConfig,
                                                                 d_nsParams);
    }

    __host__ void
    post_correct(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // XSPH_cuda
        XSPH_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                             d_nsConfig,
                                                             d_nsParams);

        // apply_dx_cuda
        apply_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                 d_nsConfig,
                                                                 d_nsParams);
    }
}