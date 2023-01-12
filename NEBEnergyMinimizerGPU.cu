// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "NEBEnergyMinimizerGPU.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#include "thrust/transform_reduce.h"
#include "thrust/functional.h"
#include "thrust/device_ptr.h"
#include "thrust/tuple.h"
#include <thrust/iterator/zip_iterator.h>

#include <assert.h>

#include <stdio.h>

/*! \file NEBEnergyMinimizerGPU.cu
    \brief Defines GPU kernel code for one performing one NEB energy
    minimization iteration on the GPU. Used by NEBEnergyMinimizerGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void gpu_neb_compute_disp_kernel(const unsigned int N,
                                        const BoxDim box,
                                        const Scalar4* d_pos,
                                        const unsigned int* d_tags,
                                        Scalar3* d_left_disp,
                                        Scalar3* d_right_disp,
                                        const Scalar4* d_left_pos,
                                        const Scalar4* d_right_pos,
                                        const unsigned int* d_left_rtags,
                                        const unsigned int* d_right_rtags)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        {
        unsigned int tag = d_tags[i];
        unsigned int left_idx = d_left_rtags[tag];
        unsigned int right_idx = d_right_rtags[tag];

        Scalar4 pi = d_pos[i];
        Scalar4 pl = d_left_pos[left_idx];
        Scalar4 pr = d_right_pos[right_idx];

        Scalar3 left_disp = box.minImage(
            make_scalar3(pi.x - pl.x, pi.y - pl.y, pi.z - pl.z));
        Scalar3 right_disp = box.minImage(
            make_scalar3(pr.x - pi.x, pr.y - pi.y, pr.z - pi.z));

        d_left_disp[i] = left_disp; // make_scalar3(0.0, 0.0, 0.0); //
        d_right_disp[i] = right_disp;
        }
    }


struct dot_kernel
    {
    __host__ __device__ Scalar operator()(const Scalar3& x) const {
        return x.x*x.x + x.y*x.y + x.z*x.z;
        }
    };

__global__ void gpu_neb_compute_tangent_kernel(const unsigned int N,
                                            const Scalar3* d_left_disp,
                                            const Scalar3* d_right_disp,
                                            const Scalar left_norm,
                                            const Scalar right_norm,
                                            Scalar3* d_tangent)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        {
        Scalar3 left_disp = d_left_disp[i];
        Scalar3 right_disp = d_right_disp[i];

        Scalar3 tangent = left_disp * left_norm + right_disp * right_norm;

        d_tangent[i] = tangent;
        }
    }

typedef thrust::tuple<Scalar4, Scalar3, Scalar3, Scalar3> tuple4;

struct nudge_kernel
    {
    Scalar k;
    __host__ __device__ Scalar operator()(tuple4 t)
        {
        Scalar4 f = thrust::get<0>(t);
        Scalar3 tangent = thrust::get<1>(t);
        Scalar3 left_disp = thrust::get<2>(t);
        Scalar3 right_disp = thrust::get<3>(t);

        Scalar3 term = k * (right_disp - left_disp) - make_scalar3(f.x, f.y, f.z);

        return dot(term, tangent);
        }
    };

__global__ void gpu_neb_apply_nudge_kernel(const unsigned int N,
                                            Scalar4 *d_net_force,
                                            const Scalar3 *d_tangent,
                                            Scalar nudge
                                            )
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        {
        Scalar3 nudge_force = d_tangent[i] * nudge;
        d_net_force[i].x += nudge_force.x;
        d_net_force[i].y += nudge_force.y;
        d_net_force[i].z += nudge_force.z;
        }
    }

hipError_t sync() {
    return hipDeviceSynchronize();
}

hipError_t gpu_neb_nudge_force(const unsigned int N,
                                const BoxDim& box,
                                Scalar4* d_net_force,
                                const Scalar4* d_pos,
                                const unsigned int* d_tags,
                                Scalar3* d_tangent,
                                Scalar3* d_left_disp,
                                Scalar3* d_right_disp,
                                const Scalar4* d_left_pos,
                                const Scalar4* d_right_pos,
                                const unsigned int* d_left_rtags,
                                const unsigned int* d_right_rtags,
                                const Scalar k)
    {
    
    int block_size = 256;
    dim3 grid((N / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    // auto L = box.getL();
    // std::cout << "N: " << N << " box: "  << L.x << " " << L.y << " " << L.z << " " << std::endl;
    // std::cout << "1: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    thrust::device_ptr<Scalar3> dev_left_disp = thrust::device_pointer_cast(d_left_disp);
    thrust::device_ptr<Scalar3> dev_right_disp = thrust::device_pointer_cast(d_right_disp);
    thrust::device_ptr<Scalar3> dev_tangent = thrust::device_pointer_cast(d_tangent);
    thrust::device_ptr<Scalar4> dev_net_force = thrust::device_pointer_cast(d_net_force);

    dot_kernel unary_op;
    thrust::plus<Scalar> binary_op;

    // std::cout << "2: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // compute disp vectors (un-normalized)
    hipLaunchKernelGGL((gpu_neb_compute_disp_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       N,
                       box,
                       d_pos,
                       d_tags,
                       d_left_disp,
                       d_right_disp,
                       d_left_pos,
                       d_right_pos,
                       d_left_rtags,
                       d_right_rtags);

    // std::cout << "3: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // hipDeviceSynchronize();

    // normalize disp vectors
    Scalar left_norm = 1.0/sqrt(thrust::transform_reduce(dev_left_disp, dev_left_disp + N, unary_op, 0.0, binary_op));

    Scalar right_norm = 1.0/sqrt(thrust::transform_reduce(dev_right_disp, dev_right_disp + N, unary_op, 0.0, binary_op));

    // std::cout << "4: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // compute tangent vector (un-normalized)
    hipLaunchKernelGGL((gpu_neb_compute_tangent_kernel),
                        dim3(grid),
                        dim3(threads),
                        0,
                        0,
                        N,
                        d_left_disp,
                        d_right_disp,
                        left_norm,
                        right_norm,
                        d_tangent);

    hipDeviceSynchronize();

    // std::cout << "5: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // normalize tangent vector
    Scalar tangent_norm = 1.0/sqrt(thrust::transform_reduce(dev_tangent, dev_tangent + N, unary_op, 0.0, binary_op));

    // compute nudge force
    // first reduce to get tan and normal force components
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_net_force, dev_tangent, dev_left_disp, dev_right_disp));
    auto zip_end = zip_begin + N;
    Scalar nudge = thrust::transform_reduce(zip_begin, zip_end, nudge_kernel{k}, 0.0, binary_op);

    // std::cout << "6: " << cudaGetLastError() << std::endl;


    // then apply to net_force
    hipLaunchKernelGGL((gpu_neb_apply_nudge_kernel),
                        dim3(grid),
                        dim3(threads),
                        0,
                        0,
                        N,
                        d_net_force,
                        d_tangent,
                        nudge);

    // std::cout << "7: " << cudaGetLastError() << std::endl;

    return hipSuccess;
    }

//! The kernel function to zeros velocities, called by gpu_neb_zero_v()
/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indices of the members of the group to zero
    \param group_size Number of members in the group
*/
__global__ void
gpu_neb_zero_v_kernel(Scalar4* d_vel, unsigned int* d_group_members, unsigned int group_size)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read the particle's velocity (MEM TRANSFER: 32 bytes)
        Scalar4 vel = d_vel[idx];

        // zero the velocity(FLOPS: ?)
        vel.x = Scalar(0.0);
        vel.y = Scalar(0.0);
        vel.z = Scalar(0.0);

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        }
    }

//! The kernel function to zero angular momenta, called by gpu_neb_zero_angmom()
__global__ void gpu_neb_zero_angmom_kernel(Scalar4* d_angmom,
                                            unsigned int* d_group_members,
                                            unsigned int group_size)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_angmom[idx] = make_scalar4(0, 0, 0, 0);
        }
    }

/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group

This function is just the driver for gpu_neb_zero_v_kernel(), see that function
for details.
*/
hipError_t gpu_neb_zero_v(Scalar4* d_vel, unsigned int* d_group_members, unsigned int group_size)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_neb_zero_v_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_vel,
                       d_group_members,
                       group_size);

    return hipSuccess;
    }

hipError_t
gpu_neb_zero_angmom(Scalar4* d_angmom, unsigned int* d_group_members, unsigned int group_size)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_neb_zero_angmom_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_angmom,
                       d_group_members,
                       group_size);

    return hipSuccess;
    }

//! Kernel function for reducing the potential energy to a partial sum
/*! \param d_group_members Device array listing the indices of the members of the group to sum
    \param group_size Number of members in the group
    \param d_net_force Pointer to the force array for all particles
    \param d_partial_sum_pe Placeholder for the partial sum
*/
__global__ void gpu_neb_reduce_pe_partial_kernel(unsigned int* d_group_members,
                                                  unsigned int group_size,
                                                  Scalar4* d_net_force,
                                                  Scalar* d_partial_sum_pe)
    {
    extern __shared__ Scalar neb_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar pe = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        // read the particle's force and extract the pe from w component (MEM TRANSFER: 32 bytes)

        Scalar4 force = d_net_force[idx];
        pe = force.w;

        // Uncoalesced Memory Read replace by Texture Read above.  Scalars4* d_net_force still being
        // passed to support this defunct structure.
        // pe = d_net_force[idx].w;
        }

    neb_sdata[threadIdx.x] = pe;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_sdata[threadIdx.x] += neb_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum_pe[blockIdx.x] = neb_sdata[0];
        }
    }

//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the partial sum
    \param num_blocks Number of blocks to execute
*/
__global__ void
gpu_neb_reduce_partial_sum_kernel(Scalar* d_sum, Scalar* d_partial_sum, unsigned int num_blocks)
    {
    extern __shared__ Scalar neb_sdata[];

    Scalar sum = Scalar(0.0);

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            neb_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            neb_sdata[threadIdx.x] = Scalar(0.0);
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                neb_sdata[threadIdx.x] += neb_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        // everybody sums up sum2K
        sum += neb_sdata[0];
        }

    if (threadIdx.x == 0)
        *d_sum = sum;
    }

/*!  \param d_group_members Device array listing the indices of the members of the group to
   integrate \param group_size Number of members in the group \param d_net_force Array containing
   the net forces \param d_sum_pe Placeholder for the sum of the PE \param d_partial_sum_pe Array
   containing the partial sum of the PE \param block_size The size of one block \param num_blocks
   Number of blocks to execute

    This is a driver for gpu_neb_reduce_pe_partial_kernel() and
    gpu_neb_reduce_partial_sum_kernel(), see them for details
*/
hipError_t gpu_neb_compute_sum_pe(unsigned int* d_group_members,
                                   unsigned int group_size,
                                   Scalar4* d_net_force,
                                   Scalar* d_sum_pe,
                                   Scalar* d_partial_sum_pe,
                                   unsigned int block_size,
                                   unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_neb_reduce_pe_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_group_members,
                       group_size,
                       d_net_force,
                       d_partial_sum_pe);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(1, 1, 1),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_sum_pe,
                       d_partial_sum_pe,
                       num_blocks);

    return hipSuccess;
    }

//! Kernel function to compute the partial sum over the P term in the NEB algorithm
/*! \param d_vel particle velocities and masses on the device
    \param d_accel particle accelerations on the device
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_partial_sum_P Array to hold the partial sum
*/
__global__ void gpu_neb_reduce_P_partial_kernel(const Scalar4* d_vel,
                                                 const Scalar3* d_accel,
                                                 unsigned int* d_group_members,
                                                 unsigned int group_size,
                                                 Scalar* d_partial_sum_P)
    {
    extern __shared__ Scalar neb_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar P = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar3 a = d_accel[idx];
        Scalar4 v = d_vel[idx];
        P = a.x * v.x + a.y * v.y + a.z * v.z;
        }

    neb_sdata[threadIdx.x] = P;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_sdata[threadIdx.x] += neb_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_P[blockIdx.x] = neb_sdata[0];
    }

// Angular terms
__global__ void gpu_neb_reduce_Pr_partial_kernel(const Scalar4* d_angmom,
                                                  const Scalar4* d_orientation,
                                                  const Scalar3* d_inertia,
                                                  const Scalar4* d_net_torque,
                                                  unsigned int* d_group_members,
                                                  unsigned int group_size,
                                                  Scalar* d_partial_sum_Pr)
    {
    extern __shared__ Scalar neb_scalar_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar Pr = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> p(d_angmom[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = 0;
        if (y_zero)
            t.y = 0;
        if (z_zero)
            t.z = 0;

        // s is the pure imaginary quaternion with im. part equal to true angular velocity
        vec3<Scalar> s = (Scalar(1. / 2.) * conj(q) * p).v;

        // rotational power = torque * angvel
        Pr = dot(t, s);
        }

    neb_scalar_sdata[threadIdx.x] = Pr;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_scalar_sdata[threadIdx.x] += neb_scalar_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_Pr[blockIdx.x] = neb_scalar_sdata[0];
    }

// Norm of angular velocity vector
__global__ void gpu_neb_reduce_wnorm_partial_kernel(const Scalar4* d_angmom,
                                                     const Scalar4* d_orientation,
                                                     unsigned int* d_group_members,
                                                     unsigned int group_size,
                                                     Scalar* d_partial_sum_w)
    {
    extern __shared__ Scalar neb_scalar_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar w = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        quat<Scalar> p(d_angmom[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> s = (Scalar(1. / 2.) * conj(q) * p).v;

        w = dot(s, s);
        }

    neb_scalar_sdata[threadIdx.x] = w;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_scalar_sdata[threadIdx.x] += neb_scalar_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_w[blockIdx.x] = neb_scalar_sdata[0];
    }

//! Kernel function to compute the partial sum over the vsq term in the NEB algorithm
/*! \param d_vel device array of particle velocities
    \param d_group_members Array listing members of the group
    \param group_size Number of members in the group
    \param d_partial_sum_vsq Array to hold the partial sum
*/
__global__ void gpu_neb_reduce_vsq_partial_kernel(const Scalar4* d_vel,
                                                   unsigned int* d_group_members,
                                                   unsigned int group_size,
                                                   Scalar* d_partial_sum_vsq)
    {
    extern __shared__ Scalar neb_vsq_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar vsq = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar4 v = d_vel[idx];
        vsq = v.x * v.x + v.y * v.y + v.z * v.z;
        }

    neb_vsq_sdata[threadIdx.x] = vsq;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_vsq_sdata[threadIdx.x] += neb_vsq_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_vsq[blockIdx.x] = neb_vsq_sdata[0];
    }

//! Kernel function to compute the partial sum over the asq term in the NEB algorithm
/*! \param d_accel device array of particle accelerations
    \param d_group_members Array listing members of the group
    \param group_size Number of members in the group
    \param d_partial_sum_asq Array to hold the partial sum
*/
__global__ void gpu_neb_reduce_asq_partial_kernel(const Scalar3* d_accel,
                                                   unsigned int* d_group_members,
                                                   unsigned int group_size,
                                                   Scalar* d_partial_sum_asq)
    {
    extern __shared__ Scalar neb_partial_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar asq = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar3 a = d_accel[idx];
        asq = a.x * a.x + a.y * a.y + a.z * a.z;
        }

    neb_partial_sdata[threadIdx.x] = asq;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_partial_sdata[threadIdx.x] += neb_partial_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_asq[blockIdx.x] = neb_partial_sdata[0];
    }

__global__ void gpu_neb_reduce_tsq_partial_kernel(const Scalar4* d_net_torque,
                                                   const Scalar4* d_orientation,
                                                   const Scalar3* d_inertia,
                                                   unsigned int* d_group_members,
                                                   unsigned int group_size,
                                                   Scalar* d_partial_sum_tsq)
    {
    extern __shared__ Scalar neb_scalar_sdata[];

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar tsq = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = 0;
        if (y_zero)
            t.y = 0;
        if (z_zero)
            t.z = 0;

        tsq = dot(t, t);
        }

    neb_scalar_sdata[threadIdx.x] = tsq;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            neb_scalar_sdata[threadIdx.x] += neb_scalar_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_tsq[blockIdx.x] = neb_scalar_sdata[0];
    }

/*! \param N number of particles in system
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_sum_all Array to hold the sum over P, vsq, and asq
    \param d_partial_sum_P Array to hold the partial sum over P (a*v)
    \param d_partial_sum_vsq Array to hold the partial sum over vsq (v*v)
    \param d_partial_sum_asq Array to hold the partial sum over asq (a*a)
    \param block_size is the size of one block
    \param num_blocks is the number of blocks to execute
    \note Currently the sums are performed consecutively. The efficiency of this
        function could be improved by computing all three sums simultaneously
    This is a driver for gpu_neb_reduce_{X}_partial_kernel() (where X = P, vsq, asq)
    and gpu_neb_reduce_partial_sum_kernel(), see them for details
*/
hipError_t gpu_neb_compute_sum_all(const unsigned int N,
                                    const Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    Scalar* d_sum_all,
                                    Scalar* d_partial_sum_P,
                                    Scalar* d_partial_sum_vsq,
                                    Scalar* d_partial_sum_asq,
                                    unsigned int block_size,
                                    unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernels
    hipLaunchKernelGGL((gpu_neb_reduce_P_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_vel,
                       d_accel,
                       d_group_members,
                       group_size,
                       d_partial_sum_P);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[0],
                       d_partial_sum_P,
                       num_blocks);

    hipLaunchKernelGGL((gpu_neb_reduce_vsq_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_vel,
                       d_group_members,
                       group_size,
                       d_partial_sum_vsq);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[1],
                       d_partial_sum_vsq,
                       num_blocks);

    hipLaunchKernelGGL((gpu_neb_reduce_asq_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_accel,
                       d_group_members,
                       group_size,
                       d_partial_sum_asq);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[2],
                       d_partial_sum_asq,
                       num_blocks);

    return hipSuccess;
    }

hipError_t gpu_neb_compute_sum_all_angular(const unsigned int N,
                                            const Scalar4* d_orientation,
                                            const Scalar3* d_inertia,
                                            const Scalar4* d_angmom,
                                            const Scalar4* d_net_torque,
                                            unsigned int* d_group_members,
                                            unsigned int group_size,
                                            Scalar* d_sum_all,
                                            Scalar* d_partial_sum_Pr,
                                            Scalar* d_partial_sum_wnorm,
                                            Scalar* d_partial_sum_tsq,
                                            unsigned int block_size,
                                            unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernels
    hipLaunchKernelGGL((gpu_neb_reduce_Pr_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_angmom,
                       d_orientation,
                       d_inertia,
                       d_net_torque,
                       d_group_members,
                       group_size,
                       d_partial_sum_Pr);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[0],
                       d_partial_sum_Pr,
                       num_blocks);

    hipLaunchKernelGGL((gpu_neb_reduce_wnorm_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_angmom,
                       d_orientation,
                       d_group_members,
                       group_size,
                       d_partial_sum_wnorm);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[1],
                       d_partial_sum_wnorm,
                       num_blocks);

    hipLaunchKernelGGL((gpu_neb_reduce_tsq_partial_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_net_torque,
                       d_orientation,
                       d_inertia,
                       d_group_members,
                       group_size,
                       d_partial_sum_tsq);

    hipLaunchKernelGGL((gpu_neb_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads1),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_all[2],
                       d_partial_sum_tsq,
                       num_blocks);

    return hipSuccess;
    }

//! Kernel function to update the velocities used by the NEB algorithm
/*! \param d_vel Array of velocities to update
    \param d_accel Array of accelerations
    \param d_group_members Device array listing the indices of the members of the group to update
    \param group_size Number of members in the grou
    \param alpha Alpha coupling parameter used by the NEB algorithm
    \param factor_t Combined factor vnorm/fnorm*alpha, or 1 if fnorm==0
*/
__global__ void gpu_neb_update_v_kernel(Scalar4* d_vel,
                                         const Scalar3* d_accel,
                                         unsigned int* d_group_members,
                                         unsigned int group_size,
                                         Scalar alpha,
                                         Scalar factor_t)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 v = d_vel[idx];
        Scalar3 a = d_accel[idx];

        v.x = v.x * (Scalar(1.0) - alpha) + a.x * factor_t;
        v.y = v.y * (Scalar(1.0) - alpha) + a.y * factor_t;
        v.z = v.z * (Scalar(1.0) - alpha) + a.z * factor_t;

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = v;
        }
    }

/*! \param d_vel array of particle velocities to update
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param alpha Alpha coupling parameter used by the NEB algorithm
    \param vnorm Magnitude of the (3*N) dimensional velocity vector
    \param invfnorm 1 over the magnitude of the (3*N) dimensional force vector

    This function is a driver for gpu_neb_update_v_kernel(), see it for details.
*/
hipError_t gpu_neb_update_v(Scalar4* d_vel,
                             const Scalar3* d_accel,
                             unsigned int* d_group_members,
                             unsigned int group_size,
                             Scalar alpha,
                             Scalar factor_t)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_neb_update_v_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_vel,
                       d_accel,
                       d_group_members,
                       group_size,
                       alpha,
                       factor_t);

    return hipSuccess;
    }

__global__ void gpu_neb_update_angmom_kernel(const Scalar4* d_net_torque,
                                              const Scalar4* d_orientation,
                                              const Scalar3* d_inertia,
                                              Scalar4* d_angmom,
                                              unsigned int* d_group_members,
                                              unsigned int group_size,
                                              Scalar alpha,
                                              Scalar factor_r)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> p(d_angmom[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = 0;
        if (y_zero)
            t.y = 0;
        if (z_zero)
            t.z = 0;

        p = p * Scalar(1.0 - alpha) + Scalar(2.0) * q * t * factor_r;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

hipError_t gpu_neb_update_angmom(const Scalar4* d_net_torque,
                                  const Scalar4* d_orientation,
                                  const Scalar3* d_inertia,
                                  Scalar4* d_angmom,
                                  unsigned int* d_group_members,
                                  unsigned int group_size,
                                  Scalar alpha,
                                  Scalar factor_r)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_neb_update_angmom_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_net_torque,
                       d_orientation,
                       d_inertia,
                       d_angmom,
                       d_group_members,
                       group_size,
                       alpha,
                       factor_r);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
