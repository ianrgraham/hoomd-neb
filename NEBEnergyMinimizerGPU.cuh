// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __NEB_ENERGY_MINIMIZER_GPU_CUH__
#define __NEB_ENERGY_MINIMIZER_GPU_CUH__

/*! \file NEBEnergyMinimizerGPU.cuh
    \brief Defines the interface to GPU kernel drivers used by NEBEnergyMinimizerGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

void sync(const hipStream_t& stream);

//! Kernel driver for zeroing velocities called by NEBEnergyMinimizerGPU
void gpu_neb_zero_v(const hipStream_t& stream, Scalar4* d_vel, unsigned int* d_group_members, unsigned int group_size);

void
gpu_neb_zero_angmom(const hipStream_t& stream, Scalar4* d_angmom, unsigned int* d_group_members, unsigned int group_size);

void gpu_neb_nudge_force(const hipStream_t& stream, const unsigned int N,
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
                                const Scalar k,
                                Scalar* d_sum1,
                                Scalar* d_sum2,
                                Scalar* d_partial_sum1,
                                Scalar* d_partial_sum2,
                                unsigned int reduce_block_size,
                                unsigned int reduce_num_blocks,
                                CachedAllocator& alloc);

//! Kernel driver for summing the potential energy called by NEBEnergyMinimizerGPU
void gpu_neb_compute_sum_pe(const hipStream_t& stream, unsigned int* d_group_members,
                                   unsigned int group_size,
                                   Scalar4* d_net_force,
                                   Scalar* d_sum_pe,
                                   Scalar* d_partial_sum_pe,
                                   unsigned int block_size,
                                   unsigned int num_blocks);

//! Kernel driver for summing over P, vsq, and asq called by NEBEnergyMinimizerGPU
void gpu_neb_compute_sum_all(const hipStream_t& stream, const unsigned int N,
                                    const Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    Scalar* d_sum_all,
                                    Scalar* d_partial_sum_P,
                                    Scalar* d_partial_sum_vsq,
                                    Scalar* d_partial_sum_asq,
                                    unsigned int block_size,
                                    unsigned int num_blocks);

void gpu_neb_compute_sum_all_angular(const hipStream_t& stream, const unsigned int N,
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
                                            unsigned int num_blocks);

//! Kernel driver for updating the velocities called by NEBEnergyMinimizerGPU
void gpu_neb_update_v(const hipStream_t& stream, Scalar4* d_vel,
                             const Scalar3* d_accel,
                             unsigned int* d_group_members,
                             unsigned int group_size,
                             Scalar alpha,
                             Scalar factor_t);

void gpu_neb_update_angmom(const hipStream_t& stream, const Scalar4* d_net_torque,
                                  const Scalar4* d_orientation,
                                  const Scalar3* d_inertia,
                                  Scalar4* d_angmom,
                                  unsigned int* d_group_members,
                                  unsigned int group_size,
                                  Scalar alpha,
                                  Scalar factor_r);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__NEB_ENERGY_MINIMIZER_GPU_CUH__
