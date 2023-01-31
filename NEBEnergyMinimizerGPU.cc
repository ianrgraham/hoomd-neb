// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NEBEnergyMinimizerGPU.h"
#include "NEBEnergyMinimizerGPU.cuh"

using namespace std;

/*! \file NEBEnergyMinimizerGPU.h
    \brief Contains code for the NEBEnergyMinimizerGPU class
*/

namespace hoomd
    {
namespace md
    {

NEBHookGPU::NEBHookGPU(NEBEnergyMinimizerGPU* neb) : m_neb(neb)
    {
    setSystemDefinition(m_neb->getSystemDefinition());
    }

NEBHookGPU::NEBHookGPU(std::shared_ptr<NEBEnergyMinimizerGPU> neb)
    {
    setSystemDefinition(m_neb->getSystemDefinition());
    }

void NEBHookGPU::update(uint64_t timestep)
    {
    m_neb->resizeBuffers();

    // Arrive at the start of the integration step to synchronize the nudging of forces
    m_neb->m_do_integration = true;
    
    if (m_neb->m_nudge)
        {
        kernel::sync(m_exec_conf->getStream());
        m_neb->arriveAndWaitAtBarriers();

        m_neb->m_do_integration = m_neb->nudgeForce(timestep);
        kernel::sync(m_exec_conf->getStream());
        m_neb->arriveAndWaitAtBarriers();
        }
    }

void NEBHookGPU::setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef)
    {
    m_exec_conf = sysdef->getParticleData()->getExecConf();
    }

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param dt Default step size
*/
NEBEnergyMinimizerGPU::NEBEnergyMinimizerGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    : NEBEnergyMinimizer(sysdef, dt), m_block_size(256)
    {

    // halfstep hook should be setup by NEBEnergyMinimizer
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("NEBEnergyMinimizerGPU requires a GPU device.");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);
    GPUArray<Scalar> sum3(3, m_exec_conf);
    m_sum3.swap(sum3);

    GPUArray<Scalar> neb_sum1(1, m_exec_conf);
    m_neb_sum1.swap(neb_sum1);
    GPUArray<Scalar> neb_sum2(1, m_exec_conf);
    m_neb_sum2.swap(neb_sum2);

    // initialize the partial sum arrays
    m_partial_sum1 = GPUVector<Scalar>(m_exec_conf);
    m_partial_sum2 = GPUVector<Scalar>(m_exec_conf);
    m_partial_sum3 = GPUVector<Scalar>(m_exec_conf);

    m_neb_partial_sum1 = GPUVector<Scalar>(m_exec_conf);
    m_neb_partial_sum2 = GPUVector<Scalar>(m_exec_conf);

    reset();

    auto hook = std::make_shared<NEBHookGPU>(this);
    setHalfStepHook(hook);
    }

bool NEBEnergyMinimizerGPU::nudgeForce(uint64_t timestep)
    {

    if (!isDynamicNode())
        {
        return false;
        }
    auto N = m_pdata->getN();
    resizeNEBPartialSumArrays(N);

    // get the neighbor positions and tags to assess the configuration space tangent vector
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar4>& pos = m_pdata->getPositions();
    const GlobalArray<unsigned int>& tags = m_pdata->getTags();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(pos, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tags(tags, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_tangent(m_tangent_buffer, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_left_disp(m_left_disp_buffer, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_right_disp(m_right_disp_buffer, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar> d_neb_partial_sum1(m_neb_partial_sum1, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_neb_partial_sum2(m_neb_partial_sum2, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_neb_sum1(m_neb_sum1, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_neb_sum2(m_neb_sum2, access_location::device, access_mode::overwrite);

    // get info from left and right minimizers
    auto left_pdata = m_left_minimizer->get()->getParticleData();
    auto right_pdata = m_right_minimizer->get()->getParticleData();

    const GlobalArray<Scalar4>& left_pos = left_pdata->getPositions();
    const GlobalArray<Scalar4>& right_pos = right_pdata->getPositions();

    ArrayHandle<Scalar4> d_left_pos(left_pos, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_right_pos(right_pos, access_location::device, access_mode::read);

    const GlobalArray<unsigned int>& left_rtags = left_pdata->getRTags();
    const GlobalArray<unsigned int>& right_rtags = right_pdata->getRTags();

    ArrayHandle<unsigned int> d_left_rtags(left_rtags, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_right_rtags(right_rtags, access_location::device, access_mode::read);

    auto global_box = m_pdata->getGlobalBox();

    auto num_block = N / m_block_size + 1;
    kernel::gpu_neb_nudge_force(m_exec_conf->getStream(), N,
                        global_box,
                        d_net_force.data,
                        d_pos.data,
                        d_tags.data,
                        d_tangent.data,
                        d_left_disp.data,
                        d_right_disp.data,
                        d_left_pos.data,
                        d_right_pos.data,
                        d_left_rtags.data,
                        d_right_rtags.data,
                        m_k,
                        d_neb_sum1.data,
                        d_neb_sum2.data,
                        d_neb_partial_sum1.data,
                        d_neb_partial_sum2.data,
                        m_block_size,
                        num_block,
                        m_exec_conf->getCachedAllocator());
    
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    return true;
    }

void NEBEnergyMinimizerGPU::resizeNEBPartialSumArrays(unsigned int N)
    {
    unsigned int num_blocks = N / m_block_size + 1;
    if (num_blocks != m_neb_partial_sum1.size())
        {
        m_neb_partial_sum1.resize(num_blocks);
        m_neb_partial_sum2.resize(num_blocks);
        }
    }

/*
 * Update the size of the memory buffers to store the partial sums, if needed.
 */
void NEBEnergyMinimizerGPU::resizePartialSumArrays()
    {
    // initialize the partial sum arrays
    unsigned int num_blocks = 0;
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getNumMembers();
        num_blocks = std::max(num_blocks, group_size);
        }

    num_blocks = num_blocks / m_block_size + 1;

    if (num_blocks != m_partial_sum1.size())
        {
        m_partial_sum1.resize(num_blocks);
        m_partial_sum2.resize(num_blocks);
        m_partial_sum3.resize(num_blocks);
        }
    }

/*! \param timesteps is the iteration number
 */
void NEBEnergyMinimizerGPU::update(uint64_t timestep)
    {
    Integrator::update(timestep);
    // if (m_converged)
    //     return;

    IntegratorTwoStep::update(timestep);

    Scalar Pt(0.0); // translational power
    Scalar Pr(0.0); // rotational power
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar energy(0.0);
    Scalar tnorm(0.0);
    Scalar wnorm(0.0);

    // update partial sum memory space if needed
    resizePartialSumArrays();

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps)/Scalar(group_size);
    unsigned int total_group_size = 0;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getNumMembers();
        total_group_size += group_size;

        ArrayHandle<unsigned int> d_index_array(current_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

            {
            ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<Scalar> d_partial_sumE(m_partial_sum1,
                                               access_location::device,
                                               access_mode::overwrite);
            ArrayHandle<Scalar> d_sumE(m_sum, access_location::device, access_mode::overwrite);

            unsigned int num_blocks = group_size / m_block_size + 1;
            kernel::gpu_neb_compute_sum_pe(m_exec_conf->getStream(), d_index_array.data,
                                            group_size,
                                            d_net_force.data,
                                            d_sumE.data,
                                            d_partial_sumE.data,
                                            m_block_size,
                                            num_blocks);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        ArrayHandle<Scalar> h_sumE(m_sum, access_location::host, access_mode::read);
        energy += h_sumE.data[0];
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        throw runtime_error("NEB minimizer does not support MPI domain decomposition");
        MPI_Allreduce(MPI_IN_PLACE,
                      &energy,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                      &total_group_size,
                      1,
                      MPI_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    m_energy_total = energy;
    energy /= (Scalar)total_group_size;

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000) * m_etol;
        }

    // sum P, vnorm, fnorm

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                access_location::device,
                                access_mode::readwrite);

    if (m_do_integration)
        {

#ifdef ENABLE_MPI
        bool aniso = false;
#endif

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

            unsigned int group_size = current_group->getNumMembers();
            ArrayHandle<unsigned int> d_index_array(current_group->getIndexArray(),
                                                    access_location::device,
                                                    access_mode::read);

                {
                ArrayHandle<Scalar> d_partial_sum_P(m_partial_sum1,
                                                    access_location::device,
                                                    access_mode::overwrite);
                ArrayHandle<Scalar> d_partial_sum_vsq(m_partial_sum2,
                                                    access_location::device,
                                                    access_mode::overwrite);
                ArrayHandle<Scalar> d_partial_sum_fsq(m_partial_sum3,
                                                    access_location::device,
                                                    access_mode::overwrite);
                ArrayHandle<Scalar> d_sum(m_sum3, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                        access_location::device,
                                        access_mode::read);
                ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                            access_location::device,
                                            access_mode::read);

                unsigned int num_blocks = group_size / m_block_size + 1;

                kernel::gpu_neb_compute_sum_all(m_exec_conf->getStream(), m_pdata->getN(),
                                                d_vel.data,
                                                d_accel.data,
                                                d_index_array.data,
                                                group_size,
                                                d_sum.data,
                                                d_partial_sum_P.data,
                                                d_partial_sum_vsq.data,
                                                d_partial_sum_fsq.data,
                                                m_block_size,
                                                num_blocks);

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

            ArrayHandle<Scalar> h_sum(m_sum3, access_location::host, access_mode::read);
            Pt += h_sum.data[0];
            vnorm += h_sum.data[1];
            fnorm += h_sum.data[2];

            if ((*method)->getAnisotropic())
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
#ifdef ENABLE_MPI
                aniso = true;
#endif

                    {
                    ArrayHandle<Scalar> d_partial_sum_Pr(m_partial_sum1,
                                                        access_location::device,
                                                        access_mode::overwrite);
                    ArrayHandle<Scalar> d_partial_sum_wnorm(m_partial_sum2,
                                                            access_location::device,
                                                            access_mode::overwrite);
                    ArrayHandle<Scalar> d_partial_sum_tsq(m_partial_sum3,
                                                        access_location::device,
                                                        access_mode::overwrite);
                    ArrayHandle<Scalar> d_sum(m_sum3, access_location::device, access_mode::overwrite);
                    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                                    access_location::device,
                                                    access_mode::read);
                    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                                access_location::device,
                                                access_mode::read);
                    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                                    access_location::device,
                                                    access_mode::read);
                    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                                access_location::device,
                                                access_mode::read);

                    unsigned int num_blocks = group_size / m_block_size + 1;

                    kernel::gpu_neb_compute_sum_all_angular(m_exec_conf->getStream(), m_pdata->getN(),
                                                            d_orientation.data,
                                                            d_inertia.data,
                                                            d_angmom.data,
                                                            d_net_torque.data,
                                                            d_index_array.data,
                                                            group_size,
                                                            d_sum.data,
                                                            d_partial_sum_Pr.data,
                                                            d_partial_sum_wnorm.data,
                                                            d_partial_sum_tsq.data,
                                                            m_block_size,
                                                            num_blocks);

                    if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }
                ArrayHandle<Scalar> h_sum(m_sum3, access_location::host, access_mode::read);
                Pr += h_sum.data[0];
                wnorm += h_sum.data[1];
                tnorm += h_sum.data[2];
                }
            }

#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            throw runtime_error("NEB minimizer does not support MPI domain decomposition");
            MPI_Allreduce(MPI_IN_PLACE,
                        &fnorm,
                        1,
                        MPI_HOOMD_SCALAR,
                        MPI_SUM,
                        m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                        &vnorm,
                        1,
                        MPI_HOOMD_SCALAR,
                        MPI_SUM,
                        m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                        &Pt,
                        1,
                        MPI_HOOMD_SCALAR,
                        MPI_SUM,
                        m_exec_conf->getMPICommunicator());

            if (aniso)
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
                MPI_Allreduce(MPI_IN_PLACE,
                            &tnorm,
                            1,
                            MPI_HOOMD_SCALAR,
                            MPI_SUM,
                            m_exec_conf->getMPICommunicator());
                MPI_Allreduce(MPI_IN_PLACE,
                            &wnorm,
                            1,
                            MPI_HOOMD_SCALAR,
                            MPI_SUM,
                            m_exec_conf->getMPICommunicator());
                MPI_Allreduce(MPI_IN_PLACE,
                            &Pr,
                            1,
                            MPI_HOOMD_SCALAR,
                            MPI_SUM,
                            m_exec_conf->getMPICommunicator());
                }
            }
#endif

        vnorm = sqrt(vnorm);
        fnorm = sqrt(fnorm);
        wnorm = sqrt(wnorm);
        tnorm = sqrt(tnorm);

        unsigned int ndof = m_sysdef->getNDimensions() * total_group_size;
        m_exec_conf->msg->notice(10) << "NEB fnorm " << fnorm << " tnorm " << tnorm << " delta_E "
                                    << energy - m_old_energy << std::endl;
        m_exec_conf->msg->notice(10) << "NEB vnorm " << vnorm << " tnorm " << wnorm << std::endl;
        m_exec_conf->msg->notice(10) << "NEB Pt " << Pt << " Pr " << Pr << std::endl;

        if ((fnorm / sqrt(Scalar(ndof)) < m_ftol && wnorm / sqrt(Scalar(ndof)) < m_wtol
            && fabs(energy - m_old_energy) < m_etol)
            && m_n_since_start >= m_run_minsteps)
            {
            m_converged = true;
            m_exec_conf->msg->notice(4) << "NEB converged in timestep " << timestep << std::endl;
            return;
            }

        // update velocities

        Scalar factor_t;
        if (fabs(fnorm) > 0)
            factor_t = m_alpha * vnorm / fnorm;
        else
            factor_t = 1.0;

        Scalar factor_r = 0.0;

        if (fabs(tnorm) > 0)
            factor_r = m_alpha * wnorm / tnorm;
        else
            factor_r = 1.0;

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

            unsigned int group_size = current_group->getNumMembers();
            ArrayHandle<unsigned int> d_index_array(current_group->getIndexArray(),
                                                    access_location::device,
                                                    access_mode::read);

            // ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
            //                            access_location::device,
            //                            access_mode::readwrite);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                        access_location::device,
                                        access_mode::read);

            kernel::gpu_neb_update_v(m_exec_conf->getStream(), d_vel.data,
                                    d_accel.data,
                                    d_index_array.data,
                                    group_size,
                                    m_alpha,
                                    factor_t);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            if ((*method)->getAnisotropic())
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
                ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                                access_location::device,
                                                access_mode::read);
                ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                                access_location::device,
                                                access_mode::read);
                ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                            access_location::device,
                                            access_mode::readwrite);
                ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                            access_location::device,
                                            access_mode::read);

                kernel::gpu_neb_update_angmom(m_exec_conf->getStream(), d_net_torque.data,
                                            d_orientation.data,
                                            d_inertia.data,
                                            d_angmom.data,
                                            d_index_array.data,
                                            group_size,
                                            m_alpha,
                                            factor_r);

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }
            }

        Scalar P = Pt + Pr;

        if (P > Scalar(0.0))
            {
            m_n_since_negative++;
            if (m_n_since_negative > m_nmin)
                {
                IntegratorTwoStep::setDeltaT(std::min(m_deltaT * m_finc, m_deltaT_max));
                m_alpha *= m_falpha;
                }
            }
        else if (P <= Scalar(0.0))
            {
            IntegratorTwoStep::setDeltaT(m_deltaT * m_fdec);
            m_alpha = m_alpha_start;
            m_n_since_negative = 0;
            m_exec_conf->msg->notice(6) << "NEB zero velocities" << std::endl;

            for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
                {
                std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
                unsigned int group_size = current_group->getNumMembers();
                ArrayHandle<unsigned int> d_index_array(current_group->getIndexArray(),
                                                        access_location::device,
                                                        access_mode::read);

                ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                        access_location::device,
                                        access_mode::readwrite);

                kernel::gpu_neb_zero_v(m_exec_conf->getStream(), d_vel.data, d_index_array.data, group_size);
                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                if ((*method)->getAnisotropic())
                    {
                    throw std::runtime_error("NEB does not support anisotropic integration");
                    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                                access_location::device,
                                                access_mode::readwrite);
                    kernel::gpu_neb_zero_angmom(m_exec_conf->getStream(), d_angmom.data, d_index_array.data, group_size);
                    if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }
                }
            }

        m_n_since_start++;
        }
    else
        {  // zero velocities
        m_exec_conf->msg->notice(6) << "NEB zero velocities" << std::endl;

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getNumMembers();
            ArrayHandle<unsigned int> d_index_array(current_group->getIndexArray(),
                                                    access_location::device,
                                                    access_mode::read);

            kernel::gpu_neb_zero_v(m_exec_conf->getStream(), d_vel.data, d_index_array.data, group_size);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            if ((*method)->getAnisotropic())
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
                ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                            access_location::device,
                                            access_mode::readwrite);
                kernel::gpu_neb_zero_angmom(m_exec_conf->getStream(), d_angmom.data, d_index_array.data, group_size);
                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }
            }
        }
    m_old_energy = energy;
    }

namespace detail
    {
void export_NEBEnergyMinimizerGPU(pybind11::module& m)
    {
    pybind11::class_<NEBEnergyMinimizerGPU,
                     NEBEnergyMinimizer,
                     std::shared_ptr<NEBEnergyMinimizerGPU>>(m, "NEBEnergyMinimizerGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>());
    }

void export_NEBHookGPU(pybind11::module& m)
    {
    pybind11::class_<NEBHookGPU, NEBHook, std::shared_ptr<NEBHookGPU>>(
        m,
        "NEBHookGPU")
        .def(pybind11::init<std::shared_ptr<NEBEnergyMinimizerGPU>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
