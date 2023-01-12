// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NEBEnergyMinimizer.h"

using namespace std;

/*! \file NEBEnergyMinimizer.h
    \brief Contains code for the NEBEnergyMinimizer class
*/

namespace hoomd
    {
namespace md
    {

NEBHook::NEBHook() {}

NEBHook::NEBHook(NEBEnergyMinimizer* neb) : m_neb(neb)
    {
    setSystemDefinition(m_neb->getSystemDefinition());
    }

NEBHook::NEBHook(std::shared_ptr<NEBEnergyMinimizer> neb)
    {
    setSystemDefinition(m_neb->getSystemDefinition());
    }

void NEBHook::update(uint64_t timestep)
    {
    m_neb->resizeBuffers();

    // Arrive at the start of the integration step to synchronize the nudging of forces
    m_neb->m_do_integration = true;
    if (m_neb->m_nudge)
        {
        m_neb->arriveAndWaitAtBarriers();
        // {
        //     pybind11::gil_scoped_acquire acquire;
        //     std::cout << timestep << " " << m_k << ": " << "compute nudge force" << std::endl;
        // }

        m_neb->m_do_integration = m_neb->nudgeForce(timestep);

        m_neb->arriveAndWaitAtBarriers();
        // {
        //     pybind11::gil_scoped_acquire acquire;
        //     std::cout << timestep << " " << m_k << ": " << do_integration << std::endl;
        // }
        }
    }

void NEBHook::setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef)
    {
    }

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param dt maximum step size

    \post The method is constructed with the given particle data.
*/
NEBEnergyMinimizer::NEBEnergyMinimizer(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    : IntegratorTwoStep(sysdef, dt), m_nmin(5), m_finc(Scalar(1.1)), m_fdec(Scalar(0.5)),
      m_alpha_start(Scalar(0.1)), m_falpha(Scalar(0.99)), m_ftol(Scalar(1e-1)),
      m_wtol(Scalar(1e-1)), m_etol(Scalar(1e-3)), m_energy_total(Scalar(0.0)),
      m_old_energy(Scalar(0.0)), m_deltaT_max(dt), m_deltaT_set(dt / Scalar(10.0)),
      m_run_minsteps(10), m_k(1.0), m_nudge(true), m_do_integration(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing NEBEnergyMinimizer" << endl;

    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    reset();

    // allocate buffers for nudged force computation
    auto N = m_pdata->getN();
    GlobalArray<Scalar3> tangent_buffer(N, m_exec_conf);
    m_tangent_buffer.swap(tangent_buffer);
    GlobalArray<Scalar3> left_disp_buffer(N, m_exec_conf);
    m_left_disp_buffer.swap(left_disp_buffer);
    GlobalArray<Scalar3> right_disp_buffer(N, m_exec_conf);
    m_right_disp_buffer.swap(right_disp_buffer);

    auto hook = std::make_shared<NEBHook>(this);
    setHalfStepHook(hook);
    }

NEBEnergyMinimizer::~NEBEnergyMinimizer()
    {
    m_exec_conf->msg->notice(5) << "Destroying NEBEnergyMinimizer" << endl;
    }

/*! \param finc is the new fractional increase to set
 */
void NEBEnergyMinimizer::setFinc(Scalar finc)
    {
    if (!(finc > 1.0))
        {
        m_exec_conf->msg->error()
            << "integrate.mode_minimize_neb: fractional increase in timestep should be > 1"
            << endl;
        throw runtime_error("Error setting parameters for NEBEnergyMinimizer");
        }
    m_finc = finc;
    }

/*! \param fdec is the new fractional decrease to set
 */
void NEBEnergyMinimizer::setFdec(Scalar fdec)
    {
    if (!(fdec < 1.0 && fdec >= 0.0))
        {
        throw runtime_error("fdec must be in the range [0,1).");
        }
    m_fdec = fdec;
    }

/*! \param alpha_start is the new initial coupling parameter to set

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void NEBEnergyMinimizer::setAlphaStart(Scalar alpha_start)
    {
    if (!(alpha_start < 1.0 && alpha_start > 0.0))
        {
        m_exec_conf->msg->error()
            << "integrate.mode_minimize_neb: alpha_start should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for NEBEnergyMinimizer");
        }
    m_alpha_start = alpha_start;
    }

/*! \param falpha is the fractional decrease in alpha upon finding a valid search direction

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void NEBEnergyMinimizer::setFalpha(Scalar falpha)
    {
    if (!(falpha < 1.0 && falpha > 0.0))
        {
        m_exec_conf->msg->error()
            << "integrate.mode_minimize_neb: falpha should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for NEBEnergyMinimizer");
        }
    m_falpha = falpha;
    }

void NEBEnergyMinimizer::reset()
    {
    m_converged = false;
    m_n_since_negative = m_nmin + 1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;
    m_energy_total = 0.0;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);

    unsigned int n = m_pdata->getN();
    for (unsigned int i = 0; i < n; i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        h_angmom.data[i] = make_scalar4(0, 0, 0, 0);
        }

    setDeltaT(m_deltaT_set);
    }

void NEBEnergyMinimizer::resizeBuffers()
    {
    auto N = m_pdata->getN();
    if (N != m_tangent_buffer.getNumElements())
        {
        m_tangent_buffer.resize(N);
        m_left_disp_buffer.resize(N);
        m_right_disp_buffer.resize(N);
        }
    }

bool NEBEnergyMinimizer::nudgeForce(uint64_t timestep)
    {

    if (!isDynamicNode())
        {
        return false;
        }

    // get the neighbor positions and tags to assess the configuration space tangent vector
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar4>& pos = m_pdata->getPositions();
    const GlobalArray<unsigned int>& tags = m_pdata->getTags();

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tags(tags, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_tangent(m_tangent_buffer, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_left_disp(m_left_disp_buffer, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_right_disp(m_right_disp_buffer, access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar3> h_tangent_force(m_tangent_force_buffer, access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar3> h_norm_force(m_norm_force_buffer, access_location::host, access_mode::readwrite);

    Scalar left_norm = 0.0;
    Scalar right_norm = 0.0;
    Scalar tangent_norm = 0.0;

    Scalar tan_force = 0.0;
    Scalar norm_force = 0.0;

    // get info from left and right minimizers
    auto left_pdata = m_left_minimizer->get()->getParticleData();
    auto right_pdata = m_right_minimizer->get()->getParticleData();

    GlobalArray<Scalar4> left_pos = left_pdata->getPositions();
    GlobalArray<Scalar4> right_pos = right_pdata->getPositions();

    ArrayHandle<Scalar4> h_left_pos(left_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_right_pos(right_pos, access_location::host, access_mode::read);

    GlobalArray<unsigned int> left_rtags = left_pdata->getRTags();
    GlobalArray<unsigned int> right_rtags = right_pdata->getRTags();

    ArrayHandle<unsigned int> h_left_rtags(left_rtags, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_right_rtags(right_rtags, access_location::host, access_mode::read);

    auto global_box = m_pdata->getGlobalBox();

    // build displacement vectors needed to find tangent
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int tag = h_tags.data[i];
        unsigned int left_idx = h_left_rtags.data[tag];
        unsigned int right_idx = h_right_rtags.data[tag];

        Scalar4 pi = h_pos.data[i];
        Scalar4 pl = h_left_pos.data[left_idx];
        Scalar4 pr = h_right_pos.data[right_idx];

        Scalar3 left_disp = global_box.minImage(
            make_scalar3(pi.x - pl.x, pi.y - pl.y, pi.z - pl.z));
        Scalar3 right_disp = global_box.minImage(
            make_scalar3(pr.x - pi.x, pr.y - pi.y, pr.z - pi.z));

        h_left_disp.data[i] = left_disp;
        h_right_disp.data[i] = right_disp;
        left_norm += dot((vec3<Scalar>)left_disp, (vec3<Scalar>)left_disp);
        right_norm += dot((vec3<Scalar>)right_disp, (vec3<Scalar>)right_disp);
        }
    
    left_norm = 1.0 / sqrt(left_norm);
    right_norm = 1.0 / sqrt(right_norm);

    // compute the displacement vector
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        Scalar3 left_disp = h_left_disp.data[i];
        Scalar3 right_disp = h_right_disp.data[i];
        Scalar3 tangent = left_norm * left_disp + right_norm * right_disp;
        h_tangent.data[i] = tangent;
        tangent_norm += dot((vec3<Scalar>)tangent, (vec3<Scalar>)tangent);
        }

    tangent_norm = 1.0 / sqrt(tangent_norm);

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_tangent.data[i] *= tangent_norm;
        }

    // nudge force
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        Scalar4 f = h_net_force.data[i];
        Scalar3 force = make_scalar3(f.x, f.y, f.z);

        Scalar3 tangent = h_tangent.data[i];
        Scalar3 left_disp = h_left_disp.data[i];
        Scalar3 right_disp = h_right_disp.data[i];

        tan_force += dot((vec3<Scalar>)right_disp - (vec3<Scalar>)left_disp, (vec3<Scalar>)tangent);
        norm_force += dot((vec3<Scalar>)force, (vec3<Scalar>)tangent);
        }

    // if (timestep%100 == 0)
    //     {
    //     pybind11::gil_scoped_acquire acquire;
    //     std::cout << "tstep: " << timestep << " tan_force: " << tan_force << " norm_force: " << norm_force << std::endl;
    //     }
    

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        Scalar3 tangent = h_tangent.data[i];
        Scalar3 nudge_force = (tan_force - norm_force) * tangent;
        h_net_force.data[i].x += nudge_force.x;
        h_net_force.data[i].y += nudge_force.y;
        h_net_force.data[i].z += nudge_force.z;
        }

    return true;
    }

/*! \param timesteps is the current timestep
 */
void NEBEnergyMinimizer::update(uint64_t timestep)
    {
    Integrator::update(timestep);
    // if (m_converged)
    //     return;

    IntegratorTwoStep::update(timestep);

    Scalar Pt(0.0); // translational power
    Scalar Pr(0.0); // rotational power
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar tnorm(0.0);
    Scalar wnorm(0.0);

    // Calculate the per-particle potential energy over particles in the group
    Scalar energy(0.0);
        
    unsigned int total_group_size = 0;

        {
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

        // total potential energy
        double pe_total = 0.0;

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getNumMembers();
            total_group_size += group_size;

            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                pe_total += (double)h_net_force.data[j].w;
                }
            }

        m_energy_total = pe_total;

#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            throw runtime_error("NEB minimizer does not support MPI domain decomposition");
            MPI_Allreduce(MPI_IN_PLACE,
                        &pe_total,
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

        energy = pe_total / Scalar(total_group_size);
        }

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000) * m_etol;
        }

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                            access_location::host,
                            access_mode::readwrite);

    if (m_do_integration)
        {

        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                access_location::host,
                                access_mode::readwrite);

#ifdef ENABLE_MPI
        bool aniso = false;
#endif

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getNumMembers();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                Pt += h_accel.data[j].x * h_vel.data[j].x + h_accel.data[j].y * h_vel.data[j].y
                    + h_accel.data[j].z * h_vel.data[j].z;
                fnorm += h_accel.data[j].x * h_accel.data[j].x + h_accel.data[j].y * h_accel.data[j].y
                        + h_accel.data[j].z * h_accel.data[j].z;
                vnorm += h_vel.data[j].x * h_vel.data[j].x + h_vel.data[j].y * h_vel.data[j].y
                        + h_vel.data[j].z * h_vel.data[j].z;
                }

            if ((*method)->getAnisotropic())
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
#ifdef ENABLE_MPI
                aniso = true;
#endif

                ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                                access_location::host,
                                                access_mode::read);
                ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                                access_location::host,
                                                access_mode::read);
                ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                            access_location::host,
                                            access_mode::read);
                ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                            access_location::host,
                                            access_mode::read);

                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);

                    vec3<Scalar> t(h_net_torque.data[j]);
                    quat<Scalar> p(h_angmom.data[j]);
                    quat<Scalar> q(h_orientation.data[j]);
                    vec3<Scalar> I(h_inertia.data[j]);

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
                    Pr += dot(t, s);
                    tnorm += dot(t, t);
                    wnorm += dot(s, s);
                    }
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

        fnorm = sqrt(fnorm);
        vnorm = sqrt(vnorm);

        tnorm = sqrt(tnorm);
        wnorm = sqrt(wnorm);

        unsigned int ndof = m_sysdef->getNDimensions() * total_group_size;
        m_exec_conf->msg->notice(10) << "NEB fnorm " << fnorm << " tnorm " << tnorm << " delta_E "
                                    << energy - m_old_energy << std::endl;
        m_exec_conf->msg->notice(10) << "NEB vnorm " << vnorm << " tnorm " << wnorm << std::endl;
        m_exec_conf->msg->notice(10) << "NEB Pt " << Pt << " Pr " << Pr << std::endl;

        if ((fnorm / sqrt(Scalar(ndof)) < m_ftol && wnorm / sqrt(Scalar(ndof)) < m_wtol
            && fabs(energy - m_old_energy) < m_etol)
            && m_n_since_start >= m_run_minsteps)
            {
            m_exec_conf->msg->notice(4) << "NEB converged in timestep " << timestep << std::endl;
            m_converged = true;
            }
        else
            {
            m_converged = false;
            }

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
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_vel.data[j].x = h_vel.data[j].x * (1.0 - m_alpha) + h_accel.data[j].x * factor_t;
                h_vel.data[j].y = h_vel.data[j].y * (1.0 - m_alpha) + h_accel.data[j].y * factor_t;
                h_vel.data[j].z = h_vel.data[j].z * (1.0 - m_alpha) + h_accel.data[j].z * factor_t;
                }

            if ((*method)->getAnisotropic())
                {
                throw std::runtime_error("NEB does not support anisotropic integration");
                ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                            access_location::host,
                                            access_mode::readwrite);
                ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                                access_location::host,
                                                access_mode::read);
                ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                                access_location::host,
                                                access_mode::read);
                ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                            access_location::host,
                                            access_mode::read);

                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    vec3<Scalar> t(h_net_torque.data[j]);
                    quat<Scalar> p(h_angmom.data[j]);
                    quat<Scalar> q(h_orientation.data[j]);
                    vec3<Scalar> I(h_inertia.data[j]);

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

                    // update angular momentum
                    p = p * Scalar(1.0 - m_alpha) + Scalar(2.0) * q * t * factor_r;
                    h_angmom.data[j] = quat_to_scalar4(p);
                    }
                }
            }

        // A simply naive measure is to sum up the power coming from translational and rotational
        // motions, more sophisticated measure can be devised later
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
                for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                    {
                    unsigned int j = current_group->getMemberIndex(group_idx);
                    h_vel.data[j].x = Scalar(0.0);
                    h_vel.data[j].y = Scalar(0.0);
                    h_vel.data[j].z = Scalar(0.0);
                    }

                if ((*method)->getAnisotropic())
                    {
                    throw std::runtime_error("NEB does not support anisotropic integration");
                    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                                access_location::host,
                                                access_mode::readwrite);
                    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                        {
                        unsigned int j = current_group->getMemberIndex(group_idx);
                        h_angmom.data[j] = make_scalar4(0, 0, 0, 0);
                        }
                    }
                }
            }
        m_n_since_start++;
        }
    else
        {
        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getNumMembers();
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = current_group->getMemberIndex(group_idx);
                h_vel.data[j].x = 0.0;
                h_vel.data[j].y = 0.0;
                h_vel.data[j].z = 0.0;
                }
            }
        }
    m_old_energy = energy;
    }

namespace detail
    {
void export_NEBEnergyMinimizer(pybind11::module& m)
    {
    pybind11::class_<NEBEnergyMinimizer, IntegratorTwoStep, std::shared_ptr<NEBEnergyMinimizer>>(
        m,
        "NEBEnergyMinimizer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("reset", &NEBEnergyMinimizer::reset)
        .def_property_readonly("converged", &NEBEnergyMinimizer::hasConverged)
        .def_property_readonly("energy", &NEBEnergyMinimizer::getEnergy)
        .def_property("min_steps_adapt",
                      &NEBEnergyMinimizer::getNmin,
                      &NEBEnergyMinimizer::setNmin)
        .def_property("finc_dt", &NEBEnergyMinimizer::getFinc, &NEBEnergyMinimizer::setFinc)
        .def_property("fdec_dt", &NEBEnergyMinimizer::getFdec, &NEBEnergyMinimizer::setFdec)
        .def_property("alpha_start",
                      &NEBEnergyMinimizer::getAlphaStart,
                      &NEBEnergyMinimizer::setAlphaStart)
        .def_property("fdec_alpha",
                      &NEBEnergyMinimizer::getFalpha,
                      &NEBEnergyMinimizer::setFalpha)
        .def_property("force_tol", &NEBEnergyMinimizer::getFtol, &NEBEnergyMinimizer::setFtol)
        .def_property("angmom_tol", &NEBEnergyMinimizer::getWtol, &NEBEnergyMinimizer::setWtol)
        .def_property("energy_tol", &NEBEnergyMinimizer::getEtol, &NEBEnergyMinimizer::setEtol)
        .def_property("min_steps_conv",
                      &NEBEnergyMinimizer::getMinSteps,
                      &NEBEnergyMinimizer::setMinSteps)
        .def_property("k",
                      &NEBEnergyMinimizer::getK,
                      &NEBEnergyMinimizer::setK)
        .def("coupleLeft", &NEBEnergyMinimizer::coupleLeft)
        .def("coupleRight", &NEBEnergyMinimizer::coupleRight)
        .def("uncoupleLeft", &NEBEnergyMinimizer::uncoupleLeft)
        .def("uncoupleRight", &NEBEnergyMinimizer::uncoupleRight)
        .def("setNudge", &NEBEnergyMinimizer::setNudge)
        .def("getNudge", &NEBEnergyMinimizer::getNudge);
    }

void export_NEBHook(pybind11::module& m)
    {
    pybind11::class_<NEBHook, HalfStepHook, std::shared_ptr<NEBHook>>(
        m,
        "NEBHook")
        .def(pybind11::init<std::shared_ptr<NEBEnergyMinimizer>>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
