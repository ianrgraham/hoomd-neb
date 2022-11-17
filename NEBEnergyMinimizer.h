// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/IntegratorTwoStep.h"

#include <memory>
#include <barrier>

#ifndef __NEB_ENERGY_MINIMIZER_H__
#define __NEB_ENERGY_MINIMIZER_H__

/*! \file NEBEnergyMinimizer.h
    \brief Declares the NEB energy minimizer class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Finds the nearest basin in the potential energy landscape
/*! \b Overview

    \ingroup updaters
*/
class PYBIND11_EXPORT NEBEnergyMinimizer : public IntegratorTwoStep, public std::enable_shared_from_this<NEBEnergyMinimizer>
    {
    public:
    //! Constructs the minimizer and associates it with the system
    NEBEnergyMinimizer(std::shared_ptr<SystemDefinition>, Scalar);
    virtual ~NEBEnergyMinimizer();

    //! Reset the minimization
    virtual void reset();

    //! Perform one minimization iteration
    virtual void update(uint64_t timestep);

    //! Return whether or not the minimization has converged
    bool hasConverged() const
        {
        return m_converged;
        }

    //! Return the potential energy after the last iteration
    Scalar getEnergy() const
        {
        if (m_was_reset)
            {
            m_exec_conf->msg->warning()
                << "NEB has just been initialized. Return energy==0." << std::endl;
            return Scalar(0.0);
            }

        return m_energy_total;
        }

    //! Set the minimum number of steps for which the search direction must be bad before finding a
    //! new direction
    /*! \param nmin is the new nmin to set
     */
    void setNmin(unsigned int nmin)
        {
        m_nmin = nmin;
        }

    //! Get the minimum number of steps for which the search direction must be
    //! bad before finding a new direction
    unsigned int getNmin()
        {
        return m_nmin;
        }

    //! Set the fractional increase in the timestep upon a valid search direction
    void setFinc(Scalar finc);

    //! get the fractional increase in the timestep upon a valid search direction
    Scalar getFinc()
        {
        return m_finc;
        }

    //! Set the fractional decrease in the timestep upon system energy increasing
    void setFdec(Scalar fdec);

    //! Get the fractional decrease in the timestep upon system energy increasing
    Scalar getFdec()
        {
        return m_fdec;
        }

    //! Set the relative strength of the coupling between the "f dot v" vs the "v" term
    void setAlphaStart(Scalar alpha0);

    //! Get the relative strength of the coupling between the "f dot v" vs the "v" term
    Scalar getAlphaStart()
        {
        return m_alpha_start;
        }

    //! Set the fractional decrease in alpha upon finding a valid search direction
    void setFalpha(Scalar falpha);

    //! Get the fractional decrease in alpha upon finding a valid search direction
    Scalar getFalpha()
        {
        return m_falpha;
        }

    //! Set the stopping criterion based on the total force on all particles in the system
    /*! \param ftol is the new force tolerance to set
     */
    void setFtol(Scalar ftol)
        {
        m_ftol = ftol;
        }

    //! get the stopping criterion based on the total force on all particles in the system
    Scalar getFtol()
        {
        return m_ftol;
        }

    //! Set the stopping criterion based on the total torque on all particles in the system
    /*! \param wtol is the new torque tolerance to set
     */
    void setWtol(Scalar wtol)
        {
        m_wtol = wtol;
        }

    //! Get the stopping criterion based on the total torque on all particles in the system
    Scalar getWtol()
        {
        return m_wtol;
        }

    //! Set the stopping criterion based on the change in energy between successive iterations
    /*! \param etol is the new energy tolerance to set
     */
    void setEtol(Scalar etol)
        {
        m_etol = etol;
        }

    //! Get the stopping criterion based on the change in energy between successive iterations
    Scalar getEtol()
        {
        return m_etol;
        }

    Scalar getK()
        {
        return m_k;
        }

    void setK(Scalar k)
        {
        m_k = k;
        }

    //! Set the a minimum number of steps before the other stopping criteria will be evaluated
    /*! \param steps is the minimum number of steps (attempts) that will be made
     */
    void setMinSteps(unsigned int steps)
        {
        m_run_minsteps = steps;
        }

    //! Get the minimum number of steps before the other stopping criteria will be evaluated
    unsigned int getMinSteps()
        {
        return m_run_minsteps;
        }

    void coupleRight(std::shared_ptr<NEBEnergyMinimizer> minimizer, bool update_other = true)
        {
        m_right_minimizer = minimizer;
        if (update_other)
            minimizer->coupleLeft(shared_from_this(), false);
        updateBarrier();
        }

    void uncoupleRight(bool update_other = true)
        {
        if (update_other && m_right_minimizer)
            m_right_minimizer->get()->uncoupleLeft(false);
        m_right_minimizer.reset();
        updateBarrier();
        }

    void coupleLeft(std::shared_ptr<NEBEnergyMinimizer> minimizer, bool update_other = true)
        {
        m_left_minimizer = minimizer;
        if (update_other)
            minimizer->coupleRight(shared_from_this(), false);
        updateBarrier();
        }

    void uncoupleLeft(bool update_other = true)
        {
        if (update_other && m_left_minimizer)
            m_left_minimizer->get()->uncoupleRight(false);
        m_left_minimizer.reset();
        updateBarrier();
        }

    int getNeighbors() const {
        return (m_left_minimizer ? 1 : 0) + (m_right_minimizer ? 1 : 0);
    }

    void updateBarrier() {
        auto n = getNeighbors() + 1;
        if (n == 1)
            m_barrier = std::nullopt;
        else
            m_barrier = std::make_shared<std::barrier<std::__empty_completion>>(n);
    }

    std::shared_ptr<hoomd::SystemDefinition> getSystemDefinition() const
        {
        return m_sysdef;
        }

    void arriveAtBarrier() {
        if (m_barrier)
            m_barrier->get()->arrive();
    }

    void arriveAndWaitAtBarriers() {
        if (m_right_minimizer)
            m_right_minimizer->get()->arriveAtBarrier();
        if (m_left_minimizer)
            m_left_minimizer->get()->arriveAtBarrier();
        if (m_barrier)
            m_barrier->get()->arrive_and_wait();
    }

    std::shared_ptr<ParticleData> getParticleData() const
        {
        return m_pdata;
        }

    bool isDynamicNode() const
        {
        return m_right_minimizer.has_value() && m_left_minimizer.has_value();
        }

    void resizeBuffers();

    bool nudgeForce();

    protected:
    //! Function to create the underlying integrator
    unsigned int m_nmin; //!< minimum number of consecutive successful search directions before
                         //!< modifying alpha
    unsigned int
        m_n_since_negative;       //!< counts the number of consecutive successful search directions
    unsigned int m_n_since_start; //!< counts the number of consecutive search attempts
    Scalar m_finc;                //!< fractional increase in timestep upon successful search
    Scalar m_fdec;                //!< fractional decrease in timestep upon unsuccessful search
    Scalar m_alpha;               //!< relative coupling strength between alpha
    Scalar m_alpha_start;         //!< starting value of alpha
    Scalar m_falpha;              //!< fraction to rescale alpha on successful search direction
    Scalar m_ftol;                //!< stopping tolerance based on total force
    Scalar m_wtol;                //!< stopping tolerance based on total torque
    Scalar m_etol;                //!< stopping tolerance based on the chance in energy
    Scalar m_energy_total;        //!< Total energy of all integrator groups
    Scalar m_old_energy;          //!< energy from the previous iteration
    bool m_converged;             //!< whether the minimization has converged
    Scalar m_deltaT_max;          //!< maximum timesteps after rescaling (set by user)
    Scalar m_deltaT_set;          //!< the initial timestep
    unsigned int m_run_minsteps;  //!< A minimum number of search attempts the search will use
    bool m_was_reset;             //!< whether or not the minimizer was reset
    Scalar m_k;                   //!< spring constant for the elastic band force

    std::optional<std::shared_ptr<std::barrier<std::__empty_completion>>> m_barrier; //!< Condition variable to signal state changes
    std::optional<std::shared_ptr<NEBEnergyMinimizer>> m_left_minimizer; //!< Optional left minimizer
    std::optional<std::shared_ptr<NEBEnergyMinimizer>> m_right_minimizer; //!< Optional right minimizer

    GlobalArray<Scalar3> m_tangent_buffer; //!< Buffer to store the tangent vector
    GlobalArray<Scalar3> m_left_disp_buffer; //!< Buffer to store the left displacement vector
    GlobalArray<Scalar3> m_right_disp_buffer; //!< Buffer to store the right displacement vector
    // GlobalArray<Scalar3> m_tangent_force_buffer; //!< Buffer to store the tangent force vector
    // GlobalArray<Scalar3> m_norm_force_buffer; //!< Buffer to store the normal force vector

    private:
    };

namespace detail
    {
    void export_NEBEnergyMinimizer(pybind11::module& m);
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __NEB_ENERGY_MINIMIZER_H__
