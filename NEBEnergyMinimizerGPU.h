// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NEBEnergyMinimizer.h"

#include <memory>

#ifndef __NEB_ENERGY_MINIMIZER_GPU_H__
#define __NEB_ENERGY_MINIMIZER_GPU_H__

/*! \file NEBEnergyMinimizer.h
    \brief Declares a base class for all energy minimization methods
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
class PYBIND11_EXPORT NEBEnergyMinimizerGPU : public NEBEnergyMinimizer
    {
    public:
    //! Constructs the minimizer and associates it with the system
    NEBEnergyMinimizerGPU(std::shared_ptr<SystemDefinition>, Scalar);

    //! Destroys the minimizer
    // virtual ~NEBEnergyMinimizerGPU() { }

    //! Iterates forward one step
    virtual void update(uint64_t timestep);
    virtual bool nudgeForce(uint64_t timestep);

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory

    GPUVector<Scalar> m_neb_partial_sum1;
    GPUVector<Scalar> m_neb_partial_sum2;
    GPUArray<Scalar> m_neb_sum1;
    GPUArray<Scalar> m_neb_sum2;

    GPUVector<Scalar> m_partial_sum1; //!< memory space for partial sum over P and E
    GPUVector<Scalar> m_partial_sum2; //!< memory space for partial sum over vsq
    GPUVector<Scalar> m_partial_sum3; //!< memory space for partial sum over asq
    GPUArray<Scalar> m_sum;           //!< memory space for sum over E
    GPUArray<Scalar> m_sum3;          //!< memory space for the sum over P, vsq, asq

    private:
    //! allocate the memory needed to store partial sums
    void resizeNEBPartialSumArrays(unsigned int N);
    void resizePartialSumArrays();
    };

class PYBIND11_EXPORT NEBHookGPU: public NEBHook
    {
    public:
    NEBHookGPU(NEBEnergyMinimizerGPU* neb);
    NEBHookGPU(std::shared_ptr<NEBEnergyMinimizerGPU> neb);

    void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef);

    void update(uint64_t timestep);

    private:
    NEBEnergyMinimizerGPU* m_neb;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    };

namespace detail
    {
    void export_NEBEnergyMinimizerGPU(pybind11::module& m);
    void export_NEBHookGPU(pybind11::module& m);
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __NEB_ENERGY_MINIMIZER_GPU_H__
