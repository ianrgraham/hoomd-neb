// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause
// License.

// Include the defined classes that are to be exported to python
#include "NEBEnergyMinimizer.h"

#include <pybind11/pybind11.h>

#ifdef ENABLE_HIP
#include "NEBEnergyMinimizerGPU.h"
#endif

namespace hoomd
    {
namespace md
    {

// specify the python module. Note that the name must explicitly match
// the PROJECT() name provided in CMakeLists (with an underscore in
// front)
PYBIND11_MODULE(_neb_plugin, m)
    {
    detail::export_NEBEnergyMinimizer(m);
    detail::export_NEBHook(m);
#ifdef ENABLE_HIP
    // detail::export_NEBEnergyMinimizerGPU(m);
#endif
    }

    } // end namespace md
    } // end namespace hoomd
