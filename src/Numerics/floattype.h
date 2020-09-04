// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

// for convenience to use conditional MPI types
#define DOUBLE_VECTOR true
#define DOUBLE_MATRIX true

using floatT = std::conditional<DOUBLE_VECTOR, double, float>::type;
using spmatT = std::conditional<DOUBLE_MATRIX, double, float>::type;

// otherwise can simply use the following
//using floatT = double;
//using spmatT = float;

#if DOUBLE_VECTOR
#define MPI_VECTOR_TYPE MPI_DOUBLE
#else
#define MPI_VECTOR_TYPE MPI_FLOAT
#endif
