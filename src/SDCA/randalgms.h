// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "floattype.h"
#include "functions.h"

using namespace functions;

namespace randalgms
{
	// SDCA need the regularization function to have strongly convex parameter 1 (unit strong convexity)
	int sdca(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda, 
		const IUnitRegularizer &g, const int max_epoch, const double eps_gap, Vector &wio, Vector &aio, 
		const char opt_sample = 'p', const char opt_update = 'd', const bool display = true);

	int sdca(const SparseMatrixCSR &X, const Vector &y, const Vector &theta, const ISmoothLoss &f, const double lambda,
		const IUnitRegularizer &g, const int max_epoch, const double eps_gap, Vector &wio, Vector &aio,
		const char opt_sample = 'p', const char opt_update = 'd', const bool display = true);

	// RPCD algorithm for solving CoCoA local optimization problems
	int rpcd_cocoa(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda_N, const double sigma,
		const int max_epoch, const Vector &Xw, const Vector &a0, Vector &a1, const char opt_sample = 'p', const bool display = true);
}