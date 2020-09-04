// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "functions.h"

namespace functions {

	class RegularizedLoss {
	private:
		SubSparseMatrixCSR *_A;
		SubVector *_b;
		ISmoothLoss *_f;			// really should be an object here.
		double _lambda;
		IUnitRegularizer *_g;

		// working buffers
		Vector *_Ax, *_drv;

	public:
		RegularizedLoss(const SparseMatrixCSR &A, const Vector &b, const char f, double lambda, const char g,
			const double l1weight = 0.0);
		~RegularizedLoss();
		double avg_loss(const Vector &x);
		double sum_loss(const Vector &x);
		double regu_loss(const Vector &x);
		// for nonsmooth regularizers, grad means gradient mapping
		double avg_grad(const Vector &x, Vector &g);
		double sum_grad(const Vector &x, Vector &g);
		double regu_grad(const Vector &x, Vector &g);
	};
}