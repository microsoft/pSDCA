// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <assert.h>
#include <stdexcept>

#include "floattype.h"
#include "dnvector.h"
#include "spmatrix.h"

namespace functions
{
	using Vector = numerics::Vector<floatT>;
	using SubVector = numerics::SubVector<floatT>;
	using SparseMatrixCSR = numerics::SparseMatrixCSR<spmatT, floatT>;
	using SubSparseMatrixCSR = numerics::SubSparseMatrixCSR<spmatT, floatT>;

	// functions to compute performance of linear regression and binary classification
	floatT regression_mse(const SparseMatrixCSR &X, const Vector &y, const Vector &w);
	floatT binary_error_rate(const SparseMatrixCSR &X, const Vector &y, const Vector &w);
	size_t binary_error_count(const SparseMatrixCSR &X, const Vector &y, const Vector &w);

	// An abstract class to serve as interface for sum of parametrized smooth loss functions 
	class ISmoothLoss
	{
	public:
		ISmoothLoss() = default;
		virtual char symbol() const = 0;
		virtual floatT smoothness() const = 0;	// smoothness parameter of single component function
		virtual floatT loss(const floatT Xiw, const floatT yi) const = 0;
		// the vector form of loss() is provided to faciliate weighted sum of losses
		virtual void   loss(const Vector &Xw, const Vector y, Vector &loss) const = 0;
		// can also add weighted_sum_loss(), but it can be done by dot(weights, loss)
		virtual floatT sum_loss(const Vector &Xw, const Vector &y) const = 0; 
		virtual floatT sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const = 0;
		// derivative() is defined such that gradient is simply expressed as sum_i d[i]*X[i,:]
		virtual floatT derivative(const floatT Xiw, const floatT yi) const = 0;
		virtual void   derivative(const Vector &Xw, const Vector &y, Vector &d) const = 0;
		virtual floatT second_derivative(const floatT Xiw, const floatT yi) const = 0;
		virtual void   second_derivative(const Vector &Xw, const Vector &y, Vector &d) const = 0;
		virtual bool   conj_feasible(const floatT ai, const floatT yi) const = 0;
		virtual bool   conj_feasible(const Vector &a, const Vector &y) const = 0;
		virtual floatT conjugate(const floatT ai, const floatT yi) const = 0;
		// the vector form of conjugate() is provided to faciliate weighted sum of losses
		virtual void   conjugate(const Vector &a, const Vector &y, Vector &conj) const = 0;
		virtual floatT sum_conjugate(const Vector &a, const Vector &y) const = 0;
		virtual floatT sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const = 0;
		virtual floatT conj_derivative(const floatT ai, const floatT yi) const = 0;
		virtual void   conj_derivative(const Vector &a, const Vector &y, Vector &cd) const = 0;
		virtual floatT conj_prox(const floatT sigma, const floatT ai, const floatT yi) const = 0;
		virtual void   conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &u) const = 0;
	};

	// f_y(t) = (1/2)*(t-y)^2 where y is a real number for linear regression
	class SquareLoss : public ISmoothLoss
	{
	public:
		SquareLoss() {};
		inline char symbol() const override { return '2'; };
		inline floatT smoothness() const override { return 1.0f; };	// smoothness of (1/2)*(x-b)^2
		floatT loss(const floatT Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT second_derivative(const floatT Xiw, const floatT yi) const override;
		void   second_derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		bool   conj_feasible(const floatT ai, const floatT yi) const override { return true; };
		bool   conj_feasible(const Vector &a, const Vector &y) const override { return true; };
		floatT conjugate(const floatT ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const override;
		floatT conj_derivative(const floatT ai, const floatT yi) const override;
		void   conj_derivative(const Vector &a, const Vector &y, Vector &cd) const override;
		floatT conj_prox(const floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	// f_y(t) = log(1+exp(-y*t) where y is +1 or -1 for binary classification
	class LogisticLoss : public ISmoothLoss
	{
	private:
		// need to test if the min_log trik is really necessary for numerical stability
		floatT _deltabd;		// boundary margin within [-1, 0] for taking logarithm
		floatT _epsilon;		// stopping accuracy for Newton's method
		int    _maxiter;		// maximum iteration for Newton's method

		floatT _xentrpy;		// entropy when dual variable are close to boundary of [-1,0]
		floatT _xderiva;		// derivative of conjugate function near the boundary

	public:
		// may not work as desired when floatT = float, 1 + 1e-12 = 1 anyway!
		LogisticLoss(const floatT deltabd = 1.0e-12f, const floatT epsilon = 1.0e-6f, const int maxiter = 100);
		inline char symbol() const override { return 'l'; };
		inline floatT smoothness() const override { return 0.25; }	// smoothness of log(1+exp(-b*t))
		floatT loss(const floatT Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT second_derivative(const floatT Xiw, const floatT yi) const override;
		void   second_derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		bool   conj_feasible(const floatT ai, const floatT yi) const override { return (yi*ai <= 0 && yi*ai >= -1); };
		bool   conj_feasible(const Vector &a, const Vector &y) const override;
		floatT conjugate(const floatT ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const override;
		floatT conj_derivative(const floatT ai, const floatT yi) const override;
		void   conj_derivative(const Vector &a, const Vector &y, Vector &cd) const override;
		floatT conj_prox(const floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	// smoothed hinge constructed by adding strong convexity to conjugate
	class SmoothedHinge : public ISmoothLoss
	{
	private:
		floatT _delta;			// conjugate smoothing parameter

	public:
		SmoothedHinge(const floatT delta = 1) { assert(delta > 0); _delta = delta; };
		inline char symbol() const override { return 's'; };
		inline floatT smoothness() const override { return 1.0f / _delta; }; // smoothness by conjugate smoothing
		floatT loss(const floatT Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT second_derivative(const floatT Xiw, const floatT yi) const override;
		void   second_derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		bool   conj_feasible(const floatT ai, const floatT yi) const override { return (yi*ai <= 0 && yi*ai >= -1); };
		bool   conj_feasible(const Vector &a, const Vector &y) const override;
		floatT conjugate(const floatT ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const override;
		floatT conj_derivative(const floatT ai, const floatT yi) const override;
		void   conj_derivative(const Vector &a, const Vector &y, Vector &cd) const override;
		floatT conj_prox(const floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	//================================================================================================
	// An abstract class to serve as interface for convex regularization functions
	class IUnitRegularizer
	{
	private:
		Vector *_offset = nullptr;
	public:
		IUnitRegularizer() = default;
		virtual char symbol() const = 0;					// enable customized algorithm implementations
		virtual Vector *offset() const { return _offset; };	// enable offset for derived classes
		floatT convexity() const { return 1.0f; };			// strong convexity parameter
		virtual floatT l1penalty() const { return 0.0f; };		// return l1 penalty parameter if any
		virtual floatT operator()(const Vector &w) const = 0;	// return function value
		// Compute f*(v) = max_w {v'*w - f(w)}, can also use maxing w if already computed
		virtual floatT conjugate(const Vector &v) const = 0;
		virtual floatT conjugate(const Vector &v, const Vector &w) const = 0;
		// prox() operator computes u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
		virtual void prox(const floatT tau, const Vector &w, Vector &u) const = 0;
		// return gradient of conjugate function: argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
		virtual void conj_grad(const Vector &v, Vector &w) const = 0;	
	};

	// g(w) = (1/2)*||w||_2^2
	class SquaredL2Norm : public IUnitRegularizer
	{
	public:
		SquaredL2Norm() {};
		inline char symbol() const override { return '2'; };
		floatT operator()(const Vector &w) const override;
		floatT conjugate(const Vector &v) const override;
		floatT conjugate(const Vector &v, const Vector &w) const override;
		void prox(const floatT tau, const Vector &w, Vector &u) const override;
		void conj_grad(const Vector &w, Vector &g) const override;	
	};

	// g(w) = (1/2)*||w||_2^2 + lambda1*||w||_1
	class ElasticNet : public IUnitRegularizer
	{
	private:
		floatT _lambda1;

	public:
		ElasticNet(const floatT lambda1) { assert(lambda1 > 0); _lambda1 = lambda1; };
		inline char symbol() const override { return 'e'; };
		inline floatT l1penalty() const override { return _lambda1; };
		floatT operator()(const Vector &w) const override;
		floatT conjugate(const Vector &v) const override;
		floatT conjugate(const Vector &v, const Vector &w) const override;
		void prox(const floatT tau, const Vector &w, Vector &u) const override;
		void conj_grad(const Vector &w, Vector &g) const override;
	};

	// R(w) = (1/2)*||w-w0||_2^2 
	class OffsetQuadratic : public IUnitRegularizer
	{
	private:
		Vector *_w0;
		Vector *_dw;
	public:
		OffsetQuadratic(const size_t dim);
		~OffsetQuadratic();
		void update_offset(const Vector &w0) { _w0->copy(w0); };
		Vector *offset() const override { return _w0; };
		char symbol() const { return 'q'; };
		floatT operator()(const Vector &w) const override;
		floatT conjugate(const Vector &v) const override;
		floatT conjugate(const Vector &v, const Vector &w) const override;
		void prox(const floatT tau, const Vector &w, Vector &u) const override;
		void conj_grad(const Vector &w, Vector &g) const override;
	};

	// need to implement the following more sophisticated regularizer for sparsity
	// g(w) = (3/2)log(d)||w||_q^2 + (lambda1/lambda2)||w||_1  (when feature has low infty norm) 
}

