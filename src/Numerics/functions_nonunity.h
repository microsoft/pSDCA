#pragma once

#include <assert.h>
#include <stdexcept>

#include "floattype.h"

namespace randalgms
{
	// functions to compute performance of linear regression and binary classification
	floatT regression_mse(const SparseMatrixCSR &X, const Vector &y, const Vector &w);
	floatT binary_error_rate(const SparseMatrixCSR &X, const Vector &y, const Vector &w);
	size_t binary_error_count(const SparseMatrixCSR &X, const Vector &y, const Vector &w);

	// An abstract class to serve as interface for sum of parametrized smooth loss functions 
	class ISmoothLoss
	{
	public:
		ISmoothLoss() = default;
		virtual floatT smoothness() const = 0;	// smoothness parameter of single component function
		virtual floatT loss(const float Xiw, const floatT yi) const = 0;
		virtual void   loss(const Vector &Xw, const Vector y, Vector &loss) const = 0;
		// can also add weighted_sum_loss(), but it can be done by dot(weights, loss)
		virtual floatT sum_loss(const Vector &Xw, const Vector &y) const = 0; 
		// derivative() is defined such that gradient is simply expressed as sum_i d[i]*X[i,:]
		virtual floatT derivative(const floatT Xiw, const floatT yi) const = 0;
		virtual void   derivative(const Vector &Xw, const Vector &y, Vector &d) const = 0;
		virtual floatT conjugate(const floatT &ai, const floatT yi) const = 0;
		virtual void   conjugate(const Vector &a, const Vector &y, Vector &conj) const = 0;
		virtual floatT sum_conjugate(const Vector &a, const Vector &y) const = 0;
		virtual floatT conj_prox(floatT sigma, const floatT ai, const floatT yi) const = 0;
		virtual void   conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &u) const = 0;
	};

	// f_y(t) = (1/2)*(t-y)^2 where y is a real number for linear regression
	class SquareLoss : public ISmoothLoss
	{
	public:
		SquareLoss() {};
		inline floatT smoothness() const override { return 1.0f; };	// smoothness of (1/2)*(x-b)^2
		floatT loss(const float Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT conjugate(const floatT &ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT conj_prox(floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	// f_y(t) = log(1+exp(-y*t) where y is +1 or -1 for binary classification
	class LogisticLoss : public ISmoothLoss
	{
	private:
		// need to test if the min_log trik is really necessary for numerical stability
		floatT _min_log;		// boundary margin within [-1, 0] for taking logarithm
		floatT _epsilon;		// stopping accuracy for Newton's method
		int    _maxiter;		// maximum iteration for Newton's method

	public:
		// may not work as desired when floatT = float, 1 + 1e-12 = 1 anyway!
		LogisticLoss(const floatT min_log = 1.0e-12F, const floatT epsilon = 1.0e-6F, const int maxiter = 10);
		inline floatT smoothness() const override { return 0.25; }	// smoothness of log(1+exp(-b*t))
		floatT loss(const float Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT conjugate(const floatT &ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT conj_prox(floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	// smoothed hinge constructed by adding strong convexity to conjugate
	class SmoothedHinge : public ISmoothLoss
	{
	private:
		floatT _delta;			// conjugate smoothing parameter

	public:
		SmoothedHinge(const floatT delta = 1) { assert(delta > 0); _delta = delta; };
		inline floatT smoothness() const override { return 1.0f / _delta; }; // smoothness by conjugate smoothing
		floatT loss(const float Xiw, const floatT yi) const override;
		void   loss(const Vector &Xw, const Vector y, Vector &loss) const override;
		floatT sum_loss(const Vector &Xw, const Vector &y) const override;
		floatT derivative(const floatT Xiw, const floatT yi) const override;
		void   derivative(const Vector &Xw, const Vector &y, Vector &d) const override;
		floatT conjugate(const floatT &ai, const floatT yi) const override;
		void   conjugate(const Vector &a, const Vector &y, Vector &conj) const override;
		floatT sum_conjugate(const Vector &a, const Vector &y) const override;
		floatT conj_prox(floatT sigma, const floatT ai, const floatT yi) const override;
		void   conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &u) const override;
	};

	//================================================================================================
	// An abstract class to serve as interface for convex regularization functions
	class IRegularizer
	{
	protected:
		bool _is_unity = false;					// _is_unity == true iff this->convexity() == 1
	public:
		IRegularizer() = default;
		virtual char symbol() const = 0;		// to enable customized algorithm implementations
		virtual floatT convexity() const = 0;	// strong convexity parameter
		inline bool is_unity() const { return _is_unity; };
		// rescale so that convexity parameter = 1, return equivalent scale (previous convexity)
		virtual floatT unitize() = 0;
		virtual floatT operator()(const Vector &w) const = 0;	// return function value
		virtual floatT conjugate(const Vector &v) const = 0;	// return conjugate function value
		virtual floatT conjugate(const Vector &v, const Vector &w) const = 0;
		// prox() operator computes u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
		virtual void prox(const floatT tau, const Vector &w, Vector &u) const = 0;
		// return gradient of conjugate function: argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
		virtual void conj_grad(const Vector &v, Vector &w) const = 0;	
	};

	// g(w) = (lambda2/2)*||w||_2^2
	class SquaredL2Norm : public IRegularizer
	{
	private:
		floatT _lambda;

	public:
		SquaredL2Norm(const floatT lambda);
		SquaredL2Norm(const SquaredL2Norm &g, bool unitize);
		inline char symbol() const override { return '2'; };
		inline floatT convexity() const override { return _lambda; };
		inline floatT unitize() override;
		floatT operator()(const Vector &w) const override;
		floatT conjugate(const Vector &v) const override;
		floatT conjugate(const Vector &v, const Vector &w) const override;
		void prox(const floatT tau, const Vector &w, Vector &u) const override;
		void conj_grad(const Vector &w, Vector &g) const override;	
	};

	// g(w) = (lambda2/2)*||w||_2^2 + lambda1*||w||_1
	class ElasticNet : public IRegularizer
	{
	private:
		floatT _lambda1, _lambda2;

	public:
		ElasticNet(const floatT lambda1, const floatT lambda2);
		ElasticNet(const ElasticNet &g, bool unitize);
		inline char symbol() const override { return 'e'; };
		inline floatT convexity() const override { return _lambda2; };
		inline floatT unitize() override;
		floatT operator()(const Vector &w) const override;
		floatT conjugate(const Vector &v) const override;
		floatT conjugate(const Vector &v, const Vector &w) const override;
		void prox(const floatT tau, const Vector &w, Vector &u) const override;
		void conj_grad(const Vector &w, Vector &g) const override;

		// member functions specific to this class, not defined in IRegularizer
		inline floatT l1parameter() const { return _lambda1; };
	};

	// g(w) = (3/2)log(d)||w||_q^2 + (lambda1/lambda2)||w||_1  (when feature has low infty norm) 
}

