// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include<string>

#include "regu_loss.h"

namespace functions
{
	RegularizedLoss::RegularizedLoss(const SparseMatrixCSR &A, const Vector &b, const char f, double lambda,
		const char g, const double l1weight)
	{
		assert(A.nrows() == b.length());

		_A = new SubSparseMatrixCSR(A, 0, A.nrows());
		_b = new SubVector(b, 0, b.length());
		_lambda = lambda;

		_Ax = new Vector(A.nrows());
		_drv = new Vector(A.nrows());

		// such conditional constructions needs to be put in a reuseable function!
		switch (f)
		{
		case 'L':
		case 'l':
			_f = new LogisticLoss();
			break;
		case 'S':
		case 's':
			_f = new SmoothedHinge();
			break;
		case '2':
			_f = new SquareLoss();
			break;
		default:
			throw std::runtime_error("Loss function type " + std::to_string(f) + " not defined.");
		}

		switch (g)
		{
		case '2':
			_g = new SquaredL2Norm();
			break;
		case 'E':
		case 'e':
			_g = new ElasticNet(l1weight / lambda);
			break;
		default:
			throw std::runtime_error("Regularizer type " + std::to_string(g) + " not defined.");
		}
	}

	RegularizedLoss::~RegularizedLoss()
	{
		delete _A;
		delete _b;
		delete _f;
		delete _g;
		delete _Ax;
		delete _drv;
	}

	double RegularizedLoss::avg_loss(const Vector &x)
	{
		return this->sum_loss(x) / _b->length();
	}

	double RegularizedLoss::sum_loss(const Vector &x)
	{
		_A->aAxby(1.0, x, 0, *_Ax);
		return _f->sum_loss(*_Ax, *_b);
	}

	double RegularizedLoss::regu_loss(const Vector &x)
	{
		return this->sum_loss(x) / _b->length() + _lambda*(*_g)(x);
	}


	double RegularizedLoss::avg_grad(const Vector &x, Vector &g)
	{
		double sum_loss = this->sum_grad(x, g);
		g.scale(1.0 / _b->length());
		return sum_loss / _b->length();
	}

	double RegularizedLoss::sum_grad(const Vector &x, Vector &g)
	{
		_A->aAxby(1.0, x, 0, *_Ax);
		_f->derivative(*_Ax, *_b, *_drv);
		_A->aATxby(1.0, *_drv, 0, g);
		return _f->sum_loss(*_Ax, *_b);
	}

	double RegularizedLoss::regu_grad(const Vector &x, Vector &g)
	{
		double sum_loss = this->sum_grad(x, g);
		g.scale(1.0 / _b->length());
		g.axpy(_lambda, x);				// this only works for L2 regularization!
		return sum_loss / _b->length() + _lambda*(*_g)(x);
	}
}