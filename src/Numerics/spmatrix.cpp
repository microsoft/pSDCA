// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <assert.h> 
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>
#include <omp.h>

#include "spmatrix.h"

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
using std::int64_t;

namespace numerics
{
	// constructor from three vectors, column index start with 0-indexing, not necessarily ordered
	template <typename floatT, typename T>
	SparseMatrixCSR<floatT, T>::SparseMatrixCSR(std::vector<floatT> &vals, std::vector<size_t> &cidx, 
		std::vector<size_t> &rptr, const bool has_bias_feature, const bool make_copy)
	{
		_nrows = rptr.size() - 1;
		_ncols = *std::max_element(cidx.begin(), cidx.end()) + 1;
		_nnzs  = vals.size();
		
		if (make_copy)	// make deep copies using std::vector<> overloaded operator=()
		{
			_mem_values = vals;
			_mem_colidx = cidx;
			_mem_rowptr = rptr;
		}
		else	// swap two vector<>, so argument memory removed at exit to reduce memory footage
		{
			_mem_values.swap(vals);
			_mem_colidx.swap(cidx);
			_mem_rowptr.swap(rptr);
		}

		// these pointers only point to existing memories, no need to use new and delete[]
		_values = _mem_values.data();
		_colidx = _mem_colidx.data();
		_rowstr = _mem_rowptr.data();
		_rowend = _mem_rowptr.data() + 1;

		// specify biased feature
		_has_bias_feature = has_bias_feature;
	}

	// copy constructor, can take a transpose if specified. A could be a SubSparseMatrixCSR!
	template <typename floatT, typename T>
	SparseMatrixCSR<floatT, T>::SparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, bool transpose)
	{
		if (!transpose)
		{
			_nrows = A._nrows;
			_ncols = A._ncols;
			_nnzs = A._nnzs;
			_mem_values.resize(_nnzs);
			_mem_colidx.resize(_nnzs);
			_mem_rowptr.resize(_nrows + 1);
			size_t k = 0;
			for (size_t i = 0; i < _nrows; i++) {
				_mem_rowptr[i] = k;
				for (size_t j = A._rowstr[i]; j < A._rowend[i]; j++)
				{
					_mem_colidx[k] = A._colidx[j];
					_mem_values[k] = A._values[j];
					k++;
				}
			}
			_mem_rowptr[_nrows] = k;

			_rowstr = _mem_rowptr.data();
			_rowend = _mem_rowptr.data() + 1;
			_has_bias_feature = A.has_bias_feature();
			return;
		}

		// the following code does not work if A is a SubSparseMatrix, contiguous or not
		/*
		if (!transpose)	// use std::vector<>::operator= to make deep copies
		{
			_nrows = A._nrows;
			_ncols = A._ncols;
			_nnzs  = A._nnzs;
			_mem_values = A._mem_values;
			_mem_colidx = A._mem_colidx;
			_mem_rowptr = A._mem_rowptr;
			_values = _mem_values.data();
			_colidx = _mem_colidx.data();
			_rowstr = _mem_rowptr.data();
			_rowend = _mem_rowptr.data() + 1;
			_has_bias_feature = A.has_bias_feature();
			return;
		}
		*/

		// SparseMatrixCSR transpose, works for un-sorted column indices
		size_t m, n, nnz, j, p, q;  
		size_t *Trwp, *Tcli;
		floatT *Tval;
		const size_t* Arstr = A._rowstr;
		const size_t* Arend = A._rowend;
		const size_t* Acli = A._colidx;
		const floatT* Aval = A._values;

		m = A._nrows; 		
		n = A._ncols;		
		nnz = A._nnzs;
		_nrows = n; 		
		_ncols = m;		
		_nnzs = nnz;

		_mem_values.resize(nnz); 
		_mem_colidx.resize(nnz); 
		_mem_rowptr.resize(n + 1);
		Trwp = _mem_rowptr.data();
		Tcli = _mem_colidx.data(); 
		Tval = _mem_values.data();

		std::vector<size_t> ws(n, 0);
		for (j = 0; j < m; j++) {
			for (p = Arstr[j]; p < Arend[j]; p++) {
				ws[Acli[p]]++;
			}
		}
		Trwp[0] = 0;
		for (j = 0; j < n; j++) {
			Trwp[j + 1] = Trwp[j] + ws[j];
			ws[j] = Trwp[j];
		}
		for (j = 0; j < m; j++) {
			for (p = Arstr[j]; p < Arend[j]; p++) {
				Tcli[q = ws[Acli[p]]++] = j;
				Tval[q] = Aval[p];
			}
		}

		_values = _mem_values.data();
		_colidx = _mem_colidx.data();
		_rowstr = _mem_rowptr.data();
		_rowend = _mem_rowptr.data() + 1;
		// hard to tell if bias_feature is meaningful in transpose
		_has_bias_feature = A.has_bias_feature();
	}

	// THIS ONE IS DANGEOUS TO USE FOR A TRANSPOSED DATA MATRIX WHTERE BIAS ARE THE LAST ROW!
	template <typename floatT, typename T>
	bool SparseMatrixCSR<floatT, T>::set_bias_feature(const floatT c)
	{
		if (_has_bias_feature) {
			auto vals = _values;
			auto rend = _rowend;
			auto nrows = _nrows;
			for (size_t i = 0; i < nrows; i++) {
				vals[rend[i]-1] = c;	
			}
		}

		return _has_bias_feature;
	}

	// THIS ONE IS DANGEOUS TO USE FOR A TRANSPOSED DATA MATRIX WHTERE BIAS ARE THE LAST ROW!
	// reset number of columns only if n is larger than this->nClos;
	template <typename floatT, typename T>
	bool SparseMatrixCSR<floatT, T>::reset_ncols(const size_t n)
	{
		if (n <= _ncols) return false;

		_ncols = n;
		if (_has_bias_feature) {
			auto cidx = _colidx;
			auto rend = _rowend;
			auto nrows = _nrows;
			for (size_t i = 0; i < nrows; i++) {
				cidx[rend[i] - 1] = n - 1;
			}
		}
		return true;
	}

	// return number of elements in each column
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::col_elem_counts(std::vector<size_t> &counts) const
	{
		counts.resize(_ncols);
		std::fill(counts.begin(), counts.end(), 0);
		for (size_t i = 0; i < _nrows; i++) {
			for (size_t p = _rowstr[i]; p < _rowend[i]; p++) {
				counts[_colidx[p]]++;
			}
		}
	}

	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::row_elem_counts(std::vector<size_t> &counts) const
	{
		counts.resize(_nrows);
		std::fill(counts.begin(), counts.end(), 0);
		for (size_t i = 0; i < _nrows; i++) {
			counts[i] = _rowend[i] - _rowstr[i];
		}
	}

	// scale ith row of sparse CSR matrix by y[i]
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::scale_rows(const Vector<T> &y)
	{
		assert(y.length() == _nrows);

		floatT *vals = _values;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			floatT yi = y[i];
			size_t endidx = _rowend[i];
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				vals[j] *= yi;
			}
		}
	}

	// return value: L[i] is 2-norm squared of ith row of the sparse CSR matrix 
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::row_sqrd2norms(Vector<T> & sqrd2norms) const
	{
		assert(sqrd2norms.length() == _nrows);

		floatT *vals = _values;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			T s = 0;
			size_t endidx = _rowend[i];
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				s += vals[j] * vals[j];
			}
			sqrd2norms[i] = s;
		}
	}

	// scale each row so that each rom has target 2-norm
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::normalize_rows(const T tgt2norm)
	{
		floatT *vals = _values;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			T s = 0;
			size_t endidx = _rowend[i];
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				s += vals[j] * vals[j];
			}
			if (s > 0) {
				s = tgt2norm / sqrt(s);
				for (size_t j = _rowstr[i]; j < endidx; j++)
				{
					vals[j] *= s;
				}
			}
		}
	}

	// return Frobenious norm of matrix
	template <typename floatT, typename T>
	T SparseMatrixCSR<floatT, T>::Frobenious_norm() const
	{
		Vector<T> L(_nrows);
		this->row_sqrd2norms(L);
		return sqrt(L.sum());
	}

	// to test if two matrices has the same values
	template <typename floatT, typename T>
	T SparseMatrixCSR<floatT, T>::norm_of_difference(SparseMatrixCSR<floatT, T> &A)
	{
		if (_nrows != A._nrows || _ncols != A._ncols || _nnzs != A._nnzs) return -1;

		T diff_sqrd = 0.0;

		floatT *vals = _values;
		floatT *A_vals = A._values;
		size_t *colidx = _colidx;
		size_t *A_colidx = A._colidx;
		size_t nrows = _nrows;
		for (int64_t i = 0; i < nrows; i++)
		{
			if (_rowend[i] - _rowstr[i] != A._rowend[i] - A._rowstr[i]) return -1;

			size_t ni = _rowend[i] - _rowstr[i];
			size_t jj, jA;
			for (size_t j = 0; j < ni; j++)
			{
				jj = _rowstr[i] + j;
				jA = A._rowstr[i] + j;
				if (colidx[jj] != A_colidx[jA]) return -1;
				diff_sqrd += std::pow(vals[jj] - A_vals[jA], 2);
			}
		}

		return std::sqrt(diff_sqrd);
	}

	// y = alpha * A * x + beta * y
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aAxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y) const
	{
		assert(y.length() == _nrows && x.length() == _ncols);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			T Aix = 0;
			size_t endidx = _rowend[i];
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				Aix += vals[j] * x[cidx[j]];
			}
			y[i] *= beta;
			y[i] += alpha*Aix;
		}
	}

	// y[i] = alpha * A[perm[i],:] * x + beta * y[i]
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aAxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y, std::vector<size_t> &rowperm) const
	{
		assert(y.length() == _nrows && x.length() == _ncols && rowperm.size() == _nrows);
		assert(*std::min_element(std::begin(rowperm), std::end(rowperm)) >= 0
			&& *std::max_element(std::begin(rowperm), std::end(rowperm)) < _nrows);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			T Aix = 0;
			size_t endidx = _rowend[rowperm[i]];
			for (size_t j = _rowstr[rowperm[i]]; j < endidx; j++)
			{
				Aix += vals[j] * x[cidx[j]];
			}
			y[i] *= beta;
			y[i] += alpha*Aix;
		}
	}

	// block matrix vector product: y = alpha * A(:,start:end) * x + beta * y. Vector x has length end-start+1
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aAxbyBlock(const T alpha, const size_t start, const Vector<T> &x, 
		const T beta, Vector<T> &y) const
	{
		assert(y.length() == _nrows && start + x.length() <= _ncols);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;
		size_t blockend = start + x.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			T Aix = 0;
			size_t endidx = _rowend[i];
			size_t idx;
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				idx = cidx[j];
				if (idx >= start && idx < blockend)
				{
					Aix += vals[j] * x[idx - start];
				}
			}
			y[i] *= beta;
			y[i] += alpha*Aix;
		}
	}

	// y = alpha * A' * x + beta * y
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aATxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y) const
	{
		assert(x.length() == _nrows && y.length() == _ncols);

		// first scale it before entering racing conditions using OpenMP
		y.scale(beta);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			size_t endidx = _rowend[i];
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				#pragma omp atomic
				y[cidx[j]] += alpha * vals[j] * x[i];
			}
		}
	}

	// y = alpha * A[rowperm,:]' * x + beta * y
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aATxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y, std::vector<size_t> &rowperm) const
	{
		assert(x.length() == _nrows && y.length() == _ncols && rowperm.size() == _nrows);
		assert(*std::min_element(std::begin(rowperm), std::end(rowperm)) >= 0
			&& *std::max_element(std::begin(rowperm), std::end(rowperm)) < _nrows);

		// first scale it before entering racing conditions using OpenMP
		y.scale(beta);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			size_t endidx = _rowend[rowperm[i]];
			for (size_t j = _rowstr[rowperm[i]]; j < endidx; j++)
			{
				#pragma omp atomic
				y[cidx[j]] += alpha * vals[j] * x[i];
			}
		}
	}

	// block matrix Transpose vector product y = alpha * A(:,start:end)' * x + beta * y. Vector y has length end-start+1
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::aATxbyBlock(const T alpha, const Vector<T> &x, const T beta, 
		const size_t start, Vector<T> &y) const
	{
		assert(x.length() == _nrows && start + y.length() <= _ncols);

		// first scale it before entering racing conditions using OpenMP
		y.scale(beta);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t nrows = _nrows;
		size_t blockend = start + x.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < nrows; i++)
		{
			size_t endidx = _rowend[i];
			size_t cj;
			for (size_t j = _rowstr[i]; j < endidx; j++)
			{
				cj = cidx[j];
				if (cj >= start && cj < blockend)
				{
					#pragma omp atomic
					y[cj - start] += alpha * vals[j] * x[i];
				}
			}
		}
	}

	// return inner product of A(idx,:) and vector x, where idx is zero-based indexing
	template <typename floatT, typename T>
	T SparseMatrixCSR<floatT, T>::row_dot(const size_t row, const Vector<T> &x) const
	{
		assert(row < _nrows && x.length() == _ncols);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t stridx = _rowstr[row];
		size_t endidx = _rowend[row];
		T a = 0;

		//#pragma omp parallel for reduction(+:a)  // dramatic slow down for small number of features 
		//#pragma omp parallel for reduction(+:a) if (endidx-stridx > 10000)	// not really helpful 
		for (int64_t j = stridx; j < endidx; j++)
		{
			a += vals[j] * x[cidx[j]];
		}
		return a;
	}

	// y = alpha * A(idx,:) + y, where idx is zero-based indexing
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::row_add(const size_t row, const T alpha, Vector<T> &y) const
	{
		assert(row < _nrows && y.length() == _ncols);

		floatT *vals = _values;
		size_t *cidx = _colidx;
		size_t stridx = _rowstr[row];
		size_t endidx = _rowend[row];

		//#pragma omp parallel for	// dramatic slow down for small number of features, need condition 
		//#pragma omp parallel for if (endidx-stridx > 10000)	// even this is not really helpful 
		for (int64_t j = stridx; j < endidx; j++)
		{
			y[cidx[j]] += alpha * vals[j];
		}
	}

	// Sparse soft-thresholding with threshold c, only using the sparsity pattern of row, not using values!
	// Only read elements of v and modify elements of w that correspond to nonzero elements in sepcified row
	template <typename floatT, typename T>
	void SparseMatrixCSR<floatT, T>::row_sst(const size_t row, const T c, const Vector<T> &v, Vector<T> &w) const
	{
		assert(v.length() == _ncols && w.length() == _ncols);
		
		size_t stridx = _rowstr[row];
		size_t endidx = _rowend[row];
		size_t *cidx = _colidx;

		//#pragma omp parallel for	// dramatic slow down for small number of features, need condition 
		//#pragma omp parallel for if (endidx-stridx > 10000)	// even this is not really helpful
		for (int64_t j = stridx; j < endidx; j++)
		{
			size_t idx = cidx[j];
			T vlb = v[idx] - c;
			T vub = v[idx] + c;
			w[idx] = vlb > 0 ? vlb : vub < 0 ? vub : 0;
		}
	}

	// Block-rows submatrix of SparseMatrixCSR with 0-based indexing, only pointers, no memory allocation
	template <typename floatT, typename T>
	SubSparseMatrixCSR<floatT, T>::SubSparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, const size_t row_start, const size_t num_rows)
	{
		this->recast(A, row_start, num_rows);
	}

	// this will work for submatrix of submatrix, but only contiguous submatrices!
	template <typename floatT, typename T>
	void SubSparseMatrixCSR<floatT, T>::recast(const SparseMatrixCSR<floatT, T> &A, const size_t row_start, const size_t num_rows)
	{
		std::vector<size_t> idx(num_rows);
		std::iota(std::begin(idx), std::end(idx), row_start);
		this->recast(A, idx);

		// The following code does not work for submatrix of submatrix if the rows are not contiguous!
		/*
		assert(num_rows > 0 && row_start + num_rows <= A._nrows);

		_nrows = num_rows;
		_ncols = A._ncols;
		_nnzs = A._rowend[row_start + num_rows - 1] - A._rowstr[row_start];

		_values = A._values;
		_colidx = A._colidx;
		_rowstr = A._rowstr + row_start;
		_rowend = A._rowend + row_start;

		_has_bias_feature = A.has_bias_feature();
		*/
	}

	template <typename floatT, typename T>
	SubSparseMatrixCSR<floatT, T>::SubSparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, const std::vector<size_t> &row_indices)
	{
		this->recast(A, row_indices);
	}
		
	template <typename floatT, typename T>
	void SubSparseMatrixCSR<floatT, T>::recast(const SparseMatrixCSR<floatT, T> &A, const std::vector<size_t> &row_indices)
	{
		assert(*min_element(row_indices.begin(), row_indices.end()) >= 0);
		assert(*max_element(row_indices.begin(), row_indices.end()) < A._nrows);

		_nrows = row_indices.size();
		_ncols = A._ncols;

		_values = A._values;
		_colidx = A._colidx;
		_mem_rowstr.resize(_nrows);
		_mem_rowend.resize(_nrows);
		_nnzs = 0;
		for (size_t i = 0; i < _nrows; i++) {
			_mem_rowstr[i] = A._rowstr[row_indices[i]];
			_mem_rowend[i] = A._rowend[row_indices[i]];
			_nnzs += _mem_rowend[i] - _mem_rowstr[i];
		}
		_rowstr = _mem_rowstr.data();
		_rowend = _mem_rowend.data();

		_has_bias_feature = A.has_bias_feature();
	}

	// instantiates template with different float types
	template class SparseMatrixCSR<float, float>;
	template class SparseMatrixCSR<float, double>;
	template class SparseMatrixCSR<double, float>;
	template class SparseMatrixCSR<double, double>;
	template class SubSparseMatrixCSR<float, float>;
	template class SubSparseMatrixCSR<float, double>;
	template class SubSparseMatrixCSR<double, float>;
	template class SubSparseMatrixCSR<double, double>;
}
