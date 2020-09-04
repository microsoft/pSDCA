// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "dnvector.h"

namespace numerics{

	template <typename, typename> class SubSparseMatrixCSR;

	// Sparse matrix storage with type floatT, interacting with Vector of type T 
	template <typename floatT, typename T>
	class SparseMatrixCSR
	{
		// derived class cannot access private/protected members of instances of parent class
		friend class SubSparseMatrixCSR<floatT, T>;

	private:
		// std::vector<>'s are used to better interface with memory allocated when reading from file
		std::vector<floatT> _mem_values;
		std::vector<size_t> _mem_colidx;
		std::vector<size_t> _mem_rowptr;	// compact storage, only use NIST format for pointers!

	protected:
		size_t _nrows;       // number of rows.
		size_t _ncols;       // number of columns.
		size_t _nnzs;        // number of non-zero elements.
		// pointers to stored memory, convenient for Python interface and derived SubSparseMatrixCSR
		floatT *_values;	// size of _nnzs, pointing nonzero float/double elements in sparse matrix
		size_t *_colidx;	// size of _nnzs, column indices (0-based for elements in _values.
		size_t *_rowstr;	// size of _nrows, rowstr[i]-rowstr[0] is index of first non-zero in row i 
		size_t *_rowend;	// size of _nrows, rowend[i]-rowstr[0] is index of last non-zero in row i

		bool _has_bias_feature = false;

		// this constructor allows derived classes, but protected to prevent explicit use
		SparseMatrixCSR() = default;

	public:

		// This constructor swaps vector<>'s by default, so input memory is lost after calling
		SparseMatrixCSR(std::vector<floatT> &vals, std::vector<size_t> &cidx, std::vector<size_t> &rptr,
					    const bool has_bias_feature = false, const bool make_copy = false);

		// copy constructor, can take a transpose if specified, works with SubSparseMatrixCSR as input
		SparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, bool transpose = false);

		// Explicit Descructor is not necessary because std::vector<> used to store data
		//~SparseMatrixCSR();
		// For ease of memory management, we do not allow copy-assignment operator 
		SparseMatrixCSR<floatT, T>& operator=(const SparseMatrixCSR<floatT, T> &A) = delete;

		// to test if two matrices has the same values
		T norm_of_difference(SparseMatrixCSR<floatT, T> &A);

		inline size_t nrows() const { return _nrows; }
		inline size_t ncols() const { return _ncols; }
		inline size_t nnzs()  const { return _nnzs;  }

		inline bool has_bias_feature() const { return _has_bias_feature; }
		// THE FOLLOWING TWO ARE DANGEOUS TO USE FOR A TRANSPOSED DATA MATRIX WHTERE BIAS ARE THE LAST ROW!
		bool set_bias_feature(const floatT c);	// only works if _has_bias_feature == true
		bool reset_ncols(const size_t n);	// also need to reset bias feature index if exist

		// return number of elements in each column
		void col_elem_counts(std::vector<size_t> &counts) const;
		void row_elem_counts(std::vector<size_t> &counts) const;

		// scale ith row of sparse CSR matrix by y[i]
		void scale_rows(const Vector<T> &y);
		// return value: L[i] is squared 2-norm of ith row of the sparse CSR matrix 
		void row_sqrd2norms(Vector<T> &sqr2dnorms) const;
		// scale each row so that each rom has target 2-norm
		void normalize_rows(const T tgt2norm = 1);
		// return Frobenious norm of matrix
		T Frobenious_norm() const;

		// y = alpha * A * x + beta * y
		void aAxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y) const;
		// y[i] = alpha * A[rowperm[i],:] * x + beta * y[i]
		void aAxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y, std::vector<size_t> &rowperm) const;
		// y = alpha * A' * x + beta * y
		void aATxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y) const;
		// y = alpha * A[rowperm,:]' * x + beta * y
		void aATxby(const T alpha, const Vector<T> &x, const T beta, Vector<T> &y, std::vector<size_t> &rowperm) const;

		// The following two functions are not efficient to take advantage of submatrix
		// block matrix vector product: y = alpha * A(:,start:end) * x + beta * y. Vector x has length end-start+1
		void aAxbyBlock(const T alpha, const size_t start, const Vector<T> &x, const T beta, Vector<T> &y) const;
		// block matrix Transpose vector product y = alpha * A(:,start:end)' * x + beta * y. y has length end-start+1
		void aATxbyBlock(const T alpha, const Vector<T> &x, const T beta, const size_t start, Vector<T> &y) const;

		// return inner product of A(idx,:) and vector x
		T row_dot(const size_t rowidx, const Vector<T> &x) const;
		// y = alpha * A(idx,:) + y
		void row_add(const size_t rowidx, const T alpha, Vector<T> &y) const;
		// sparse soft-thresholding using the sparse pattern of one particular row
		void row_sst(const size_t row, const T c, const Vector<T> &v, Vector<T> &w) const;
	};

	// This class defines a mask for a submatrix consisting of contiguous rows of a sparse matrix
	template <typename floatT, typename T>
	class SubSparseMatrixCSR final : public SparseMatrixCSR<floatT, T>
	{
	private:
		std::vector<size_t> _mem_rowstr;	// store NIST format row indices when sub-matrix
		std::vector<size_t> _mem_rowend;	// is constructed from a subset of row indices

	public:
		// constructor from a SparseMatrixCSR with 0-based indexing
		SubSparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, const size_t row_start, const size_t num_rows);
		SubSparseMatrixCSR(const SparseMatrixCSR<floatT, T> &A, const std::vector<size_t> &row_indices);
		void recast(const SparseMatrixCSR<floatT, T> &A, const size_t row_start, const size_t num_rows);
		void recast(const SparseMatrixCSR<floatT, T> &A, const std::vector<size_t> &row_indices);

		// For ease of memory management, we do not allow copy constructor and copy-assignment operator
		SubSparseMatrixCSR() = delete;
		SubSparseMatrixCSR(const SubSparseMatrixCSR<floatT, T>&) = delete;
		SubSparseMatrixCSR(const SubSparseMatrixCSR<floatT, T>&, bool) = delete;
		SubSparseMatrixCSR<floatT, T>& operator=(const SubSparseMatrixCSR<floatT, T>&) = delete;
	};
}

