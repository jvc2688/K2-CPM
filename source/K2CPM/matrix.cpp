//==================================================================//
// Copyright 2017 Clement Ranc
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//==================================================================//
// This file define a class of square matrix.
//==================================================================//

#include<iomanip>
#include<cmath>
#include "matrix.h"

//==================================================================//
Matrix::Matrix(int d1) : Table(d1, d1) { assert(d1>0); }
//==================================================================//
Matrix::Matrix(const Table& table_in) : Table(table_in) {
    assert(table_in.get_size1() == table_in.get_size2());
    assert(table_in.get_size3() == 1);
}
//==================================================================//
Matrix::Matrix(const Matrix& table_in) : Table(table_in) {}
//==================================================================//
Matrix::~Matrix() {}
//==================================================================//
void Matrix::operator=(const Matrix& table_in) {
    Table::operator=(table_in);
}
//==================================================================//
void Matrix::operator=(const Table& table_in) {
    assert(table_in.get_size1() == table_in.get_size2());
    assert(table_in.get_size3() == 1);
    assert(size1 = table_in.get_size1());

    Table::operator=(table_in);
}
//==================================================================//
void Matrix::operator=(double x) { Table::operator=(x); }
//==================================================================//
Table Matrix::operator*(const Table& table_in) const {
    int s1 = table_in.get_size1();
    int s2 = table_in.get_size2();
    int s3 = table_in.get_size3();
    assert((size2 == s1) && (s3 == 1));
    assert((s1==s2) || (s2==1));

    Table result(s1, s2);
    for (int i=0; i<s1; i++){
	    for (int j=0; j<s2; j++) {
	        double sum = 0;
	        for (int k=0; k<s1; k++) sum += (*this)(i, k)*table_in(k,j);
	        result.set(i,j) = sum;
	    }
	}
    return result;
}
//==================================================================//
void Matrix::print(ostream& ost) const {
    assert(tab != 0x0);

    ost << "Square matrix " << size1 << "x" << size2 << endl;
    ost << setprecision(5);
    for (int i=0; i<size1; i++) {
        for (int j=0; j<size2; j++) {
            ost << (*this)(i, j) << '\t';
        }
        ost << endl ;
    }
    ost << endl;
}
//==================================================================//
void Matrix::cholesky_factor() {
/*
    Cholesky's decomposition of a symmetric and positive-definite matrix
    (let's call the matrix A). The result L of this factorisation is such
    as A = L L^T, where L^T is the transposition of L. This implementation
    is adapted from [1].

    Return: N/A
        Overwrite the matrix by the lower Cholesky factor.

    References
        [1] http://rosettacode.org/wiki/Cholesky_decomposition#C
*/

    // Declarations and initialisation
    // -------------------------------
    int i, j, k, n;
    double s;
    Matrix mat(size1);

    assert(tab != 0x0);

    n = size1;
    for(i=0; i<n; ++i){
        for(j=i+1; j<n; ++j) mat.set(i, j) = 0.0;
    }

    // Calculations
    // ------------
    for(i=0; i<n; ++i){
        for(j=0; j<i+1; ++j){
            s = 0;
            for(k=0; k<j; ++k) s += mat(i, k) * mat(j, k);
            if(i == j) mat.set(i, j) = sqrt((*this)(i, i) - s);
            else mat.set(i, j) = (1.0 / mat(j, j) * ((*this)(i, j) - s));
        }
    }

    // Overwrite the matrix by the lower Cholesky factor.
    *this = mat;
}
//==================================================================//
void Matrix::cholesky_solve(Table& table_in) {
/*
    Solve the linear equation L L^T X = Y, where L is a lower triangular
    matrix, L^T is the transposition of L. Dimension of the matrix is
    n x n, dimension of Y is n.

    Input
    -----
    table_in -- Address of a Table of dimension n
        Right part of the equation AX = Y.

    Return: N/A
        Overwrite the Table table_in by the solution X.
*/

    // Declarations and initializations
    // --------------------------------
    int i, j, n;
    double * Z, * X;

    n = size1;

    Z = new (nothrow) double [n];
    if(Z == nullptr) exit(EXIT_FAILURE);
    for(i = 0; i < n; ++i) Z[i] = 0;

    X = new (nothrow) double [n];
    if(X == nullptr) exit(EXIT_FAILURE);
    for(i = 0; i < n; ++i) X[i] = 0;

    // Forward substitution
    // --------------------
    Z[0] = table_in(0) / (*this)(0, 0);
    for(i = 1; i < n; ++i){
        Z[i] = table_in(i);
        for(j = 0; j < i; ++j) Z[i] -= (*this)(i, j) * Z[j];
        Z[i] /= (*this)(i, i);
    }

    // Backward substitution
    // ---------------------
    X[n-1] = Z[n-1] / (*this)(n-1, n-1);
    for(i = 2; i < n+1; ++i){
        X[n-i] = Z[n-i];
        for(j = 1; j < i; ++j){
            X[n-i] -= (*this)(n-i+j, n-i) * X[n-i+j];
        }
        X[n-i] /= (*this)(n-i, n-i);
    }

    // Overwrite the Table table_in by the solution X
    for(i = 0; i < n; ++i) table_in.set(i) = X[i];

    // Release memory
    delete[] Z;
    delete[] X;
}
