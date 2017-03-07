// A class that define a square matrix
//==================================================================//

#ifndef __MATRIX_H_
#define __MATRIX_H_
#include "table.h"

class Matrix: public Table {
    // Constructors
    // ------------
    public:
    explicit Matrix(int);
    Matrix (const Table&);
    Matrix (const Matrix&);

    virtual ~Matrix();

    // Definitions
    // -----------
    void operator=(const Matrix&);
    void operator=(const Table&);
    void operator=(double);

    virtual Table operator*(const Table&) const;

    // Useful functions
    // ----------------
    protected:
    virtual void print(ostream&) const;

    public:
    void cholesky_factor();  // Caution: overwrite the matrix by lower Choleski factor
    void cholesky_solve(Table&);  // Caution: overwrite argument
};

#endif
