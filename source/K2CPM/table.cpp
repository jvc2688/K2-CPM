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
// This file define a class of three dimensional table.
//==================================================================//

#include<fstream>
#include<iomanip>
#include<cmath>
#include "table.h"

//==================================================================//
Table::Table(int d1, int d2, int d3) : size1(d1),
        size2(d2), size3(d3), tab(0x0) {
    assert((d1>0) && (d2>0) && (d3>0));
    tab = new double [d1 * d2 * d3];
}
//==================================================================//
Table::Table(const Table& table_in) : size1(table_in.size1),
        size2(table_in.size2), size3(table_in.size3), tab(0x0) {

    assert((size1>0) && (size2>0) && (size3>0));
    int size = size1 * size2 * size3;
    tab = new double [size];
    assert(table_in.tab != 0x0);
    for (int i=0; i<size; i++) tab[i] = table_in.tab[i];
}
//==================================================================//
Table::Table(const char* fname) {
    ifstream file(fname);
    assert(file);
    file >> size1 >> size2 >> size3;
    assert((size1>0) && (size2>0) && (size3>0));
    int size = size1 * size2 * size3;
    tab = new double[size];
    for (int i=0; i<size; ++i) {
        assert(file);
        file >> tab[i];
    }
}
//==================================================================//
Table::Table(const char* fname, const char* fname_mask, const int axis) {
    int i, i2, itab, size, n_ref, size_old, size2_old, size3_old;
    double x;
    Table mask(fname_mask);
    assert((mask.get_size2()==1) && (mask.get_size3()==1));
    n_ref = 0;
    for (i=0; i<mask.get_size1(); ++i) {
        if (mask(i) < 0.5) ++n_ref;
    }

    ifstream file(fname);
    assert(file);
    file >> size1 >> size2 >> size3;
    size2_old = size2;
    size3_old = size3;
    size_old = size1 * size2 * size3;
    switch(axis) {
        case 0 :
            size1 -= n_ref;
            break;
        case 1 :
            size2 -= n_ref;
            break;
        case 3 :
            size3 -= n_ref;
            break;
    }
    assert((size1>0) && (size2>0) && (size3>0));
    size = size1 * size2 * size3;
    tab = new double[size];
    i2 = 0;
    for (i=0; i<size_old; ++i) {
        assert(file);

        switch(axis) {
            case 0 :
                itab = i / (size2_old * size3_old);  // value of first index
                break;
            case 1 :
                itab = (i % (size2_old * size3_old)) / size3_old;  // value of second index
                break;
            case 3 :
                itab = (i % (size2_old * size3_old)) % size3_old;  // value of third index
                break;
        }
        if (mask(itab) < 0.5) file >> x;
        else{
            file >> tab[i2];
            ++i2;
        }
    }
}
//==================================================================//
Table::~Table() {
  if (tab != 0x0) delete[] tab ;
}
//==================================================================//
void Table::operator=(const Table& table_in) {
    assert(size1 == table_in.size1);
    assert(size2 == table_in.size2);
    assert(size3 == table_in.size3);
    assert(table_in.tab != 0x0);
    assert(tab != 0x0);

    int size = size1 * size2 * size3;
    for (int i=0; i<size; i++) tab[i] = table_in.tab[i];
}
 //==================================================================//
void Table::operator=(double x) {
    assert(tab != 0x0);

    int size = size1 * size2 * size3;
    for (int i=0; i<size; i++) tab[i] = x;
}
//==================================================================//
void Table::print(ostream& ost) const {
    assert(tab != 0x0);

    int tabdim = 0;
    if (size1 > 1) tabdim++;
    if (size2 > 1) tabdim++;
    if (size3 > 1) tabdim++;

    ost << "Table of ";
    if (size1 > 1) {
        ost << size1;
        if (tabdim>1) {
            ost << "x";
            tabdim--;
        }
    }
    if (size2>1) {
        ost << size2;
        if (tabdim>1) ost << "x";
    }
    if (size3>1) ost << size3;
    ost << " elements" << endl;

    ost << setprecision(12);

    if (size3 == 1) {
        for (int i=0; i<size1; i++) {
            for (int j=0; j<size2; j++) {
                ost << (*this)(i, j) << '\t';
	        }
            if (size2 >1) ost << endl;
        }
        ost << endl;
    }
    else {
        for (int i=0; i<size1; i++) {
            ost << "i=" << i << '\n';
            for (int j=0; j<size2; j++) {
                for (int k=0; k<size3; k++) {
                    ost << (*this)(i, j, k) << '\t';
                }
                ost << endl;
            }
            ost << endl ;
        }
        ost << endl ;
    }
}
//==================================================================//
ostream& operator<<(ostream& ost, const Table& table_in ) {
    assert(table_in.tab != 0x0) ;
    table_in.print(ost) ;
    return ost;
}
//==================================================================//
Table operator-(const Table& table_in) {
    int s1 = table_in.get_size1();
    int s2 = table_in.get_size2();
    int s3 = table_in.get_size3();
    int size = s1 * s2 * s3;

    Table result(s1, s2, s3);
    for (int i=0; i<size; i++) result.tab[i] = -table_in.tab[i];
    return result;
}
//==================================================================//
Table operator+(const Table& t1, const Table& t2) {
    int s1 = t1.get_size1();
    int s2 = t1.get_size2();
    int s3 = t1.get_size3();
    assert ((t2.get_size1() == s1) && (t2.get_size2() == s2)
      && (t2.get_size3() == s3));
    int size = s1 * s2 * s3;

    Table result(s1, s2, s3);
    for (int i=0; i<size; i++)
        result.tab[i]  = t1.tab[i] + t2.tab[i];
    return result;
}
//==================================================================//
Table operator+(const Table& t1, double x) {
    int s1 = t1.get_size1();
    int s2 = t1.get_size2();
    int s3 = t1.get_size3();

    Table result(s1, s2, s3);
    result = x;
    return t1 + result;
}
//==================================================================//
Table operator+(double x, const Table& t1) {
    return t1 + x;
}
//==================================================================//
Table operator-(const Table& t1, const Table& t2) {
    return t1 + (-t2);
}
//==================================================================//
Table operator-(const Table& t1, double x) {
    return t1 + (-x);
}
//==================================================================//
Table operator-(double x, const Table& t1) {
    return x + (-t1);
}
//==================================================================//
Table operator*(const Table& t1, double x) {
    int s1 = t1.get_size1();
    int s2 = t1.get_size2();
    int s3 = t1.get_size3();
    int size = s1 * s2 * s3;

    Table result(s1, s2, s3);
    for (int i=0; i<size; i++) result.tab[i] = t1.tab[i] * x;
    return result;
}
//==================================================================//
Table operator*(double x, const Table& t1) {
    return t1 * x;
}
//==================================================================//
Table operator/(const Table& t1, const Table& t2) {
    int s1 = t1.get_size1();
    int s2 = t1.get_size2();
    int s3 = t1.get_size3();
    assert ((t2.get_size1() == s1) && (t2.get_size2() == s2)
      && (t2.get_size3() == s3));
    int size = s1 * s2 * s3;

    Table result(s1, s2, s3);
    for (int i=0; i<size; i++) result.tab[i]  = t1.tab[i] / t2.tab[i];
    return result;
}
//==================================================================//
Table operator/(const Table& t1, double x) {
    assert(x > 1e-16);
    return t1 * (1.0 / x);
}
//==================================================================//
Table operator/(double x, const Table& t1) {
    int s1 = t1.get_size1();
    int s2 = t1.get_size2();
    int s3 = t1.get_size3();

    Table result(s1, s2, s3) ;
    result = x;
    return result/t1;
}
//==================================================================//
Table sqrt(const Table& table_in) {
    int s1 = table_in.get_size1();
    int s2 = table_in.get_size2();
    int s3 = table_in.get_size3();
    int size = s1 * s2 * s3;

    Table result(s1, s2, s3);
    for (int i=0; i<size; i++) result.tab[i] = sqrt(table_in.tab[i]);
    return result;
}
//==================================================================//
Table pow(const Table& table_in, double x) {
  int s1 = table_in.get_size1();
  int s2 = table_in.get_size2();
  int s3 = table_in.get_size3();
  int size = s1 * s2 * s3;

  Table result(s1, s2, s3);
  for (int i=0; i<size; i++) result.tab[i] = pow(table_in.tab[i], x);
  return result;
}
//==================================================================//
void Table::save(const char* fname) const {
    assert(tab != 0x0);

    ofstream file(fname);
    int size = size1*size2*size3;
    file << size1 << ' ' << size2 << ' ' << size3 << endl;
    file << setprecision(16) ;
    for (int i=0; i<size; i++)
        file << tab[i] << ' ';
    file << endl ;
}
