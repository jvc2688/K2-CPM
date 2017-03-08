// A class that define a three-dimensional tables.
//==================================================================//

#ifndef __TABLE_H_
#define __TABLE_H_

#include<cassert>
#include<iostream>

using namespace std;

class Table {

    // Data
    protected:
    int size1;
    int size2;
    int size3;
    double* tab;

    // Constructors
    public:
    explicit Table(int d1, int d2=1, int d3=1);
    Table(const Table&);

    // Destructor
    virtual ~Table();

    // Definitions
    void operator=(const Table&);
    void operator=(double);


    // Reading access to data
    int get_size1() const { return size1; };
    int get_size2() const { return size2; };
    int get_size3() const { return size3; };

    double operator()(int i, int j=0, int k=0) const {
        assert ((i>=0) && (i<size1));
        assert ((j>=0) && (j<size2));
        assert ((k>=0) && (k<size3));
        return tab[(i * size2 + j) * size3 + k];
    };

    // Writing access to data
    double& set(int i, int j=0, int k=0){
        assert ((i>=0) && (i<size1));
        assert ((j>=0) && (j<size2));
        assert ((k>=0) && (k<size3));
        return tab[(i * size2 + j) * size3 + k];
    };

    // Useful functions
    protected:
    virtual void print(ostream&) const;  // Print the table

    friend ostream& operator<<(ostream&, const Table& );
    friend Table operator-(const Table&);
    friend Table operator+(const Table&, const Table&);
    friend Table operator+(const Table&, double);
    friend Table operator+(double, const Table&);
    friend Table operator-(const Table&, const Table&);
    friend Table operator-(const Table&, double);
    friend Table operator-(double, const Table&);
    friend Table operator*(const Table&, double);
    friend Table operator*(double, const Table&);
    friend Table operator/(const Table&, const Table&);
    friend Table operator/(const Table&, double);
    friend Table operator/(double, const Table&);

    friend Table sqrt(const Table&);
    friend Table pow(const Table&, double);
};

ostream& operator<<(ostream&, const Table& );
Table operator-(const Table&);
Table operator+(const Table&, const Table&);
Table operator+(const Table&, double);
Table operator+(double, const Table&);
Table operator-(const Table&, const Table&);
Table operator-(const Table&, double);
Table operator-(double, const Table&);
Table operator*(const Table&, double);
Table operator*(double, const Table&);
Table operator/(const Table&, const Table&);
Table operator/(const Table&, double);
Table operator/(double, const Table&);

Table sqrt(const Table&);
Table pow(const Table&, double);

#endif
