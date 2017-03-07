#ifndef __LIBCPM_H_
#define __LIBCPM_H_

#include "matrix.h"

void linear_least_squares(Table*, Table*, const Table*, const Table*, Table*);
void fit_target(Table&, Table&, Table&, Table&, double*, Table&);
void get_fit_matrix_ffi(Table&, Table&, Table&, int, double, Table&);
void cpm_part2(int);

#endif
