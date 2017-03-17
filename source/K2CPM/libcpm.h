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
//
// This code is a C++ adaptation of the K2-CPM code [1][2][3].
//
// References
// ----------
// [1] Wang, D., Hogg, D. W., Foreman-Mackey, D. & Schölkopf, B. A Causal,
//     Data-driven Approach to Modeling the Kepler Data. Publications of the
//     Astronomical Society of the Pacific 128, 94503 (2016).
// [2] https://github.com/jvc2688/K2-CPM
// [3] https://github.com/rpoleski/K2-CPM
//
//==================================================================//

#ifndef __LIBCPM_H_
#define __LIBCPM_H_

#include "matrix.h"

void linear_least_squares(Table*, Table*, const Table*, const Table*, Table*);
void fit_target(const Table&, Table&, const Table&, const double*, Table&);
void get_fit_matrix_ffi(const Table&, const int, const int, const int, Table&);
void cpm_part2(string, string, double);

#endif
