// A C++ version of K2-CPM code.
//==================================================================//

#include <iostream>
#include<fstream>
#include<iomanip>
#include <cmath>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <sstream>

#include "libcpm.h"

using namespace std;

//==================================================================//
// Functions
//==================================================================//
void linear_least_squares(Table* a, Table* y, Table* yvar,
    const Table* l2_tab, Table* result){
/*
    Solver of linear systems using Cholesky's decomposition. Let's define
    a matrix A (dimension n1 x n2) and two vectors X (dimension n2) and Y
    (dimension n1). This function solves the equation AX = Y. The
    following steps are considered:

    1/ Add the observational uncertainties to the data.

    2/ Compute the square matrix A^T A and the vector A^T Y.

    3/ Find the square matrix L so that A^T A = L L^T, where L^T is the
    transposition of L (L is triangular and lower).

    4/ Find the solution X of the system A^T A X = L L^T X = A^T Y.

    The solver assumes that the matrix A^T A is a square matrix, symmetric
    and positive-definite.

    Inputs
    ------
    a -- Pointer to Table of dimension n_data x n_predictors
        The basis matrix.
    y -- Pointer to Table with dimension n_data
        The observations.
    yvar -- Pointer to Table with dimension n_data
        The observational variance of the points y.
    l2 -- Pointer to Table with dimension n_predictors
        The L2 regularization strength.
    result -- Pointer to Table with dimension n_predictors
        The solution will be written in this Table.
*/

    // Declarations and initialization
    // -------------------------------
    int i, j, k, dim1a, dim2a, x;
    double s;

    dim1a = a->get_size1();
    dim2a = a->get_size2();
    x = y->get_size1();
    assert(dim1a == x);
    x = y->get_size2();
    assert(x == 1);
    x = y->get_size3();
    assert(x == 1);

    x = yvar->get_size1();
    assert(dim1a == x);
    x = yvar->get_size2();
    assert(x == 1);
    x = yvar->get_size3();
    assert(x == 1);

    x = l2_tab->get_size1();
    assert(dim2a == x);
    x = l2_tab->get_size2();
    assert(x == 1);
    x = l2_tab->get_size3();
    assert(x == 1);

    Matrix ata(dim2a);
    Table cia(dim1a, dim2a), at(dim2a, dim1a), ciy(dim1a), b(dim2a);

    // Incorporate observational uncertainties
    // ---------------------------------------
    for(i=0; i<dim1a; ++i){
        for(j=0; j<dim2a; ++j){
            cia.set(i, j) = (*a)(i, j) / (*yvar)(i);
            at.set(j, i) = (*a)(i, j);  // compute transpose of a
        }
    }
    ciy = (*y) / (*yvar);

    // Compute the pre-factor
    // ----------------------
    for(i = 0; i < dim2a; ++i){
        for(j = 0; j < dim2a; ++j){
            s = 0;
            for(k = 0; k < dim1a; ++k) s += at(i, k) * cia(k, j);
            ata.set(i, j) = s;
        }
    }

    for(i=0; i<dim2a; ++i){
        s = 0;
        for(j=0; j<dim1a; ++j) s += at(i, j) * ciy(j);
        b.set(i) = s;
    }

    // Incorporate any L2 regularization
    // ---------------------------------
    for(i = 0; i < dim2a; ++i) { ata.set(i, i) += (*l2_tab)(i); }

    // Solve the equations overwriting the matrix and tables
    // -----------------------------------------------------
    ata.cholesky_factor();
    ata.cholesky_solve(b);

    *result = b;
}
//==================================================================//
void fit_target(const Table& tpf_timeserie, Table& pre_matrix2,
    Table& l2_tab, double* train_lim, Table& result){
/*
    Fit the fluxes of the pixels.

    Input
    -----
    target_flux -- Table, dimension n_dates
        The target flux.
    predictor_flux_matrix -- Table, dimension n_dates x n_pred
        The flux of nearby stars used in the fitting process.
    time -- Table, dimension n_dates
        Date of the observations.
    covar_list -- Table, dimension n_dates
        List of the standard deviation for the predictors.
    l2_vector -- Table, dimension n_pred
        Array of L2 regularization strength.
    train_lim -- array of double, dimension 2.
        The dates between train_lim[0] and train_lim[1] are excluded from
        the fit.
    result -- Table, dimension n_pred
        Result of the fit will be written in this Table.
*/

    // Declarations and initializations
    // --------------------------------
    int i, i2, j, n_dates, n_dates2, n_pre;

    n_dates = tpf_timeserie.get_size1();
    n_pre = pre_matrix2.get_size2();

    // Size of the train window
    // ------------------------
    if ((train_lim[0]>0) && (train_lim[1]>train_lim[0])) {
        n_dates2 = 0;
        for(i=0; i<n_dates; ++i) {
            if ((tpf_timeserie(i, 0)<train_lim[0]) || (tpf_timeserie(i, 0)>train_lim[1])) ++n_dates2;
        }
    }
    else n_dates2 = n_dates;

    // Fit
    // ---
    Table y(n_dates2), yvar(n_dates2);
    if(n_dates2 == n_dates) {
        for(i=0; i<n_dates2; ++i) {
            y.set(i) = tpf_timeserie(i, 1);
            // yvar.set(i) = pow(tpf_timeserie(i, 2), 2);  // --> Commented to follow python version.
        }
        yvar = 1.0;
        linear_least_squares(&pre_matrix2, &y, &yvar, &l2_tab, &result);
    }
    else {
        Table pre_matrix3(n_dates2, n_pre);
        i2=0;
        for(i=0; i<n_dates; ++i) {
            if ((tpf_timeserie(i, 0)<train_lim[0]) || (tpf_timeserie(i, 0)>train_lim[1])) {
                for(j=0; j<n_pre; ++j) {
                    pre_matrix3.set(i2, j) = pre_matrix2(i, j);
                }
                y.set(i2) = tpf_timeserie(i, 1);
                // yvar.set(i) = pow(tpf_timeserie(i, 2), 2);  // --> Commented to follow python version.
                ++i2;
            }
        }
        yvar = 1.0;
        linear_least_squares(&pre_matrix2, &y, &yvar, &l2_tab, &result);
    }
}
//==================================================================//
void get_fit_matrix_ffi(Table& pre_matrix, int n_dates, int n_pre,
    int poly, Table& pre_matrix2){
/*
    Prepare matrix to fit the fluxes.

    Input
    -----
    target_flux -- Table, dimension n_dates
        The target flux (masks already included).
    predictor_matrix -- Table, dimension n_dates x n_pred
        The flux of nearby stars used in the fitting process (masks already
        included).
    time -- Table, dimension n_dates
        Date of the observations.
    poly -- int
        Order of polynomials on time to be added.
    ml -- ?
        ?
    predictor_matrix_mp -- Table, dimension n_dates x (n_pred + poly + 1)
        Same as predictor_matrix with polynomial terms.
*/

    // Add polynomial terms
    // --------------------
    // Concatenate with the Vandermonde matrix
    int n_pre2 = n_pre + poly + 1;
    for(int i=0; i<n_dates; ++i) {
        for(int j=0; j<n_pre2; ++j){
            if (j<n_pre) pre_matrix2.set(i, j) = pre_matrix(i, j);
            if ((j>=n_pre) && (j<n_pre2)) pre_matrix2.set(i, j) = pow(i, j-n_pre);
            // if (j==n_pred+poly+1) pre_matrix2.set(i, j) = ml(i);
        }
    }

    // ************ !!! ADD HERE CONCATENATION WITH ml !!! ************
}
//==================================================================//
void cpm_part2(string path_input, string prefix){

    // Declaration and initialisations
    // -------------------------------
    int i, i2, j, poly=0, lsize;
    int n_dates, n_pre, n_pre2, n_pre_dates;
    int n_pred, n_dates_wmask, n_pred_poly, * epoch_mask;
    double l2 = 1000.0, x;
    double train_lim[2];
    string pixel_flux_fname, epoch_mask_fname, pre_matrix_fname;
    string pre_epoch_mask_fname, ml_model_fname, result_fname, cpmflux_fname;
    string predicted_flux_fname;

    string line, last_line, lastline, delimiter, auxstring;

    train_lim[0] = -1;
    train_lim[1] = -1;

    // Define file names
    auxstring = path_input + prefix;
    pixel_flux_fname = auxstring + "_pixel_flux.cpp.dat";
    epoch_mask_fname = auxstring + "_epoch_mask.cpp.dat";
    pre_matrix_fname = auxstring + "_pre_matrix_xy.cpp.dat";
    pre_epoch_mask_fname = auxstring + "_predictor_epoch_mask.cpp.dat";
    ml_model_fname = auxstring + "_time_magnification.cpp.dat";
    result_fname = auxstring + "_results.dat";
    predicted_flux_fname = auxstring + "_predicted_flux.dat";
    cpmflux_fname = auxstring + "_cpmflux.dat";

    // Load TPF data
    // -------------
    Table tpf_timeserie(pixel_flux_fname.c_str(), epoch_mask_fname.c_str(), 0);
    n_dates = tpf_timeserie.get_size1();

    // Load predictors matrix
    // ----------------------
    Table pre_matrix(pre_matrix_fname.c_str());
    n_pre_dates = pre_matrix.get_size1();
    n_pre = pre_matrix.get_size2();

    // Calculations
    // ------------
    n_pre2 = n_pre + poly + 1;
    // n_pred_poly = n_pred + poly + 1 + 1;  // +1 for poly +1 for ml

    // Add polynomial terms to predictor matrix
    Table pre_matrix2(n_dates, n_pre2);
    assert(poly >= 0);
    get_fit_matrix_ffi(pre_matrix, n_dates, n_pre, poly, pre_matrix2);

    // Prepare regularization
    Table l2_tab(n_pre2);
    l2_tab = l2;
    if (n_dates < n_pre2) for(i=n_dates; i<n_pre2; ++i) l2_tab.set(i) = 0.0;

    // Prepare uncertainties
//    Table covar_list(n_dates);
//    covar_list = tpf_flux_err;

    // Fit target
    Table result(n_pre2);
    fit_target(tpf_timeserie, pre_matrix2, l2_tab, train_lim, result);

    Table flux_fit(n_dates);
    for(i=0; i<n_dates; ++i){
        x = 0;
        for(j=0; j<n_pre2; ++j) x += pre_matrix2(i, j) * result(j);
        flux_fit.set(i) = x;
    }

    Table dif(n_dates);
    for(i=0; i<n_dates; ++i) dif.set(i) = tpf_timeserie(i, 1) - flux_fit(i);

    // Save results in files
    // ---------------------
    ofstream result_file (result_fname);
    if (result_file.is_open()){
        result_file << fixed << setprecision(6);
        for (i=0; i<n_pre2; ++i) result_file << result(i) << endl;
        result_file.close();
    }
    else cout << "Unable to open file";

    ofstream cpmflux (cpmflux_fname);
    if (cpmflux.is_open()){
        for (i=0; i<n_dates; ++i) {
            cpmflux << fixed << setprecision(5);
            cpmflux << tpf_timeserie(i, 0) << " ";
            cpmflux << fixed << setprecision(8);
            cpmflux << flux_fit(i) << " " << dif(i) << endl;
        }
        cpmflux.close();
    }
    else cout << "Unable to open file";

    // Release memory
    // --------------
    delete[] epoch_mask;
}
//==================================================================//
int main(int argc, char* argv[]) {

    // Declarations
    // ------------
    string path_input, prefix;

    // Check command line options
    // --------------------------
    assert(argc == 3);

    // Run CPM part 2
    // --------------
    path_input = argv[1];
    prefix = argv[2];
    cpm_part2(path_input, prefix);

    return 0;
}
