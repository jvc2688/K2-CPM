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

for(i=0; i<dim1a; ++i){
    if ((*yvar)(i)==0) cout << i << endl;
    }

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
        b.set(i) = 1;
    }

// *************************************
//    for(i=b.get_size1()-10; i<b.get_size1(); ++i){
//    for(i=0; i<10; ++i){
//        cout << b(i) << "\t";
//    }
//    cout << endl;
//cout << at.get_size1() << " " << at.get_size2() << endl;
//    for(i=at.get_size1()-10; i<at.get_size1(); ++i){
//        for(j=at.get_size2()-10; j<at.get_size2(); ++j){
//            cout << at(i, j) << "\t";
//        }
//        cout << endl;
//    }
// *************************************

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
    cout << n_dates2 << " " << n_dates << endl;

    // Fit
    // ---
    Table y(n_dates2), yvar(n_dates2);
    if(n_dates2 == n_dates) {
        for(i=0; i<n_dates2; ++i) {
            y.set(i) = tpf_timeserie(i, 1);
            yvar.set(i) = pow(tpf_timeserie(i, 2), 2);
        }
        linear_least_squares(&pre_matrix2, &y, &yvar, &l2_tab, &result);
    }
//    else {
//        Table predictor_flux_matrix_curr(n_trainmask, n_pred), target_flux_curr(n_trainmask);
//        i2=0;
//        for(i=0; i<n_dates; ++i) {
//            if ((time(i)<train_lim[0]) || (time(i)>train_lim[1])) {
//                for(j=0; j<n_pred; ++j) {
//                    predictor_flux_matrix_curr.set(i2, j) = predictor_flux_matrix(i, j);
//                }
//                target_flux_curr.set(i2) = target_flux(i);
//                covar_list_curr.set(i2) = covar_list(i);
//                i2++;
//            }
//        }
//        covar_list_curr = pow(covar_list_curr, 2);
//        linear_least_squares(&predictor_flux_matrix_curr, &target_flux_curr, &covar_list_curr, &l2_vector, &result);
//    }
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
    string pre_epoch_mask_fname, ml_model_fname, result_fname, dif_fname;
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
    result_fname = auxstring + "_results.cpp.dat";
    dif_fname = auxstring + "_dif.cpp.dat";

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
    for(i=0; i<n_dates; ++i) dif.set(i) = tpf_timeserie(i, 0) - flux_fit(i);

    // Save results in files
    // ---------------------
    ofstream result_file (result_fname);
    if (result_file.is_open()){
        result_file << fixed << setprecision(6);
        for (i=0; i<n_pre2; ++i) result_file << result(i) << endl;
        result_file.close();
    }
    else cout << "Unable to open file";

    ofstream dif_file (dif_fname);
    if (dif_file.is_open()){
        dif_file << fixed << setprecision(6);
        for (i=0; i<n_dates; ++i) dif_file << dif(i) << endl;
        dif_file.close();
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




//    // Number of dates
//    n_dates=0;
//    ifstream pixel_flux_file_lines (pixel_flux_fname);
//    if (pixel_flux_file_lines.is_open()){
//        while (pixel_flux_file_lines >> x >> x >> x) ++n_dates;
//        pixel_flux_file_lines.close();
//    }
//    else cout << "Unable to open file";
//    assert(n_dates > 0);
//
//    // Find quickly the number of predictors
//    delimiter = " ";
//    n_pred = 0;
//    ifstream file_pre_matrix_file_lines;  // Look directly to last line
//    file_pre_matrix_file_lines.open(pre_matrix_fname);
//    if (file_pre_matrix_file_lines.is_open()){
//        i2 = -1;
//        lastline="";
//        lsize=-1;
//        while((i2==-1) || (lsize <= lastline.size())){
//            lsize = lastline.size();
//            file_pre_matrix_file_lines.seekg (i2, file_pre_matrix_file_lines.end);
//            getline (file_pre_matrix_file_lines, lastline);
//            i2--;
//        }
//        getline (file_pre_matrix_file_lines, lastline);
//        file_pre_matrix_file_lines.close();
//
//        i = 0;  // Find n_pred value
//        string token;
//        while ((i = lastline.find(delimiter)) != string::npos) {
//            token = lastline.substr(0, i);
//            n_pred = stoi(token);
//            lastline.erase(0, i + delimiter.length());
//        }
//    }
//    else cout << "Unable to open file";
//    ++n_pred;
//    assert((n_pred>0));
//
//    // Load files
//    // ----------
//    // Load time and flux
//    n_dates_wmask = 0;
//    epoch_mask = new int [n_dates];
//    ifstream epoch_mask_file (epoch_mask_fname);
//    if (epoch_mask_file.is_open()){
//        for (i=0; i<n_dates; ++i){
//            epoch_mask_file >> line;
//            for(j=0; j<line.length(); j++) line[j] = toupper(line[j]);
//            assert((line=="TRUE") || (line=="FALSE"));
//            if(line=="TRUE") {
//                epoch_mask[i] = 1;
//                ++n_dates_wmask;
//            }
//            else epoch_mask[i] = 0;
//        }
//        epoch_mask_file.close();
//    }
//    else cout << "Unable to open file";
//    assert((n_dates_wmask>0) && (n_dates_wmask<=n_dates));
//
//    Table tpf_time(n_dates_wmask), tpf_flux(n_dates_wmask), tpf_flux_err(n_dates_wmask);
//    ifstream pixel_flux_file (pixel_flux_fname);
//    if (pixel_flux_file.is_open()){
//        i2 = 0;
//        for (i=0; i<n_dates; ++i){
//            if (epoch_mask[i]){
//                pixel_flux_file >> tpf_time.set(i2) >> tpf_flux.set(i2) >> tpf_flux_err.set(i2);
//                ++i2;
//            }
//            else pixel_flux_file >> x >> x >> x;
//        }
//        pixel_flux_file.close();
//    }
//    else cout << "Unable to open file";
//    assert(i2==n_dates_wmask);
//
//    // Load predictor matrix
//    //Table pre_matrix(n_dates_wmask, n_pred);
//    ifstream file_pre_matrix_file (pre_matrix_fname);
//    if (file_pre_matrix_file.is_open()){
//        for (i=0; i<n_dates_wmask; ++i){
//            for (j=0; j<n_pred; ++j){
//                file_pre_matrix_file >> x >> x >> pre_matrix.set(i, j);
//            }
//        }
//        file_pre_matrix_file.close();
//    }
//    else cout << "Unable to open file";

    // Load microlensing model
/*    int n_ml=0;
    ifstream f1_mask (ml_model_fname);
    if (f1_mask.is_open()){
        while(f1_mask >> x) ++n_ml;
    }
    else cout << "Unable to open file";
    f1_mask.close();

    Table ml_mask(n_ml);
    ifstream f2_mask (pre_epoch_mask_fname);
    if (f2_mask.is_open()){
        for (i=0; i<n_ml; ++i){
            f2_mask >> i2;
            if(i2) {
                ml_mask.set(i) = 1;
            }
            else ml_mask.set(i) = 0;
        }
    }
    else cout << "Unable to open file";
    f2_mask.close();

    Table ml(n_dates);
    ifstream f1_ml (ml_model_fname);
    if (f1_ml.is_open()){
        i2 = 0;
        for (i=0; i<n_ml; ++i){
            f1_ml >> x >> x;
            if(ml_mask(i)) {
                ml.set(i2) = x;
                i2++;
            }
        }
    }
    else cout << "Unable to open file";
    f1_ml.close();*/