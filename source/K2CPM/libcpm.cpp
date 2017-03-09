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
void linear_least_squares(Table* a, Table* y, const Table* yvar, const Table* l2, Table* result){
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

    x = l2->get_size1();
    assert(dim2a == x);
    x = l2->get_size2();
    assert(x == 1);
    x = l2->get_size3();
    assert(x == 1);

    Matrix ata(dim2a);
    Table cia(dim1a, dim2a), at(dim2a, dim1a), ciy(dim1a), b(dim2a);

    // Incorporate observational uncertainties
    // ---------------------------------------
    for(i = 0; i < dim1a; ++i){
        for(j = 0; j < dim2a; ++j){
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
            for(k = 0; k < dim1a; ++k) s += (at)(i, k) * cia(k, j);
            ata.set(i, j) = s;
        }
    }

    for(i = 0; i < dim2a; ++i){
        s = 0;
        for(j = 0; j < dim1a; ++j) s += at(i, j) * ciy(j);
        b.set(i) = s;
    }

    // Incorporate any L2 regularization
    // ---------------------------------
    for(i = 0; i < dim2a; ++i) { ata.set(i, i) += (*l2)(i); }

    // Solve the equations overwriting the matrix and tables
    // -----------------------------------------------------
    ata.cholesky_factor();
    ata.cholesky_solve(b);

    *result = b;
}
//==================================================================//
void fit_target(Table& target_flux, Table& predictor_flux_matrix,
        Table& time, Table& covar_list, Table& l2_vector, double* train_lim,
        Table& result){
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
    int i, i2, j, n_dates, n_trainmask, n_pred;

    n_dates = time.get_size1();
    n_pred = predictor_flux_matrix.get_size2();

    // Size of the mask
    // ----------------
    if ((train_lim[0]>0) && (train_lim[1]>train_lim[0])) {
        n_trainmask = 0;
        for(i=0; i<n_dates; ++i) {
            if ((time(i)<train_lim[0]) || (time(i)>train_lim[1])) ++n_trainmask;
        }
    }
    else n_trainmask = n_dates;

    // Fit
    // ---
    Table covar_list_curr(n_trainmask);
    if(n_dates == n_trainmask) {
        covar_list_curr = covar_list;
        covar_list_curr = pow(covar_list_curr, 2);
        linear_least_squares(&predictor_flux_matrix, &target_flux, &covar_list_curr, &l2_vector, &result);
    }
    else {
        Table predictor_flux_matrix_curr(n_trainmask, n_pred), target_flux_curr(n_trainmask);
        i2=0;
        for(i=0; i<n_dates; ++i) {
            if ((time(i)<train_lim[0]) || (time(i)>train_lim[1])) {
                for(j=0; j<n_pred; ++j) {
                    predictor_flux_matrix_curr.set(i2, j) = predictor_flux_matrix(i, j);
                }
                target_flux_curr.set(i2) = target_flux(i);
                covar_list_curr.set(i2) = covar_list(i);
                i2++;
            }
        }
        covar_list_curr = pow(covar_list_curr, 2);
        linear_least_squares(&predictor_flux_matrix_curr, &target_flux_curr, &covar_list_curr, &l2_vector, &result);
    }
}
//==================================================================//
void get_fit_matrix_ffi(Table& target_flux, Table& predictor_matrix,
    Table& time, int poly, Table& ml, Table& predictor_matrix_mp){
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

    // Declarations and initializations
    int i, j, n_dates, n_pred;

    n_dates = time.get_size1();
    n_pred = predictor_matrix.get_size2();

    // Add polynomial terms
    // --------------------
    // Concatenate with the Vandermonde matrix
    assert(poly >= 0);
    for(i=0; i<n_dates; ++i) {
        for(j=0; j<n_pred+poly+1+1; ++j){
            if (j<n_pred) predictor_matrix_mp.set(i, j) = predictor_matrix(i, j);
            if ((j>=n_pred) && (j<n_pred+poly+1)) predictor_matrix_mp.set(i, j) = pow(i, j-n_pred);
            if (j==n_pred+poly+1) predictor_matrix_mp.set(i, j) = ml(i);
        }
    }

    // ************ !!! ADD HERE CONCATENATION WITH ml !!! ************
}
//==================================================================//
void cpm_part2(int n_test=1){

    // Declaration and initialisations
    int i, i2, j, poly=0, lsize;
    int n_dates, n_pred, n_dates_wmask, n_pred_poly, * epoch_mask;
    double l2 = 0.0, x;
    double train_lim[2];
//    int * pixel;
    string line, last_line, lastline, delimiter;

    train_lim[0] = -1;
    train_lim[1] = -1;

//    pixel = new int [2];
//    pixel[0] = 883;
//    pixel[1] = 670;
//    pixel[0] = 883;
//    pixel[1] = 670;

    // Input files
    string in_directory ("../test/intermediate/");
    // string in_directory2 ("../test/intermediate/expected/");
    string out_directory ("../test/output/");
    string pre_matrix_file ("-pre_matrix_xy.dat");
    pre_matrix_file = in_directory + to_string(n_test) + pre_matrix_file;
    string pixel_flux_file_name ("-pixel_flux.dat");
    pixel_flux_file_name = in_directory + to_string(n_test) + pixel_flux_file_name;
    string epoch_mask_file_name ("-epoch_mask.dat");
    epoch_mask_file_name = in_directory + to_string(n_test) + epoch_mask_file_name;
    string fname_ml ("-time_magnification.dat");
    fname_ml = in_directory + to_string(n_test) + fname_ml;
    string fname_mask ("-mask.dat");
    fname_mask = in_directory + to_string(n_test) + fname_mask;

    // Output files
    string result_file_name ("-result.dat");
    result_file_name = out_directory + to_string(n_test) + result_file_name;
    string dif_file_name ("-dif.dat");
    dif_file_name = out_directory + to_string(n_test) + dif_file_name;

    // Dimensions of the tables
    // ------------------------

    // Number of dates
    n_dates=0;
    ifstream pixel_flux_file_lines (pixel_flux_file_name);
    if (pixel_flux_file_lines.is_open()){
        while (pixel_flux_file_lines >> x >> x >> x) ++n_dates;
        pixel_flux_file_lines.close();
    }
    else cout << "Unable to open file";
    assert(n_dates > 0);

    // Find quickly the number of predictors
    delimiter = " ";
    n_pred = 0;
    ifstream file_pre_matrix_file_lines;  // Look directly to last line
    file_pre_matrix_file_lines.open(pre_matrix_file);
    if (file_pre_matrix_file_lines.is_open()){
        i2 = -1;
        lastline="";
        lsize=-1;
        while((i2==-1) || (lsize <= lastline.size())){
            lsize = lastline.size();
            file_pre_matrix_file_lines.seekg (i2, file_pre_matrix_file_lines.end);
            getline (file_pre_matrix_file_lines, lastline);
            i2--;
        }
        getline (file_pre_matrix_file_lines, lastline);
        file_pre_matrix_file_lines.close();

        i = 0;  // Find n_pred value
        string token;
        while ((i = lastline.find(delimiter)) != string::npos) {
            token = lastline.substr(0, i);
            n_pred = stoi(token);
            lastline.erase(0, i + delimiter.length());
        }
    }
    else cout << "Unable to open file";
    ++n_pred;
    assert((n_pred>0));

    // Load files
    // ----------
    // Load time and flux
    n_dates_wmask = 0;
    epoch_mask = new int [n_dates];
    ifstream epoch_mask_file (epoch_mask_file_name);
    if (epoch_mask_file.is_open()){
        for (i=0; i<n_dates; ++i){
            epoch_mask_file >> line;
            for(j=0; j<line.length(); j++) line[j] = toupper(line[j]);
            assert((line=="TRUE") || (line=="FALSE"));
            if(line=="TRUE") {
                epoch_mask[i] = 1;
                ++n_dates_wmask;
            }
            else epoch_mask[i] = 0;
        }
        epoch_mask_file.close();
    }
    else cout << "Unable to open file";
    assert((n_dates_wmask>0) && (n_dates_wmask<=n_dates));

    Table tpf_time(n_dates_wmask), tpf_flux(n_dates_wmask), tpf_flux_err(n_dates_wmask);
    ifstream pixel_flux_file (pixel_flux_file_name);
    if (pixel_flux_file.is_open()){
        i2 = 0;
        for (i=0; i<n_dates; ++i){
            if (epoch_mask[i]){
                pixel_flux_file >> tpf_time.set(i2) >> tpf_flux.set(i2) >> tpf_flux_err.set(i2);
                ++i2;
            }
            else pixel_flux_file >> x >> x >> x;
        }
        pixel_flux_file.close();
    }
    else cout << "Unable to open file";
    assert(i2==n_dates_wmask);

    // Load predictor matrix
    Table pre_matrix(n_dates_wmask, n_pred);
    ifstream file_pre_matrix_file (pre_matrix_file);
    if (file_pre_matrix_file.is_open()){
        for (i=0; i<n_dates_wmask; ++i){
            for (j=0; j<n_pred; ++j){
                file_pre_matrix_file >> x >> x >> pre_matrix.set(i, j);
            }
        }
        file_pre_matrix_file.close();
    }
    else cout << "Unable to open file";

    // Load microlensing model
    int n_ml=0;
    ifstream f1_mask (fname_mask);
    if (f1_mask.is_open()){
        while(f1_mask >> x) ++n_ml;
    }
    else cout << "Unable to open file";
    f1_mask.close();

    Table ml_mask(n_ml);
    ifstream f2_mask (fname_mask);
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
    ifstream f1_ml (fname_ml);
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
    f1_ml.close();

    // Calculations
    // ------------
    n_pred_poly = n_pred + poly + 1 + 1;  // +1 for poly +1 for ml

    // Prepare flux matrix to fit
    Table predictor_matrix_mp(n_dates_wmask, n_pred_poly);
    get_fit_matrix_ffi(tpf_flux, pre_matrix, tpf_time, poly, ml, predictor_matrix_mp);

    // Prepare regularization
    Table l2_vector(n_pred_poly);
    l2_vector = l2;
    if (n_dates_wmask < n_pred_poly){
        for(i=n_dates_wmask; i<n_pred_poly; ++i) l2_vector.set(i) = 0.0;
    }

    // Prepare uncertainties
    Table covar_list(n_dates_wmask);
    covar_list = tpf_flux_err;

    // Fit target
    Table result(n_pred_poly);
    fit_target(tpf_flux, predictor_matrix_mp, tpf_time, covar_list, l2_vector, train_lim, result);

    Table fit_flux(n_dates_wmask);
    for(i=0; i<n_dates_wmask; ++i){
        x = 0;
        for(j=0; j<n_pred_poly; ++j) x += predictor_matrix_mp(i, j) * result(j);
        fit_flux.set(i) = x;
    }

    Table dif(n_dates_wmask);
    dif = tpf_flux - fit_flux;

    // Save results in files
    // ---------------------
    ofstream result_file (result_file_name);
    if (result_file.is_open()){
        result_file << fixed << setprecision(6);
        for (i=0; i<n_pred_poly; ++i) result_file << result(i) << endl;
        result_file.close();
    }
    else cout << "Unable to open file";

    ofstream dif_file (dif_file_name);
    if (dif_file.is_open()){
        dif_file << fixed << setprecision(6);
        for (i=0; i<n_dates_wmask; ++i) dif_file << dif(i) << endl;
        dif_file.close();
    }
    else cout << "Unable to open file";

    // Release memory
    // --------------
//    delete[] pixel;
    delete[] epoch_mask;
}
//==================================================================//
int main(int argc, char* argv[]) {

    // Declarations
    // ------------
    int i=1;

    // Take into account command-line arguments
    // ----------------------------------------

    if(argc==2) i = stoi(argv[1]);

    // Run the independent part of CPM
    // -------------------------------
    cpm_part2(i);

    return 0;
}

// ./libcpm inputdir stem