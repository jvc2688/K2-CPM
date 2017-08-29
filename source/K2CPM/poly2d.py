import numpy as np
from math import fsum


def degree_to_n_coefs(degree):
    """how many coefficients has a 2d polynomial of given degree"""
    return int((degree+1)*(degree+2)/2.+0.5)

def n_coefs_to_degree(n_coefs):
    """what is degree if 2d polynomial has n_coefs coeficients"""
    delta_sqrt = int((8 * n_coefs + 1.)**.5 + 0.5)
    if delta_sqrt**2 != (8*n_coefs+1.):
        raise ValueError('Wrong input in n_coefs_to_degree(): {:}'.format(n_coefs))
    return int((delta_sqrt - 3.) / 2. + 0.5)

def eval_poly_2d_no_coefs(x, y, deg):
    """evaluate powers of given values and return as a table: [1, x, y, x^2, xy, y^2] for deg = 2"""
    pow_x = np.polynomial.polynomial.polyvander(x, deg)
    pow_y = np.polynomial.polynomial.polyvander(y, deg)
    results = []
    for i in range(deg+1):
        for j in range(i+1):
            results.append(pow_x[:,i-j]*pow_y[:,j])
    return np.array(results)

def eval_poly_2d_coefs(x, y, coefs):
    """evaluate 2d polynomial without summing up"""
    c = np.copy(coefs).reshape(coefs.size, 1)
    deg = n_coefs_to_degree(len(c))
    return c * eval_poly_2d_no_coefs(x=x, y=y, deg=deg)

def eval_poly_2d(x, y, coefs):
    """evaluate 2d polynomial"""
    monomials = eval_poly_2d_coefs(x=x, y=y, coefs=coefs)
    results = []
    for i in range(monomials.shape[1]):
        results.append(fsum(monomials[:,i]))
    return np.array(results)

def fit_two_poly_2d(x_in, y_in, x_out, y_out, degree):
    """fits 2 polynomials: x_out = f(x_in, y_in, coefs) and same for y_out"""
    basis = eval_poly_2d_no_coefs(x_in, y_in, degree).T
    (coeffs_x, residuals_x, rank_x, singular_x) = np.linalg.lstsq(basis, x_out)
    (coeffs_y, residuals_y, rank_y, singular_y) = np.linalg.lstsq(basis, y_out)
    return (coeffs_x, coeffs_y)


