#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>

void pixelILC_DefineCovarianceMatrix(unsigned long ipix, unsigned int nside, unsigned int Nfreqs, PyObject* TQUmaps, double sigma, gsl_matrix *CovT, gsl_matrix *CovQ, gsl_matrix *CovU);
void invert_a_matrix(gsl_matrix *matrix, gsl_matrix *inv, unsigned int size);
void pixelILC_CalculateILCWeight(PyObject* a, gsl_matrix *CovTi, gsl_matrix *CovQi, gsl_matrix *CovUi, double* weights, unsigned int Nfreqs, unsigned int p);
void print_mat_contents(gsl_matrix *matrix, unsigned int size);
void empty_mat_contents(gsl_matrix *matrix, unsigned int size);