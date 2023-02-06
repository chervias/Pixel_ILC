#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

void print_mat_contents(gsl_matrix *matrix,  int size);
void empty_mat_contents(gsl_matrix *matrix,  int size);
void invert_a_matrix_single(gsl_matrix_float *matrix, gsl_matrix_float *inv,  int size);
void invert_a_matrix(gsl_matrix *matrix, gsl_matrix *inv,  int size);

void pixelILC_DefineCovMat_NILC_SHTSmoothing_SingleField( long ipix,  int Nfreqs, double* TEBmaps, gsl_matrix *CovF,  int Nfreqs2);
void pixelILC_CalculateILCWeight_NILC_SingleField(double* a, gsl_matrix *CovFi, double* weights,  int Nfreqs,  int p);
void pixelILC_CalculateILCWeight_CNILC_SingleField(double* a, double* b, gsl_matrix *CovFi, double* weights,  int Nfreqs,  int p);
void pixelILC_DefineCovMat_NILC_CovarPixelSpace_SingleField(long ipix,  int Nfreqs, int nside, double* Covar_maps, double* Field_filtered_map, double* mask, long *pixel_buffer, gsl_matrix *CovF,  int Nfreqs2, double fwhm);