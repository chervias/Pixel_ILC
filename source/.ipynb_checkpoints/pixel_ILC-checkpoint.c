#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <query_disc_wrapper.h>

void invert_a_matrix(gsl_matrix *matrix, gsl_matrix *inv, int size){
    gsl_permutation *p = gsl_permutation_alloc(size);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);
}

void print_mat_contents(gsl_matrix *matrix,  int size){
     int i, j;
    double element;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            element = gsl_matrix_get(matrix, i, j);
            printf("%E ", element);
        }
        printf("\n");
    }
}

void empty_mat_contents(gsl_matrix *matrix,  int size){
     int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {    
            // set entry at i, j to random_value
            gsl_matrix_set(matrix, i, j, 0.0);
        }
    }
}

void pixelILC_DefineCovMat_NILC_SHTSmoothing_SingleField(long ipix,  int Nfreqs, double* TEBmaps, gsl_matrix *CovF,  int Nfreqs2){
	// 
	int n,nn,c,ipix_int;
	double vF;
	//float vF_float;
	c = 0;
	ipix_int = (int) ipix ;
	for(n=0;n<Nfreqs;n++){
		for(nn=n;nn<Nfreqs;nn++){
			// TEBmaps is a numpy array with shape npix_per_window,Nfreqs2 = Nfreqs*(Nfreqs+1)/2
			vF = TEBmaps[ipix_int*Nfreqs2 + c] ;
			gsl_matrix_set(CovF, n, nn, vF );
			if(n!=nn){
				gsl_matrix_set(CovF, nn, n, vF );
			}
			c = c + 1;
		}
	}
}

void pixelILC_DefineCovMat_NILC_CovarPixelSpace_SingleField(long ipix,  int Nfreqs, int nside, double* Covar_maps, double* Field_filtered_map, double* mask, long *pixel_buffer, gsl_matrix *CovF,  int Nfreqs2, double fwhm){
	// 
	int sucess,nipix;
	int n,nn,ii,c,ipix_int,ipix2_int;
	ipix_int = (int) ipix;
	c = 0;
	for(n=0;n<Nfreqs;n++){
		for(nn=n;nn<Nfreqs;nn++){
			// we need to know the pixels in the disc shaped domain around ipix, we use query_disc for that
			query_disc_wrapper(ipix, 0.5*fwhm, nside, pixel_buffer, &nipix, &sucess);
			// now the pixels in the disc are in the array pixel_buffer with size nipix, we loop over them summing
			for(ii=0;ii<nipix;ii++){
				ipix2_int = (int) pixel_buffer[ii];
				Covar_maps[ipix_int*Nfreqs2 + c] += Field_filtered_map[n*12*nside*nside + ipix2_int] * Field_filtered_map[nn*12*nside*nside + ipix2_int] * mask[ipix2_int] ;
			}
			gsl_matrix_set(CovF, n, nn, Covar_maps[ipix_int*Nfreqs2 + c] );
			if(n!=nn){
				gsl_matrix_set(CovF, nn, n, Covar_maps[ipix_int*Nfreqs2 + c] );
			}
			c += 1;
		}
	}
}

void pixelILC_CalculateILCWeight_NILC_SingleField(double* a, gsl_matrix *CovFi, double* weights,  int Nfreqs,  int p){
	// shape of weights Npixels_*Nfreqs_
	double aCia_F=0.0;
	int i,j;
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			aCia_F += a[i] * gsl_matrix_get(CovFi,i,j) * a[j] ;
		}
	}
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			// This is the F weight
			weights[p*Nfreqs + i] += a[j] * gsl_matrix_get(CovFi,j,i) / aCia_F ;
		}
	}
	// after this weights will have the calculated weights.
}

void pixelILC_CalculateILCWeight_CNILC_SingleField(double* a, double* b, gsl_matrix *CovFi, double* weights,  int Nfreqs,  int p){
	// shape of weights Npixels_*Nfreqs_
	// Ci means covariance inverse
	double aCia_F=0.0,aCib_F=0.0,bCib_F=0.0;
	double up,down ;
	int i,j;
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			aCia_F += a[i] * gsl_matrix_get(CovFi,i,j) * a[j] ;
			aCib_F += a[i] * gsl_matrix_get(CovFi,i,j) * b[j] ;
			bCib_F += b[i] * gsl_matrix_get(CovFi,i,j) * b[j] ;
		}
	}
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			// This is the F weight
			// eq. 19 in arXiv:2006.0862
			up = bCib_F*a[j]*gsl_matrix_get(CovFi,j,i) - aCib_F*b[j]*gsl_matrix_get(CovFi,j,i) ;
			down = aCia_F * bCib_F - aCib_F*aCib_F ;
			weights[p*Nfreqs + i] += (up / down) ;
		}
	}
	// after this weights will have the calculated weights.
}