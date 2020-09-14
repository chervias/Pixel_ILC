#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <query_disc_wrapper.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>

void pixelILC_DefineCovarianceMatrix(unsigned long ipix, unsigned int nside, unsigned int Nfreqs, PyObject* TQUmaps, double sigma, gsl_matrix *CovT, gsl_matrix *CovQ, gsl_matrix *CovU){
	// First, we need to determine which pixels are within the radius (which is in radians)
	unsigned int i,n,nn,p;
	unsigned int npix_max = 200000;
	double radius = 5 * sigma ;
	unsigned long* ipix_arr = calloc(npix_max,sizeof(long));
	double* pixel_distances = calloc(npix_max,sizeof(double));
	unsigned long nipix , ipix_p;
	//printf("pixel %i\n",ipix);
	query_disc_wrapper(ipix,radius,nside,ipix_arr,&nipix,pixel_distances);
	// Now in nipix we have the number of pixels inside the disc
	//printf("There are %i pixels in the Domain of pixel %i\n",nipix,ipix);
	// Cov will have shape [3,Nfreqs,Nfreqs] with indices i,n,nn
	// This is how it is indexed Cov[ i*Nfreqs*Nfreqs + n *Nfreqs + nn ]
	// We iterate over the pixels
	for(p=0;p<nipix;p++){
		ipix_p = ipix_arr[p];
		//printf("pixel %i",ipix_p);
		// we iterate over the bands
		for(n=0;n<Nfreqs;n++){
			for(nn=n;nn<Nfreqs;nn++){
				// TQUmaps shape [Nfreqs,3,npix]
				//(1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,0,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,0,ipix_p))
				//printf("Value %f\n", pixel_distances[p] );
				gsl_matrix_set(CovT, n, nn, gsl_matrix_get(CovT,n,nn) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,0,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,0,ipix_p)) );
				gsl_matrix_set(CovQ, n, nn, gsl_matrix_get(CovQ,n,nn) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,1,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,1,ipix_p)) );
				gsl_matrix_set(CovU, n, nn, gsl_matrix_get(CovU,n,nn) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,2,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,2,ipix_p)) );
				if(n!=nn){
					// We also copy the symmetric, we swap n and nn
					gsl_matrix_set(CovT, nn, n, gsl_matrix_get(CovT,nn,n) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,0,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,0,ipix_p)) );
					gsl_matrix_set(CovQ, nn, n, gsl_matrix_get(CovQ,nn,n) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,1,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,1,ipix_p)) );
					gsl_matrix_set(CovU, nn, n, gsl_matrix_get(CovU,nn,n) + (1.0/nipix)*(1.0/2.0/M_PI/pow(sigma,2)) * exp(-0.5 * pow(pixel_distances[p]/sigma,2)) * (*(double*)PyArray_GETPTR3(TQUmaps,n,2,ipix_p)) * (*(double*)PyArray_GETPTR3(TQUmaps,nn,2,ipix_p)) );
				}
			}
		}
	}
	// free the arrays
	free(ipix_arr);
	free(pixel_distances);
	// Now the Cov matrices are filled out
}

void invert_a_matrix(gsl_matrix *matrix, gsl_matrix *inv, unsigned int size){
    gsl_permutation *p = gsl_permutation_alloc(size);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);
}

void pixelILC_CalculateILCWeight(PyObject* a, gsl_matrix *CovTi, gsl_matrix *CovQi, gsl_matrix *CovUi, double* weights, unsigned int Nfreqs, unsigned int p){
	double aCia_T=0.0,aCia_Q=0.0,aCia_U=0.0;
	unsigned int i,j;
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			aCia_T += (*(double*)PyArray_GETPTR1(a,i)) * gsl_matrix_get(CovTi,i,j) * (*(double*)PyArray_GETPTR1(a,j)) ;
			aCia_Q += (*(double*)PyArray_GETPTR1(a,i)) * gsl_matrix_get(CovQi,i,j) * (*(double*)PyArray_GETPTR1(a,j)) ;
			aCia_U += (*(double*)PyArray_GETPTR1(a,i)) * gsl_matrix_get(CovUi,i,j) * (*(double*)PyArray_GETPTR1(a,j)) ;
		}
	}
	for(i=0;i<Nfreqs;i++){
		for(j=0;j<Nfreqs;j++){
			// This is the T weight
			weights[p*3*Nfreqs + 0*Nfreqs + i] += (*(double*)PyArray_GETPTR1(a,j)) * gsl_matrix_get(CovTi,j,i) / aCia_T ;
			// This is the Q weight
			weights[p*3*Nfreqs + 1*Nfreqs + i] += (*(double*)PyArray_GETPTR1(a,j)) * gsl_matrix_get(CovQi,j,i) / aCia_Q ;
			// This is the U weight
			weights[p*3*Nfreqs + 2*Nfreqs + i] += (*(double*)PyArray_GETPTR1(a,j)) * gsl_matrix_get(CovUi,j,i) / aCia_U ;
		}
	}
	// after this weights will have the calculated weights.
}

void print_mat_contents(gsl_matrix *matrix, unsigned int size){
    unsigned int i, j;
    double element;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            element = gsl_matrix_get(matrix, i, j);
            printf("%E ", element);
        }
        printf("\n");
    }
}

void empty_mat_contents(gsl_matrix *matrix, unsigned int size){
    unsigned int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {    
            // set entry at i, j to random_value
            gsl_matrix_set(matrix, i, j, 0.0);
        }
    }
}