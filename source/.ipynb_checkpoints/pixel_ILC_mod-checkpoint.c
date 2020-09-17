#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/ndarrayobject.h>
#include <pixel_ILC.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>

static PyObject *doPixelILC(PyObject *self, PyObject *args){
	/* Getting the elements */
	// TQUmap will be a numpy array with the shape [Nfreqs,3,npix]
	// ipix_arr will be the array with all the pixel indices
	//a will be an array with shape [Nfreqs] which contains the CMB SED (in RJ units)
	PyObject *TQUmaps_s1 = NULL;
	PyObject *TQUmaps_s2 = NULL;
	PyObject *nside = NULL;
	PyObject *a = NULL;
	PyObject *fwhm = NULL;
	PyObject *Nfreqs = NULL;
	PyObject *ipix_arr=NULL;
	PyObject *Npixels=NULL;
	PyObject *rank=NULL;
	
	if (!PyArg_ParseTuple(args, "OOOOOOOOO",&TQUmaps_s1,&TQUmaps_s2, &nside, &a, &fwhm, &Nfreqs, &ipix_arr, &Npixels, &rank))
		return NULL;
	
	unsigned int p;
	unsigned long ipix;
	unsigned int nside_map = (int) PyLong_AsLong(nside);
	unsigned int Nfreqs_ = (int) PyLong_AsLong(Nfreqs);
	unsigned long Npixels_ = (long) PyLong_AsLong(Npixels);
	unsigned int rank_ = (int) PyLong_AsLong(rank);
	
	double fwhm_ = PyFloat_AsDouble(fwhm);
	double sigma = fwhm_ / (2.0*sqrt(2*log(2.0)));
	
	double* weights = calloc(Npixels_*3*Nfreqs_,sizeof(double));
	
	gsl_matrix *CovT = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	gsl_matrix *CovQ = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	gsl_matrix *CovU = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	gsl_matrix *CovTi = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	gsl_matrix *CovQi = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	gsl_matrix *CovUi = gsl_matrix_calloc(Nfreqs_, Nfreqs_);
	
	for(p=0;p<Npixels_;p++){
		if(p%5000==0){
			printf("Rank %i is working on pixel number %i of %i \n",rank_,p,Npixels_);
		}
		// This is the index of the pixel to process
		ipix = (*(long*)PyArray_GETPTR1(ipix_arr,p));
		//if(p==0){
			//double value = (*(double*)PyArray_GETPTR3(TQUmaps,0,1,ipix));
		//	printf("The pixel is %i, rank %i\n",ipix,rank_);
		//}
		// Cov is a gsl_matrix and has shape [Nfreqs,Nfreqs] with indices n,nn
		// I need 3 of them, for T,Q,U
		// I need to make sure to empty their content from the previous iteration
		empty_mat_contents(CovT,Nfreqs_);
		empty_mat_contents(CovQ,Nfreqs_);
		empty_mat_contents(CovU,Nfreqs_);
		empty_mat_contents(CovTi,Nfreqs_);
		empty_mat_contents(CovQi,Nfreqs_);
		empty_mat_contents(CovUi,Nfreqs_);
		pixelILC_DefineCovarianceMatrix(ipix, nside_map, Nfreqs_, TQUmaps_s1, TQUmaps_s2, sigma, CovT, CovQ, CovU);
		// Now we need to invert the Cov matrices
		invert_a_matrix(CovT,CovTi,Nfreqs_);
		invert_a_matrix(CovQ,CovQi,Nfreqs_);
		invert_a_matrix(CovU,CovUi,Nfreqs_);
		pixelILC_CalculateILCWeight(a,CovTi, CovQi, CovUi, weights, Nfreqs_,p);
	}

	gsl_matrix_free(CovT);
	gsl_matrix_free(CovQ);
	gsl_matrix_free(CovU);
	gsl_matrix_free(CovTi);
	gsl_matrix_free(CovQi);
	gsl_matrix_free(CovUi);
	
	npy_intp npy_shape[3] = {Npixels_,3,Nfreqs_};
	PyObject *arr 		= PyArray_SimpleNewFromData(3,npy_shape, NPY_DOUBLE, weights);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_OWNDATA);
	return(arr);
}

static PyMethodDef PixelILCMethods[] = {
  {"doPixelILC",  doPixelILC, METH_VARARGS,NULL},
 {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef PixelILC_module = {
    PyModuleDef_HEAD_INIT,
    "PixelILC",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    PixelILCMethods
};

PyMODINIT_FUNC PyInit_PixelILC(void){
  PyObject *m;
  m = PyModule_Create(&PixelILC_module);
  import_array();  // This is important for using the numpy_array api, otherwise segfaults!
  return(m);
}