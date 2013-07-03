#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

using namespace std;

extern unsigned nvisible, nhidden, ninst, h_miniBatch, nStream, streamBatch;
extern float *h_data, *h_weight, *h_a, *h_b;

extern void cublasRunRBM();
extern void arrayToMatrix(float *);
extern void printArray(float *array, unsigned height, unsigned width);

#define CUBLAS_HANDLE_ERROR( err ) (CublasHandleError( err, __FILE__, __LINE__ ))
static void CublasHandleError( cublasStatus_t err,
                         const char *file,
                         int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate returned error code %d, line(%d)\n", err, __LINE__);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define CURAND_HANDLE_ERROR( err ) (CurandHandleError( err, __FILE__, __LINE__ ))
static void CurandHandleError(curandStatus_t err, const char *file, int line){
  if(err != CURAND_STATUS_SUCCESS) { 
        printf( "error in %s at line %d\n",
                file, line );
        exit( EXIT_FAILURE );
  }
}

