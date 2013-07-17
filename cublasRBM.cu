#include "cuRBM.h"

__constant__ unsigned nCase;
//__constant__ unsigned nVis, nHid, nCase, miniBatch;
//__constant__ float *a, *b, *ones, *vis_data, *vis_reco, *hid_data, *hid_reco;
__device__ unsigned seed;

__device__  float  my_rand() {

	// constants for random no gen.
	unsigned long a = 16807;  		
	unsigned long m = 2147483647;   	// 2^31 - 1
	unsigned long x = (unsigned long) seed;

	x = (a * x)%m;

	seed = (unsigned int) x;

 	return ((float)x)/m;
}

__global__ void addBiasAndSampling(unsigned nVH, float *c, float *bb){
  extern __shared__ float vh_bias[];
  int tid = threadIdx.x;
  if (tid + blockDim.x * blockIdx.x < nVH){
    vh_bias[tid] = bb[tid + blockDim.x * blockIdx.x];
    for(unsigned i = tid + blockDim.x * blockIdx.x; i < nCase * nVH; i += nVH)
      if(my_rand() > 1/(1 + exp(-c[i] - vh_bias[tid])))
        c[i] = 0;
      else
        c[i] = 1;
  }
}

__global__ void addBias(unsigned nVH, float *c, float *bb){
  extern __shared__ float vh_bias[];
  int tid = threadIdx.x;
  if (tid + blockDim.x * blockIdx.x < nVH){
    vh_bias[tid] = bb[tid + blockDim.x * blockIdx.x];
    for(unsigned i = tid + blockDim.x * blockIdx.x; i < nCase * nVH; i += nVH)
      c[i] += vh_bias[tid];
  }
}

float *d_weight, *d_a, *d_b;
float *d_data_v, *d_data_h, *d_rand;
float *d_vis_data, *d_vis_reco, *d_hid_data, *d_hid_reco, *d_ones;
cublasHandle_t handle;
curandGenerator_t gen;
const float alpha = 1.0f;
const float beta  = .0f;
const float beta_one  = 1.0f;
unsigned currentBatch;
const float learn_rate  = 0.0001;
const float learn_rate_neg  = -0.0001;

void deviceMemoryAlloc();
void deviceMemoryFree();

unsigned copyMiniBatchToDevice(int idx_batch){
    /* copy mini batch */
  unsigned currentBatch = h_miniBatch > (ninst - idx_batch)? (ninst - idx_batch): h_miniBatch;
  CUBLAS_HANDLE_ERROR(cublasSetMatrix(nvisible, currentBatch, sizeof(float),
                      h_data + idx_batch * nvisible, nvisible, d_data_v, nvisible));
  HANDLE_ERROR(cudaMemcpyToSymbol(nCase, &currentBatch, sizeof(unsigned), 0,
               cudaMemcpyHostToDevice));
  return currentBatch;
}

void calcUnits(unsigned nunits, float *dev_data, float *b, int sampled){
  //dim3 g(currentBatch, (nunits- 1)/blockSize + 1);
  if(sampled){
    /* set seed for random number generator, generate random numbers (0, 1] */
    //CURAND_HANDLE_ERROR(curandSetPseudoRandomGeneratorSeed(gen, (unsigned) time(NULL)));
    //CURAND_HANDLE_ERROR(curandGenerateUniform(gen, d_rand, currentBatch * nunits));
    addBiasAndSampling<<<(nunits- 1)/blockSize + 1, blockSize, blockSize*sizeof(float)>>>(nunits, dev_data, b);
  }
  else
    addBias<<<(nunits- 1)/blockSize + 1, blockSize, blockSize*sizeof(float)>>>(nunits, dev_data, b);
  cudaError_t ret = cudaGetLastError();
  HANDLE_ERROR(ret);
}

void calcViHj(float *dev_v, float *dev_h){
    /* calculate (Hi)data/reco and (Vi)data/reco */
    const float avg_alpha = 1.0/currentBatch;
    cublasStatus_t ret;
    ret = cublasSgemv(handle, CUBLAS_OP_N, nvisible, currentBatch, &avg_alpha, d_data_v, nvisible, d_ones, 1, &beta, dev_v, 1);
    CUBLAS_HANDLE_ERROR(ret);

    ret = cublasSgemv(handle, CUBLAS_OP_N, nhidden, currentBatch, &avg_alpha, d_data_h, nhidden, d_ones, 1, &beta, dev_h, 1);
    CUBLAS_HANDLE_ERROR(ret);
}

void cublasRunRBM(){
  // data
  //unsigned bigger = nvisible < nhidden? nhidden: nvisible;
  float *h_data_h = (float *)malloc(sizeof(float) * 2* nvisible * nvisible);

  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  cublasStatus_t ret;
  ret = cublasCreate(&handle);
  CUBLAS_HANDLE_ERROR(ret);

  deviceMemoryAlloc();
  
  /* create random generator */
  CURAND_HANDLE_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  
  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    currentBatch = copyMiniBatchToDevice(i);

    /* matrix multiplication for hidden units calculation */
    ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                      nhidden, currentBatch, nvisible, &alpha,
                      d_weight, nvisible, d_data_v, nvisible, &beta, d_data_h, nhidden);
    CUBLAS_HANDLE_ERROR(ret);
    calcUnits(nhidden, d_data_h, d_b, 1);
    calcViHj(d_vis_data, d_hid_data);
/*
*/
    HANDLE_ERROR(cudaMemcpy(h_data_h, d_data_h, sizeof(float)*currentBatch*nhidden, cudaMemcpyDeviceToHost));
    //printArray(h_data_h, nhidden, currentBatch);

    /* recontruct visible units */
    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                      nvisible, currentBatch, nhidden, &alpha,
                      d_weight, nvisible, d_data_h, nhidden, &beta, d_data_v, nvisible);
    CUBLAS_HANDLE_ERROR(ret);
    calcUnits(nvisible, d_data_v, d_a, 1);

    /* recontruct hidden units */
    ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                      nhidden, currentBatch, nvisible, &alpha,
                      d_weight, nvisible, d_data_v, nvisible, &beta, d_data_h, nhidden);
    CUBLAS_HANDLE_ERROR(ret);
    calcUnits(nhidden, d_data_h, d_b, 0);
    calcViHj(d_vis_reco, d_hid_reco);

    /* update weight */
    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                      nvisible, nhidden, 1, &learn_rate,
                      d_vis_data, nvisible, d_hid_data, nhidden, &beta_one, d_weight, nvisible);
    CUBLAS_HANDLE_ERROR(ret);
    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                      nvisible, nhidden, 1, &learn_rate_neg,
                      d_vis_reco, nvisible, d_hid_reco, nhidden, &beta_one, d_weight, nvisible);
    CUBLAS_HANDLE_ERROR(ret);

    /* update bias */
    ret = cublasSaxpy(handle, nvisible, &learn_rate, d_vis_data, 1, d_a, 1);
    CUBLAS_HANDLE_ERROR(ret);
    ret = cublasSaxpy(handle, nvisible, &learn_rate_neg, d_vis_reco, 1, d_a, 1);
    CUBLAS_HANDLE_ERROR(ret);

    ret = cublasSaxpy(handle, nhidden, &learn_rate, d_hid_data, 1, d_b, 1); 
    CUBLAS_HANDLE_ERROR(ret);
    ret = cublasSaxpy(handle, nhidden, &learn_rate_neg, d_hid_reco, 1, d_b, 1);
    CUBLAS_HANDLE_ERROR(ret);

/*
    HANDLE_ERROR(cudaMemcpy(h_data_h, d_a, sizeof(float)*nvisible, cudaMemcpyDeviceToHost));
    printArray(h_data_h, 1, nvisible);
    HANDLE_ERROR(cudaMemcpy(h_data_h, d_b, sizeof(float)*nhidden, cudaMemcpyDeviceToHost));
    printArray(h_data_h, 1, nhidden);
    //cout << "result:" << h_data_h[0] << " " << h_data_h[1] << " " << h_data_h[nvisible] << endl;
*/
  }
  cublasDestroy(handle);

  HANDLE_ERROR(cudaEventRecord(stop, NULL));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("\tcublas: %.2f msec\n", msecTotal);

  deviceMemoryFree();
  free(h_data_h);
}

void deviceMemoryFree(){
  HANDLE_ERROR(cudaFree(d_data_v));
  HANDLE_ERROR(cudaFree(d_data_h));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_a));
  HANDLE_ERROR(cudaFree(d_b));
  HANDLE_ERROR(cudaFree(d_rand));
  HANDLE_ERROR(cudaFree(d_ones));
  HANDLE_ERROR(cudaFree(d_vis_data));
  HANDLE_ERROR(cudaFree(d_hid_data));
  HANDLE_ERROR(cudaFree(d_vis_reco));
  HANDLE_ERROR(cudaFree(d_hid_reco));
}

void deviceMemoryAlloc(){
  // basic parametes to constant memory
  //HANDLE_ERROR(cudaMemcpyToSymbol(miniBatch, &h_miniBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemcpyToSymbol(nVis, &nvisible, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemcpyToSymbol(nHid, &nhidden, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  // allocate mini batch on device
  HANDLE_ERROR(cudaMalloc((void **)&d_data_v, h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_data_h, h_miniBatch * nhidden * sizeof(float)));
  
  // weights 
  HANDLE_ERROR(cudaMalloc((void **)&d_weight, nhidden * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_weight, h_weight, nhidden * nvisible * sizeof(float), cudaMemcpyHostToDevice));
  
  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_a, h_a, nvisible * sizeof(float), cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemcpyToSymbol(a, &d_a, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, nhidden * sizeof(float), cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemcpyToSymbol(b, &d_b, sizeof(float *), 0, cudaMemcpyHostToDevice));
  
  /* allocate memory for random numbers */
  unsigned bigger = nvisible < nhidden? nhidden: nvisible;
  HANDLE_ERROR(cudaMalloc((void **)&d_rand, h_miniBatch * bigger * sizeof(float)));

  float *h_ones = (float *)malloc(h_miniBatch * sizeof(float));
  fill_n (h_ones, h_miniBatch, 1);
  HANDLE_ERROR(cudaMalloc((void **)&d_ones, h_miniBatch * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_ones, h_ones, h_miniBatch * sizeof(float), cudaMemcpyHostToDevice));
  //HANDLE_ERROR(cudaMemcpyToSymbol(ones, &d_ones, sizeof(float *), 0, cudaMemcpyHostToDevice));
  free(h_ones);

  HANDLE_ERROR(cudaMalloc((void **)&d_vis_data, nvisible * sizeof(float)));
  //HANDLE_ERROR(cudaMemcpyToSymbol(vis_data, &d_vh, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_vis_reco, nvisible * sizeof(float)));
  //HANDLE_ERROR(cudaMemcpyToSymbol(vis_reco, &d_vh, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_data, nhidden * sizeof(float)));
  //HANDLE_ERROR(cudaMemcpyToSymbol(hid_data, &d_vh, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_reco, nhidden * sizeof(float)));
  //HANDLE_ERROR(cudaMemcpyToSymbol(hid_reco, &d_vh, sizeof(float *), 0, cudaMemcpyHostToDevice));
}

