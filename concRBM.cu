#include "cuRBM.h"

__constant__ unsigned nCase;

__device__  float  my_rand(unsigned int *seed) {
	// constants for random no gen.
	unsigned long a = 16807;  		
	unsigned long m = 2147483647;   	// 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

 	return ((float)x)/m;
}

__global__ void addBiasAndSampling(unsigned nVH, float *c, float *bb){
  extern __shared__ float vh_bias[];
  int tid = threadIdx.x;
  unsigned seed = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + tid;  
  if (tid + blockDim.x * blockIdx.y < nVH){
    vh_bias[tid] = bb[tid + blockDim.x * blockIdx.y];
    for(int i = blockIdx.x * nVH + blockIdx.y * blockDim.x + tid; i < nCase * nVH; i += nVH* gridDim.x)
      //if(rand[i] > 1/(1 + exp(-c[i] - vh_bias[tid])))
      if(my_rand(&seed) > 1/(1 + exp(-c[i] - vh_bias[tid])))
        c[i] = 0;
      else
        c[i] = 1;
  }
}

__global__ void addBias(unsigned nVH, float *c, float *bb){
  extern __shared__ float vh_bias[];
  int tid = threadIdx.x;
  if (tid + blockDim.x * blockIdx.y < nVH){
    vh_bias[tid] = bb[tid + blockDim.x * blockIdx.y];
    for(int i = blockIdx.x * nVH + blockIdx.y * blockDim.x + tid; i < nCase * nVH; i += nVH* gridDim.x)
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

unsigned copyMiniBatchToDevice(int idx_batch, cudaStream_t *s){
    /* copy mini batch */
  unsigned currentBatch = h_miniBatch > (ninst - idx_batch)? (ninst - idx_batch): h_miniBatch;
  CUBLAS_HANDLE_ERROR(cublasSetStream(handle, *s));
  CUBLAS_HANDLE_ERROR(cublasSetMatrix(nvisible, currentBatch, sizeof(float),
                      h_data + idx_batch * nvisible, nvisible, d_data_v, nvisible));
  return currentBatch;
}

void calcUnits(unsigned nunits, float *dev_data, float *b, int sampled, cudaStream_t *s){
  dim3 g(currentBatch, (nunits- 1)/256 + 1);
  if(sampled){
    addBiasAndSampling<<<g, 256, 256*sizeof(float), *s>>>(nunits, dev_data, b);
  }
  else
    addBias<<<g, 256, 256*sizeof(float), *s>>>(nunits, dev_data, b);
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
  // allocate mini batch on device
  HANDLE_ERROR(cudaMalloc((void **)&d_data_v, h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_data_h, h_miniBatch * nhidden * sizeof(float)));
  
  // weights 
  HANDLE_ERROR(cudaMalloc((void **)&d_weight, nvisible * nStream * streamBatch * sizeof(float)));

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
  free(h_ones);

  HANDLE_ERROR(cudaMalloc((void **)&d_vis_data, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_vis_reco, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_data, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_reco, nhidden * sizeof(float)));
}

void calcHidden(cudaStream_t strm[]){
  for(unsigned k = 0; k < nhidden; k += nStream * streamBatch){

      unsigned currentStreamBatch = streamBatch; 
      unsigned streamBatch_start = k * nStream * streamBatch; 
      for(int j = 0; j < nStream; ++ j){
	if(streamBatch_start > nhidden)
	  break;

        CUBLAS_HANDLE_ERROR(cublasSetStream(handle, strm[j]));

	if(streamBatch_start + streamBatch > nhidden)
	  currentStreamBatch = nhidden - streamBatch_start;

        // copy partial weights 
	float * h_weight_sb = h_weight + streamBatch_start * nvisible;
	float * d_weight_sb = d_weight + j * nvisible;
        CUBLAS_HANDLE_ERROR(cublasSetMatrix(nvisible, currentStreamBatch, sizeof(float),
            h_weight_sb, nvisible, d_weight_sb, nvisible));

        /* matrix multiplication for hidden units calculation */
	float * d_data_h_sb = d_data_h + streamBatch_start;
        cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                      currentStreamBatch, currentBatch, nvisible, &alpha,
                      d_weight_sb, nvisible, d_data_v, nvisible, &beta, d_data_h_sb, nhidden);
        CUBLAS_HANDLE_ERROR(ret);
        calcUnits(nhidden, d_data_h, d_b, 0, &strm[j]);
        calcViHj(d_vis_data, d_hid_data);
	streamBatch_start += streamBatch;
      }

      /* recontruct visible units */
      for(int j = 0; j < nStream; ++ j){
	if(j>0)
	  HANDLE_ERROR(cudaStreamWaitEvent());
        ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                      nvisible, currentBatch, nhidden, &alpha,
                      d_weight, nvisible, d_data_h, nhidden, &beta, d_data_v, nvisible);
        CUBLAS_HANDLE_ERROR(ret);
      }
  }
}


void cublasRunRBM(){
  // data
  //unsigned bigger = nvisible < nhidden? nhidden: nvisible;
  float *h_data_h = (float *)malloc(sizeof(float) * nhidden* nvisible);

  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  cublasStatus_t ret;
  ret = cublasCreate(&handle);
  CUBLAS_HANDLE_ERROR(ret);

  deviceMemoryAlloc();
  
  /* initialize streams and events */
  cudaStream_t strm[nStream];
  cudaEvent_t evt[nStream];
  for(int j = 0; j < nStream; ++ j){
    HANDLE_ERROR(cudaStreamCreate(&strm[j]));
    HANDLE_ERROR(cudaEventCreate(&evt[j]));
  }

  /* main loop over all samples by mini-batch */
  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    currentBatch = copyMiniBatchToDevice(i, &strm[0]);
    HANDLE_ERROR(cudaEventRecord(evt[0], strm[0]));

    /* sync for mini-batch copy */
    for(int j = 1; j < nStream; ++ j)
      HANDLE_ERROR(cudaStreamWaitEvent(strm[j], evt[0], 0));

    /* first calculation for hidden */
    calcHidden(strm);
    cudaDeviceSynchronize();

    calcVisible(strm);
    cudaDeviceSynchronize();
  }
  cublasDestroy(handle);

  HANDLE_ERROR(cudaEventRecord(stop, NULL));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("\tcublas: %.2f msec\n", msecTotal);

  deviceMemoryFree();
  free(h_data_h);
}
