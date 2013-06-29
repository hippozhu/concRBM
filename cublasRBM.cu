#include "cuRBM.h"

extern float *d_weight, *d_a, *d_b;

__global__ void populateBias(float *c){
  extern __shared__ float v_bias[];
  //int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  if (tid + blockDim.x * blockIdx.y < nHid){
    v_bias[tid] = b[tid + blockDim.x * blockIdx.y];
    for(int i = blockIdx.x * nHid + blockIdx.y * blockDim.x + tid; i < nCase * nHid; i += nHid* gridDim.x)
      c[i] = v_bias[tid];
  }
}

__global__ void addBias(float *c, float *rand){
  extern __shared__ float v_bias[];
  int tid = threadIdx.x;
  if (tid + blockDim.x * blockIdx.y < nHid){
    v_bias[tid] = b[tid + blockDim.x * blockIdx.y];
    for(int i = blockIdx.x * nHid + blockIdx.y * blockDim.x + tid; i < nCase * nHid; i += nHid* gridDim.x)
      /*
      c[i] += v_bias[tid];
      */
      if(rand[i] > 1/(1 + exp(-c[i] - v_bias[tid])))
        c[i] = 0;
      else
        c[i] = 1;
  }
}

float *d_data_v, *d_data_h, *d_rand;

void memoryMove();

void cublasRunRBM(){
  // data
  float *m_data = (float *)malloc(sizeof(float)*ninst*nvisible);
  arrayToMatrix(m_data);
  float *h_data_h = (float *)malloc(sizeof(float)*h_miniBatch*nhidden);

  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));
	
  memoryMove();
  
  cublasHandle_t handle;
  cublasStatus_t ret;
  ret = cublasCreate(&handle);
  CUBLAS_HANDLE_ERROR(ret);
  const float alpha = 1.0f;
  const float beta  = .0f;

  /* set up random generator */
  curandGenerator_t gen;
  CURAND_HANDLE_ERROR(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
  
  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    /* copy mini batch */
    unsigned currentBatch = h_miniBatch > (ninst - i)? (ninst - i): h_miniBatch;
    HANDLE_ERROR(cudaMemcpy(d_data_v, m_data + i * nvisible, currentBatch * nvisible * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(nCase, &currentBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
    
    /* matrix multiplication */
    ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                      nhidden, currentBatch, nvisible, &alpha,
                      d_weight, nvisible, d_data_v, nvisible, &beta, d_data_h, nhidden);
    CUBLAS_HANDLE_ERROR(ret);

    /* set seed for random number generator, generate random numbers (0, 1] */
    //CURAND_HANDLE_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_HANDLE_ERROR(curandSetPseudoRandomGeneratorSeed(gen, (unsigned) time(NULL)));
    CURAND_HANDLE_ERROR(curandGenerateUniform(gen, d_rand, currentBatch * nhidden));

    /* add bias, sigmoid, sampling */
    dim3 g(currentBatch, (nhidden - 1)/256 + 1);
    addBias<<<g ,256 ,256*sizeof(float)>>>(d_data_h, d_rand);
    cudaError_t ret1 = cudaGetLastError();
    HANDLE_ERROR(ret1);
    HANDLE_ERROR(cudaMemcpy(h_data_h, d_data_h, sizeof(float)*nhidden*h_miniBatch, cudaMemcpyDeviceToHost));
    cout << "result:" << h_data_h[0] << " " << h_data_h[1] << " " << h_data_h[nhidden];

    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                      nvisible, currentBatch, nhidden, &alpha,
                      d_weight, nvisible, d_data_h, nhidden, &beta, d_data_v, nvisible);
    CUBLAS_HANDLE_ERROR(ret);

  }
  cublasDestroy(handle);

        HANDLE_ERROR(cudaEventRecord(stop, NULL));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("\tcublas: %.2f msec\n", msecTotal);

  HANDLE_ERROR(cudaFree(d_data_v));
  HANDLE_ERROR(cudaFree(d_data_h));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_a));
  HANDLE_ERROR(cudaFree(d_b));
  free(h_data_h);
  free(m_data);
}

void memoryMove(){
  // basic parametes to constant memory
  HANDLE_ERROR(cudaMemcpyToSymbol(miniBatch, &h_miniBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nVis, &nvisible, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nHid, &nhidden, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  // allocate mini batch on device
  HANDLE_ERROR(cudaMalloc((void **)&d_data_v, h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_data_h, h_miniBatch * nhidden * sizeof(float)));
  
  // weights 
  HANDLE_ERROR(cudaMalloc((void **)&d_weight, nhidden * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_weight, h_weight, nhidden * nvisible * sizeof(float), cudaMemcpyHostToDevice));
  
  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_a, h_a, nvisible * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(a, &d_a, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, nhidden * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(b, &d_b, sizeof(float *), 0, cudaMemcpyHostToDevice));
  
  /* allocate memory for random numbers */
  unsigned rand_size = nvisible < nhidden? nhidden: nvisible;
  HANDLE_ERROR(cudaMalloc((void **)&d_rand, h_miniBatch * rand_size * sizeof(float)));
}
