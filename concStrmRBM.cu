#include "cuRBM.h"

__constant__ unsigned nCase;
__constant__ float *data_vis, *data_hid;
__constant__ float *data_v_reco[100];

__device__ unsigned seed;

__device__  float  my_rand() {
	// constants for random no gen.
	unsigned long a = 16807;  		
	unsigned long m = 2147483647;   	// 2^31 - 1
	unsigned long x = (unsigned long) seed;

	x = (a * x)%m;
	seed = (unsigned) x;
 	return ((float)x)/m;
}

__global__ void bias(float *c, float *bb, unsigned offset, unsigned nVH, unsigned sb){
  extern __shared__ float vh_bias[];
  if(blockDim.x * blockIdx.x + threadIdx.x < sb){
    unsigned c_idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    vh_bias[threadIdx.x] = bb[c_idx];
    for(; c_idx < nCase * nVH; c_idx += nVH)
      c[c_idx] += vh_bias[threadIdx.x];
  }
}

__global__ void biasSampling(float *c, float *bb, unsigned offset, unsigned nVH, unsigned sb){
  extern __shared__ float vh_bias[];
  if(blockDim.x * blockIdx.x + threadIdx.x < sb){
    unsigned c_idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    vh_bias[threadIdx.x] = bb[c_idx];
    for(; c_idx < nCase * nVH; c_idx += nVH){
      if(my_rand() > 1/(1 + exp(-c[c_idx] - vh_bias[threadIdx.x])))
        c[c_idx] = 0;
      else
        c[c_idx] = 1;
    }
  }
}

__global__ void sumUpVisReco(int ns, unsigned len, float *c){
  for(unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x){
   float s =.0; 
   for(int j = 0; j < ns; ++ j)
     s += *(data_v_reco[j] + i);
   c[i] = s;
  }
}

float *d_weight, *d_a, *d_b;
float *d_data_v, *d_data_h, *d_data_v_reco, * d_data_h_reco;
float *dev_data_v_reco[100];
float *d_vis_data, *d_vis_reco, *d_hid_data, *d_hid_reco, *d_ones;
const float alpha = 1.0f;
const float beta  = .0f;
const float beta_one  = 1.0f;
unsigned currentBatch;
unsigned *batchStart;
//const float learn_rate  = 0.0001;
const float learn_rate  = 10;
//const float learn_rate_neg  = -0.0001;
const float learn_rate_neg  = -10;

cublasHandle_t handle;
curandGenerator_t gen;
cudaStream_t *strm;
cudaEvent_t *evt;

void deviceMemoryAlloc();
void deviceMemoryFree();

unsigned copyMiniBatchToDevice(int idx_batch){
  /* copy mini batch */
  unsigned nBatch = h_miniBatch > (ninst - idx_batch)? (ninst - idx_batch): h_miniBatch;
  //CUBLAS_HANDLE_ERROR(cublasSetMatrix(nvisible, nBatch, sizeof(float),
  //                    h_data + idx_batch * nvisible, nvisible, d_data_v, nvisible));
  CUBLAS_HANDLE_ERROR(cublasSetMatrix(nvisible, nBatch, sizeof(float),
                      h_data, nvisible, d_data_v, nvisible));
  HANDLE_ERROR(cudaMemcpyToSymbol(nCase, &nBatch, sizeof(unsigned), 0,
               cudaMemcpyHostToDevice));
  return nBatch;
}

void calcVHij(unit_t u, unsigned offset, unsigned len){
    /* calculate (Hi)data/reco and (Vi)data/reco */
    const float avg_alpha = 1.0/currentBatch;
    float *vhij, *dev_data_vh;
    int stride;
    switch (u){
      case VISIBLE:
        vhij = d_vis_data + offset;
        dev_data_vh = d_data_v + offset;
        stride = nvisible;
        break;
      case HIDDEN:
        vhij = d_hid_data + offset;
        dev_data_vh = d_data_h + offset;
        stride = nhidden;
        break;
      case VISIBLE_RECO:
        vhij = d_vis_reco + offset;
        dev_data_vh = d_data_v_reco + offset;
        stride = nvisible;
        break;
      case HIDDEN_RECO:
        vhij = d_hid_reco + offset;
        dev_data_vh = d_data_h + offset;
        stride = nhidden;
        break;
      default:
        break;
    }
    cublasStatus_t ret;
    ret = cublasSgemv(handle, CUBLAS_OP_N, len, currentBatch, &avg_alpha, dev_data_vh, stride, d_ones, 1, &beta, vhij, 1);
    CUBLAS_HANDLE_ERROR(ret);
}

void deviceMemoryAlloc(){
  // allocate for visible & hidden data 
  HANDLE_ERROR(cudaMalloc((void **)&d_data_v, h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_data_h, h_miniBatch * nhidden * sizeof(float)));
  for(int j = 0; j < nStream; ++ j)
    HANDLE_ERROR(cudaMalloc((void **)&dev_data_v_reco[j], h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpyToSymbol(data_v_reco, &dev_data_v_reco, nStream * sizeof(float *), 0, cudaMemcpyHostToDevice));

  // allocate for vis/hid reconstruction
  HANDLE_ERROR(cudaMalloc((void **)&d_data_v_reco, h_miniBatch * nvisible * sizeof(float)));
  
  // weights 
  HANDLE_ERROR(cudaMalloc((void **)&d_weight, nvisible * nStream * streamBatch * sizeof(float)));

  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_a, h_a, nvisible * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, nhidden * sizeof(float), cudaMemcpyHostToDevice));
  
  // allocate & copy ones vector
  float *h_ones = (float *)malloc(h_miniBatch * sizeof(float));
  fill_n (h_ones, h_miniBatch, 1);
  HANDLE_ERROR(cudaMalloc((void **)&d_ones, h_miniBatch * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_ones, h_ones, h_miniBatch * sizeof(float), cudaMemcpyHostToDevice));
  free(h_ones);

  // allocate for Vi Hj
  HANDLE_ERROR(cudaMalloc((void **)&d_vis_data, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_vis_reco, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_data, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_hid_reco, nhidden * sizeof(float)));
}

void deviceMemoryFree(){
  HANDLE_ERROR(cudaFree(d_data_v));
  HANDLE_ERROR(cudaFree(d_data_h));
  for(int j = 0; j < nStream; ++ j)
    HANDLE_ERROR(cudaFree(dev_data_v_reco[j]));
  HANDLE_ERROR(cudaFree(d_data_v_reco));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_a));
  HANDLE_ERROR(cudaFree(d_b));
  HANDLE_ERROR(cudaFree(d_ones));

  HANDLE_ERROR(cudaFree(d_vis_data));
  HANDLE_ERROR(cudaFree(d_hid_data));
  HANDLE_ERROR(cudaFree(d_vis_reco));
  HANDLE_ERROR(cudaFree(d_hid_reco));
}

void updateBias(unit_t u, unsigned offset, unsigned len){
    float *d_bias, *d_data, *d_reco;
    if(u == VISIBLE){
      d_bias = d_a + offset;
      d_data = d_vis_data;
      d_reco = d_vis_reco;
    }
    else{
      d_bias = d_b + offset;
      d_data = d_hid_data;
      d_reco = d_vis_reco;
    }

    cublasStatus_t ret;
    ret = cublasSaxpy(handle, len, &learn_rate, d_data, 1, d_bias, 1);
    CUBLAS_HANDLE_ERROR(ret);
    ret = cublasSaxpy(handle, len, &learn_rate_neg, d_reco, 1, d_bias, 1);
    CUBLAS_HANDLE_ERROR(ret);
}

void updateWeight(int offset, int len, float *dev_w){
    cublasStatus_t ret;
    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                      nvisible, len, 1, &learn_rate,
	              d_vis_data, nvisible, d_hid_data + offset,
		      len, &beta_one, dev_w, nvisible);
    CUBLAS_HANDLE_ERROR(ret);
    ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                      nvisible, len, 1, &learn_rate_neg,
	              d_vis_reco, nvisible, d_hid_reco + offset,
	              len, &beta_one, dev_w, nvisible);
    CUBLAS_HANDLE_ERROR(ret);
}


void phase1_TillVisibleRecon(int idx_strm){
  unsigned currentStreamBatch;
  float *d_weight_strm = d_weight + idx_strm * streamBatch * nvisible;
  cublasStatus_t ret;
  //for(unsigned streamBatch_start = idx_strm * streamBatch; streamBatch_start < nhidden; streamBatch_start += nStream * streamBatch){
    /* calculate starting position and length */
    if(batchStart[idx_strm] + streamBatch > nhidden)
      currentStreamBatch = nhidden - batchStart[idx_strm];
    else
      currentStreamBatch = streamBatch;
      
    /* copy partial weights */
    float *h_weight_strm = h_weight + batchStart[idx_strm] * nvisible;
    CUBLAS_HANDLE_ERROR(cublasSetMatrixAsync(nvisible, currentStreamBatch,
    sizeof(float), h_weight_strm, nvisible, d_weight_strm, nvisible, strm[idx_strm]));

    /* matrix multiplication for hidden units calculation */
    float *d_data_h_strm = d_data_h + batchStart[idx_strm];
    ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
          currentStreamBatch, currentBatch, nvisible, &alpha,
          d_weight_strm, nvisible, d_data_v, nvisible, &beta, d_data_h_strm, nhidden);
    CUBLAS_HANDLE_ERROR(ret);

    /* add bias and sampling */
    biasSampling<<<(currentStreamBatch - 1)/blockSize + 1, blockSize, blockSize*sizeof(float), strm[idx_strm]>>>(d_data_h, d_b, batchStart[idx_strm], nhidden, currentStreamBatch);
    cudaError_t cuda_ret = cudaGetLastError();
    HANDLE_ERROR(cuda_ret);

    /* calculate H_j_data */
    calcVHij(HIDDEN, batchStart[idx_strm], currentStreamBatch);
    
    /* partially reconstruct visible units */
    if(batchStart[idx_strm] < nStream * streamBatch)
      ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            nvisible, currentBatch, currentStreamBatch, &alpha,
            d_weight_strm, nvisible, d_data_h_strm, nhidden, &beta, dev_data_v_reco[idx_strm], nvisible);
    else
      ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            nvisible, currentBatch, currentStreamBatch, &alpha,
            d_weight_strm, nvisible, d_data_h_strm, nhidden, &beta_one, dev_data_v_reco[idx_strm], nvisible);

    CUBLAS_HANDLE_ERROR(ret);
  //}
}

void phase2(int idx_strm){
  unsigned currentStreamBatch;
  float *d_weight_strm = d_weight + idx_strm * nvisible * streamBatch;
  //for(unsigned streamBatch_start = idx_strm * streamBatch; streamBatch_start < nhidden; streamBatch_start += nStream * streamBatch){
    /* calculate starting position and length */
    if(batchStart[idx_strm] + streamBatch > nhidden)
      currentStreamBatch = nhidden - batchStart[idx_strm];
    else
      currentStreamBatch = streamBatch;
      
    /* copy partial weights */
    float *h_weight_strm = h_weight + batchStart[idx_strm] * nvisible;
    CUBLAS_HANDLE_ERROR(cublasSetMatrixAsync(nvisible, currentStreamBatch,
    sizeof(float), h_weight_strm, nvisible, d_weight_strm, nvisible, strm[idx_strm]));

    /* matrix multiplication for hidden units calculation */
    float *d_data_h_strm = d_data_h + batchStart[idx_strm];
    cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
           currentStreamBatch, currentBatch, nvisible, &alpha,
           d_weight_strm, nvisible, d_data_v_reco, nvisible, &beta, d_data_h_strm, nhidden);
    CUBLAS_HANDLE_ERROR(ret);

    /* add bias and sampling */
    biasSampling<<<(currentStreamBatch - 1)/blockSize + 1, blockSize, blockSize*sizeof(float), strm[idx_strm]>>>(d_data_h, d_b, batchStart[idx_strm], nhidden, currentStreamBatch);
    cudaError_t cuda_ret = cudaGetLastError();
    HANDLE_ERROR(cuda_ret);

    /* calculate H_j_reco */
    calcVHij(HIDDEN_RECO, batchStart[idx_strm], currentStreamBatch);

    /* update bias for hidden */
    updateBias(HIDDEN, batchStart[idx_strm], currentStreamBatch);

    /* update weights */
    updateWeight(batchStart[idx_strm], currentStreamBatch, d_weight_strm);

    /* copy the new weights back to host */
    CUBLAS_HANDLE_ERROR(cublasGetMatrixAsync(nvisible, currentStreamBatch,
    sizeof(float), d_weight_strm, nvisible, h_weight_strm, nvisible, strm[idx_strm]));
  //}
}

void cublasRunRBM(){
  // data
  //float *h_data_h = (float *)malloc(sizeof(float) * nhidden* nvisible);

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
  strm = (cudaStream_t *)malloc(nStream * sizeof(cudaStream_t));
  evt = (cudaEvent_t *)malloc(nStream * sizeof(cudaEvent_t));
  batchStart = (unsigned *)malloc(nStream * sizeof(unsigned));
  for(int j = 0; j < nStream; ++ j){
    HANDLE_ERROR(cudaStreamCreate(&strm[j]));
    HANDLE_ERROR(cudaEventCreate(&evt[j]));
  }

  /* main loop over all samples by mini-batch */
  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    /* copy mini-batch in default stream */
    //if((i/h_miniBatch)%100==0)
      //cout << "iter " << i/h_miniBatch << endl;
    CUBLAS_HANDLE_ERROR(cublasSetStream(handle, NULL));
    currentBatch = copyMiniBatchToDevice(i);

    /* sync for mini-batch copy */
    cudaDeviceSynchronize();

    /* calculate V_i_data */
    calcVHij(VISIBLE, 0, nvisible);

    /* concurrent streams */
    for(int j = 0; j < nStream; ++ j)
      batchStart[j] = j * streamBatch;
    for(int k = 0; k < nhidden; k += nStream * streamBatch){
      for(int j = 0; j < nStream; ++ j){
        if(batchStart[j] < nhidden){
          CUBLAS_HANDLE_ERROR(cublasSetStream(handle, strm[j]));
          phase1_TillVisibleRecon(j);
          batchStart[j] += nStream * streamBatch;
	}
      }
    }

    /* sync for visible recon matrix by all streams and sum up 
       return to default NULL stream, implicit sync */
    int streamUsed;
    if(1.0*nhidden/streamBatch > (nStream - 1))
      streamUsed = nStream;
    else
      streamUsed = (nhidden - 1)/streamBatch + 1;
    sumUpVisReco<<<(currentBatch * nvisible)/blockSize + 1, blockSize>>>(streamUsed, currentBatch * nvisible, d_data_v_reco);
    biasSampling<<<(nvisible - 1)/blockSize + 1, blockSize, blockSize*sizeof(float)>>>(d_data_v_reco, d_a, 0, nvisible, nvisible);
    //cudaDeviceSynchronize();
    cudaError_t cuda_ret = cudaGetLastError();
    HANDLE_ERROR(cuda_ret);

    /* calculate V_i_reco */
    CUBLAS_HANDLE_ERROR(cublasSetStream(handle, NULL));
    calcVHij(VISIBLE_RECO, 0, nvisible);

    /* update bias for visible */
    updateBias(VISIBLE, 0, nvisible);
    cudaDeviceSynchronize();

    /* concurrent streams */
    for(int j = 0; j < nStream; ++ j)
      batchStart[j] = j * streamBatch;
    for(int k = 0; k < nhidden; k += nStream * streamBatch){
      for(int j = 0; j < nStream; ++ j){
        if(batchStart[j] < nhidden){
          CUBLAS_HANDLE_ERROR(cublasSetStream(handle, strm[j]));
	  phase2(j);
          batchStart[j] += nStream * streamBatch;
	}
      }
    }
  }
  cudaDeviceSynchronize();
  cublasDestroy(handle);
  deviceMemoryFree();

  HANDLE_ERROR(cudaEventRecord(stop, NULL));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
  run_time = msecTotal;
  //cout << msecTotal << ",";
    //unsigned row = currentBatch;
    //unsigned row = 1;
    //HANDLE_ERROR(cudaMemcpy(h_data_h, d_weight, sizeof(float)*row*col, cudaMemcpyDeviceToHost));
    //printArray(h_data_h, row, col);
    //printArray(h_weight, row, col);
    //printArray(eigen_data_h, row, col);
    //cout << "sqare norm: " << sqn(h_data_h, eigen_data_h, row * col) << endl;
    /*
    unsigned row = nvisible;
    unsigned col = nhidden;
    cout << "sqare norm: " << sqn(h_weight, eigen_data_h, row * col) << endl;
  free(h_data_h);
    */

  //printf("\tcublas: %.2f msec\n", msecTotal);
}

