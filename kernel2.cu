#include "cuRBM.h"

__global__ void kernel2(){
  __shared__ int ds[16][16];
  __shared__ float ws[16][16];

  int tid_x = threadIdx.x, tid_y = threadIdx.y;
  float result = .0;
  int nActive, jup;

  for(int i = 0; i < (lenVis - 1)/16 + 1; ++ i){
    __syncthreads();
    if(blockIdx.x * 16 + tid_x < miniBatch && i * 16 + tid_y < lenVis)
      ds[tid_x][tid_y] = getData(data, blockIdx.x * 16 + tid_x, i * 16 + tid_y, pitch_data);
    else
      ds[tid_x][tid_y] = .0;

    if(i==(lenVis - 1)/16)
      jup = (nVis - i * 512 - 1)/16 + 1;
    else
      jup = 32;
    for(int j = 0; j < jup; ++j){
      __syncthreads();
      if(i * 512 + j * 16 + tid_x < nVis && blockIdx.y * 16 + tid_y < nHid)
        ws[tid_x][tid_y] = getData(weight, i * 512 + j * 16 + tid_x, blockIdx.y * 16 + tid_y, pitch_weight);
      else
        ws[tid_x][tid_y] = .0;

      __syncthreads();
      if(blockIdx.x * 16 + tid_x < miniBatch && blockIdx.y * 16 +tid_y < nHid)
	nActive = nVis - i * 512 - j * 16 > 16? 16:nVis - i * 512 - j * 16;
        for(int k = 0; k < nActive; ++ k){
          if (ds[tid_x][j/2] & (1<<((31-16*(j%2)-k))))
	   result += ws[k][tid_y];
        }
    }
  }
  __syncthreads();

  if(blockIdx.x * 16 + tid_x < miniBatch && blockIdx.y * 16 +tid_y < nHid)
    setData(data_hid_float, blockIdx.x * 16 + tid_x, blockIdx.y * 16 + tid_y, nHid * sizeof(float) , result);
}

//extern __shared__ float sm[];
__global__ void kernel2_1(){
  __shared__ int ds[16][16];
  __shared__ float ws[16][16];

  int tid_x = threadIdx.x, tid_y = threadIdx.y;
  float result = .0;
  int nActive, jup;

  for(int i = 0; i < (lenVis - 1)/16 + 1; ++ i){
    __syncthreads();
    if(blockIdx.x * 16 + tid_x < miniBatch && i * 16 + tid_y < lenVis)
      ds[tid_y][tid_x] = getData(data, blockIdx.x * 16 + tid_x, i * 16 + tid_y, pitch_data);
    else
      ds[tid_y][tid_x] = .0;

    if(i==(lenVis - 1)/16)
      jup = (nVis - i * 512 - 1)/16 + 1;
    else
      jup = 32;
    for(int j = 0; j < jup; ++j){
      __syncthreads();
      if(i * 512 + j * 16 + tid_x < nVis && blockIdx.y * 16 + tid_y < nHid)
        ws[tid_y][tid_x] = getData(weight, i * 512 + j * 16 + tid_x, blockIdx.y * 16 + tid_y, pitch_weight);
      else
        ws[tid_y][tid_x] = .0;

      __syncthreads();
      if(blockIdx.x * 16 + tid_x < miniBatch && blockIdx.y * 16 +tid_y < nHid)
	nActive = nVis - i * 512 - j * 16 > 16? 16:nVis - i * 512 - j * 16;
        for(int k = 0; k < nActive; ++ k){
          if (ds[j/2][tid_x] & (1<<((31-16*(j%2)-k))))
	   result += ws[tid_y][k];
        }
    }
  }
  __syncthreads();

  if(blockIdx.x * 16 + tid_x < miniBatch && blockIdx.y * 16 +tid_y < nHid)
    setData(data_hid_float, blockIdx.x * 16 + tid_x, blockIdx.y * 16 + tid_y, nHid * sizeof(float) , result);
}

__global__ void kernel2_2(){
  __shared__ float ds[16][16];
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  ds[tid_x][tid_y] = 0;
  //ds[tid_y][tid_x] = 0;
}
void runRBM(){
        float msecTotal = 0.0f;
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, NULL));
	
  deviceInit();

  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    unsigned currentBatch = h_miniBatch > (ninst - i)? (ninst - i): h_miniBatch;
    batchTransfer(i, currentBatch);
    /*
    dim3 g((ninst - 1)/16 + 1, nhidden);
    dim3 b(256);
    kernel1<<<g, b>>>();
    */
    dim3 g((ninst - 1)/16 + 1, (nhidden -1)/16 + 1);
    dim3 b(16, 16);
    //kernel2<<<g, b>>>();
    kernel2_1<<<g, b>>>();
    //kernel2_2<<<1, b>>>();

    cudaThreadSynchronize();
    cudaError_t ret = cudaGetLastError();
    HANDLE_ERROR(ret);
    float *h_data_hid_float = (float *)malloc(ninst * nhidden * sizeof(float));
    HANDLE_ERROR(cudaMemcpy(h_data_hid_float, d_data_hid_float, h_miniBatch * nhidden * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "result:"  << h_data_hid_float[0] << " " << h_data_hid_float[1] << " " << h_data_hid_float[nhidden];
    //printArray(h_data_hid_float, h_miniBatch, nhidden);
    free(h_data_hid_float);
  }

        HANDLE_ERROR(cudaEventRecord(stop, NULL));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("\tcuRMB: %.2f msec\n", msecTotal);
}

