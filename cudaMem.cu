#include "cuRBM.h"

int *d_data, d_data_hid;
float *d_data_hid_float;
float *d_weight, *d_a, *d_b;
size_t d_pitch_weight, d_pitch_data,  d_pitch_data_hid;

void batchTransfer(unsigned start, unsigned batch_size){
  // Copy data to device coalesced
  int *batch_data = h_data + len * start;
  HANDLE_ERROR(cudaMemcpy2D(d_data, d_pitch_data, batch_data, h_pitch_data, width, batch_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nCase, &batch_size, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
}

void deviceInit(){
  // basic parametes to constant memory
  HANDLE_ERROR(cudaMemcpyToSymbol(miniBatch, &h_miniBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nVis, &nvisible, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nHid, &nhidden, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  // allocate global memory for data of mini batch 
  HANDLE_ERROR(cudaMallocPitch((void **)&d_data, &d_pitch_data, len * sizeof(int), h_miniBatch));
  HANDLE_ERROR(cudaMemcpyToSymbol(data, &d_data, sizeof(int *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(pitch_data, &d_pitch_data, sizeof(size_t), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(lenVis, &len, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMallocPitch((void **)&d_data_hid, &d_pitch_data_hid, len_hid * sizeof(int), h_miniBatch));
  HANDLE_ERROR(cudaMemcpyToSymbol(data_hid, &d_data_hid, sizeof(int *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(pitch_data_hid, &d_pitch_data_hid, sizeof(size_t), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(lenHid, &len_hid, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **)&d_data_hid_float, ninst * nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpyToSymbol(data_hid_float, &d_data_hid_float, sizeof(int *), 0, cudaMemcpyHostToDevice));

  // weights to global memory
  HANDLE_ERROR(cudaMallocPitch((void **)&d_weight, &d_pitch_weight, nhidden * sizeof(float), nvisible));
  HANDLE_ERROR(cudaMemcpy2D(d_weight, d_pitch_weight, h_weight, nhidden * sizeof(float),  nhidden * sizeof(float), nvisible, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(weight, &d_weight, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(pitch_weight, &d_pitch_weight, sizeof(size_t), 0, cudaMemcpyHostToDevice));
  
  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_a, h_a, nvisible * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(a, &d_a, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, nhidden * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(b, &d_b, sizeof(float *), 0, cudaMemcpyHostToDevice));
}

