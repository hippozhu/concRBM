all: cuRBM

cuRBM: main.o cublasRBM.o cudaMem.o kernels.o kernel1.o kernel2.o 
	g++ -g main.o cublasRBM.o cudaMem.o kernels.o kernel1.o kernel2.o -o cuRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

main.o: cuRBM.cpp
	g++ -g -c cuRBM.cpp -o main.o

cublasRBM.o: cublasRBM.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cublasRBM.cu -o cublasRBM.o

cudaMem.o: cudaMem.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cudaMem.cu -o cudaMem.o

kernels.o: cublasRBM.o cudaMem.o kernel1.o kernel2.o
	nvcc -g -G -dlink -gencode arch=compute_20,code=sm_20 kernel1.o kernel2.o cublasRBM.o cudaMem.o -o kernels.o

kernel1.o: cuRBM.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cuRBM.cu -o kernel1.o

kernel2.o: kernel2.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 kernel2.cu -o kernel2.o

clean:
	rm -rf *.o cuRBM
