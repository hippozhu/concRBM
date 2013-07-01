all: cuRBM

cuRBM: main.o cublasRBM.o 
	g++ -g main.o cublasRBM.o -o cuRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

main.o: cuRBM.cpp
	g++ -g -c cuRBM.cpp -o main.o

cublasRBM.o: cublasRBM.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 cublasRBM.cu -o cublasRBM.o

clean:
	rm -rf *.o cuRBM
