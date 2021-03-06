all: concRBM cuRBM streamRBM concStrmRBM

cuRBM: main.o cublasRBM.o 
	g++ -g main.o cublasRBM.o -o cuRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

concRBM: main.o concRBM.o 
	g++ -g main.o concRBM.o -o concRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

streamRBM: main.o streamRBM.o 
	g++ -g main.o streamRBM.o -o streamRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

concStrmRBM: main.o concStrmRBM.o 
	g++ -g main.o concStrmRBM.o -o concStrmRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas -lcurand

main.o: cuRBM.cpp
	g++ -g -c cuRBM.cpp -o main.o

cublasRBM.o: cublasRBM.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 cublasRBM.cu -o cublasRBM.o

concRBM.o: concRBM.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 concRBM.cu -o concRBM.o

streamRBM.o: streamRBM.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 streamRBM.cu -o streamRBM.o

concStrmRBM.o: concStrmRBM.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 concStrmRBM.cu -o concStrmRBM.o

clean:
	rm -rf *.o concRBM cuRBM streamRBM concStrmRBM 
