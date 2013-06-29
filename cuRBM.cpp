#include <cstring>
#include <cmath>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Dense>
#include "cuRBM.h"

using namespace boost;
using namespace Eigen;

size_t h_pitch_data, h_pitch_data_hid, width, width_hid;
unsigned len, len_hid, nvisible, nhidden, ninst, h_miniBatch;
int *h_data, *h_data_hid;
float *h_weight, *h_weight_hid_maj, *h_a, *h_b;
int nbits = sizeof(int) * 8;

void initData(){
  // Initialize data with random values on host
  len = (nvisible - 1)/nbits + 1;
  h_pitch_data = len * sizeof(int);
  width = len * sizeof(int);

  h_data = (int *)malloc(len * ninst * sizeof(int));
  unsigned i = 0;
  while(i < len * ninst)
    h_data[i++] = rand();

  len_hid = (nhidden - 1)/nbits + 1;
  h_pitch_data_hid = len_hid * sizeof(int);
  width_hid = len_hid * sizeof(int);

  h_data_hid = (int *)malloc(len_hid * ninst * sizeof(int));
}

void syncWeightHidMaj(){
  for(unsigned i = 0; i < nvisible; ++ i)
    for(unsigned j = 0; j < nhidden; ++ j)
      h_weight_hid_maj[i * nhidden + j] = h_weight[j * nvisible + i];
}

void syncWeightHidMaj1(){
  for(unsigned i = 0; i < nhidden; ++ i)
    for(unsigned j = 0; j < nvisible; ++ j)
      h_weight_hid_maj[j * nhidden + i] = h_weight[i * nvisible + j];
}

void initWeight(){
  // Initialize weights by random numbers of a normal distribution (0, 0.01)
  mt19937 rng;
  normal_distribution<float> nd(0.0, .01);
  variate_generator<mt19937&, normal_distribution<float> > var_nor(rng, nd);

  //h_weight = (float *)malloc(nvisible * nhidden * sizeof(float));
  cudaMallocHost((void**)&h_weight, nvisible * nhidden * sizeof(float));
  unsigned i = 0;
  while(i < nvisible * nhidden)
    h_weight[i++] = var_nor();
    //h_weight[i++] = 1;
  h_weight_hid_maj = (float *)malloc(nvisible * nhidden * sizeof(float));
  syncWeightHidMaj();
}

void initVisBias(){
  // Initialize bias for visible units
  h_a = (float *)malloc(nvisible * sizeof(float));
  unsigned *on_count = (unsigned *)malloc(nvisible * sizeof(unsigned));
  memset(on_count, 0, nvisible * sizeof(unsigned));
  for(int i = 0; i < ninst; ++ i){
    for(int j = 0; j < nvisible; ++j){
      if(h_data[j/nbits] & (1<<(nbits-1-j%nbits)))
        ++ on_count[j];
    }
  }
  for(int i = 0; i < nvisible; ++ i){
    double p = 1.0 * on_count[i] / ninst;
    h_a[i] = log(p) - log(1-p);
  }
  free(on_count);
}

void initHidBias(){
  // Initialize bias for hidden units
  h_b = (float *)malloc(nhidden * sizeof(float));
  for(int i = 0; i < nhidden; ++ i)
    h_b[i] = -4;
}

void arrayToMatrix(MatrixXf &m_data){
  for(unsigned i = 0; i < ninst; ++i)
    for(unsigned j = 0; j < nvisible; ++j){
	  int compressed = *(h_data + i*len + j/nbits);
	  unsigned mask = 1 << (nbits - 1 - j%nbits);
	  if(compressed & mask)
	    m_data(i, j) = 1;
	  else
	    m_data(i, j) = 0;
	}
}

void arrayToMatrix(float *m_data){
  for(unsigned i = 0; i < ninst; ++i)
    for(unsigned j = 0; j < nvisible; ++j){
	  int compressed = *(h_data + i*len + j/nbits);
	  unsigned mask = 1 << (nbits - 1 - j%nbits);
	  if(compressed & mask)
	    *(m_data + i * nvisible + j) = 1;
	  else
	    *(m_data + i * nvisible + j) = 0;
	}
}

void printArray(float *array, unsigned height, unsigned width){
  cout << endl;
  for(unsigned i = 0; i < height; ++ i){
    for(unsigned j = 0; j < width; ++ j)
      cout << *(array + i * width + j) << " ";
    cout << endl;
  }
  cout << endl;
}

void rbm(){
  MatrixXf m_data(ninst, nvisible);
  arrayToMatrix(m_data);
  Map<MatrixXf> m_weight(h_weight, nvisible, nhidden);
  Map<VectorXf> m_a(h_a, nvisible);
  Map<VectorXf> m_b(h_b, nhidden);
  /*
  cout << "data * weight" << endl;
  cout << m_data << endl;
  cout << m_weight.transpose() << endl;
  */
  cout << m_data.rows() << "*" << m_data.cols() << endl;
  cout << m_weight.rows() << "*" << m_weight.cols() << endl;
  clock_t tStart = clock();
  MatrixXf result = m_data*m_weight;
  result.rowwise() += m_b.transpose();
  cout << "result:" << result(0,0) << " " << result (0,1) << " " << result(1, 0);
  printf("\tEigen: %.2f msec\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000));
}

int main(int argc, char **argv){
  h_miniBatch = atoi(argv[1]);
  ninst = atoi(argv[1]);
  nvisible = atoi(argv[2]);
  nhidden = atoi(argv[3]);

  clock_t tStart = clock();
  cout << "Generating data ...";
  initData();
  initWeight();
  initVisBias();
  initHidBias();
  printf("\t (%.2f)s\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  
  //rbm();
  //runRBM();
  cublasRunRBM();
}

