#include <cstring>
#include <cmath>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Dense>
#include "cuRBM.h"

using namespace boost;
using namespace Eigen;

unsigned nvisible, nhidden, ninst, h_miniBatch;
float *h_data, *h_weight, *h_a, *h_b;

void initData(){
  h_data = (float *)malloc(ninst * nvisible * sizeof(float));
  unsigned i = 0;
  while(i < ninst * nvisible)
    h_data[i++] = rand()%2;
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
}

void initVisBias(){
  // Initialize bias for visible units
  h_a = (float *)malloc(nvisible * sizeof(float));
  unsigned *on_count = (unsigned *)malloc(nvisible * sizeof(unsigned));
  memset(on_count, 0, nvisible * sizeof(unsigned));
  for(int i = 0; i < ninst; ++ i)
    for(int j = 0; j < nvisible; ++j)
      if(h_data[i * nvisible + j] == 1.0)
        ++ on_count[j];
    
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

void printArray(float *array, unsigned height, unsigned width){
  for(unsigned i = 0; i < height; ++ i){
    for(unsigned j = 0; j < width; ++ j)
      cout << *(array + i * width + j) << " ";
    cout << endl;
  }
  cout << endl;
}

void rbm(){
  Map<MatrixXf> m_data_v(h_data, nvisible, ninst);
  Map<MatrixXf> m_weight(h_weight, nvisible, nhidden);
  Map<VectorXf> m_a(h_a, nvisible);
  Map<VectorXf> m_b(h_b, nhidden);


  clock_t tStart = clock();
  VectorXf vis_data = m_data_v.rowwise().sum()/h_miniBatch;
  MatrixXf m_data_h = m_weight.transpose() * m_data_v;
  m_data_h.colwise() += m_b;
  //cout << m_data_h << endl;
  VectorXf hid_data = m_data_h.rowwise().sum()/h_miniBatch;
  MatrixXf m_data_v_reco = m_weight * m_data_h;
  m_data_v_reco.colwise() += m_a;
  MatrixXf m_data_h_reco = m_weight.transpose() * m_data_v_reco;
  m_data_h_reco.colwise() += m_b;
  VectorXf vis_reco = m_data_v_reco.rowwise().sum()/h_miniBatch;
  VectorXf hid_reco = m_data_h_reco.rowwise().sum()/h_miniBatch;
  //cout << "result:" << m_data_h_reco(0,0) << " " << m_data_h_reco(1,0) << " " << m_data_h_reco(0,1);
  MatrixXf m_weight_new = m_weight + 0.0001 * (vis_data * hid_data.transpose() - 
                        vis_reco * hid_reco.transpose());
  VectorXf m_a_new = m_a + 0.0001 * (vis_data - vis_reco);
  VectorXf m_b_new = m_b + 0.0001 * (hid_data - hid_reco);
  printf("\tEigen: %.2f msec\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000));
}

int main(int argc, char **argv){
  ninst = atoi(argv[1]);
  h_miniBatch = atoi(argv[2]);
  nvisible = atoi(argv[3]);
  nhidden = atoi(argv[4]);

  clock_t tStart = clock();
  cout << "Generating data ...";
  initData();
  initWeight();
  initVisBias();
  initHidBias();
  printf("\t (%.2f)s\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  
  rbm();
  cublasRunRBM();
}

