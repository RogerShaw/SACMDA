#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#define Nr 495
#define Nd 383

void read_matrix_sparse_d(const char *filename, MatrixXd &M){
  FILE *fp = fopen(filename,"r");
  int i,j;
  while(fscanf(fp,"%d %d\n",&j,&i)!=EOF)
    M((i-1),(j-1)) = 1.0;
  fclose(fp);
}

void read_matrix_sparse_i(const char *filename, MatrixXi &M){
  FILE *fp = fopen(filename,"r");
  int i,j;
  while(fscanf(fp,"%d %d\n",&j,&i)!=EOF)
    M((i-1),(j-1)) = 1;
  fclose(fp);
}

void read_matrix_dense_d(const char *filename, MatrixXd &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	double val;
	for (i = 0; i<M.rows(); i++)
		for (j = 0; j<M.cols(); j++) {
			fscanf(fp, "%lf", &val);
			M(i, j) = val;
		}
	fclose(fp);
}

void CalcF(const MatrixXd &A, const MatrixXd &DIS, const MatrixXd &MIR, MatrixXd &F) {
	MatrixXd GRAPH(Nd + Nr, Nd + Nr);
	GRAPH.block(0, 0, Nd, Nd) = DIS;
	GRAPH.block(0, Nd, Nd, Nr) = A;
	GRAPH.block(Nd, Nd, Nr, Nr) = MIR;
	GRAPH.block(Nd, 0, Nr, Nd) = A.transpose();
	MatrixXd WEIGHT(Nd + Nr, Nd + Nr);
	WEIGHT = MatrixXd::Zero(Nd + Nr, Nd + Nr);
	WEIGHT.diagonal() = (GRAPH * GRAPH.transpose()).diagonal();
	for (int i = 0; i < Nd + Nr; i++) {
		if (WEIGHT(i, i) < 0.000001) WEIGHT(i, i) = 10000000;
	}
	WEIGHT = WEIGHT.inverse();
	MatrixXd Route2E(Nd + Nr, Nd + Nr);
	Route2E = GRAPH * WEIGHT * GRAPH;
	MatrixXd Route3E(Nd + Nr, Nd + Nr);
	Route3E = Route2E * WEIGHT * GRAPH;
	F = (Route2E + Route3E).block(0, Nd, Nd, Nr);
}

double find_rank(const ArrayXd &V,double val){
  int lowerbound=0,upperbound=V.size()-1;
  int middle;
  double ret;
  while(1){
    if(upperbound-lowerbound == 1) break;
    middle = (lowerbound+upperbound)/2;
    if(abs(V(middle)-val)<0.00000001){
      if(abs(V(lowerbound)-val)>0.00000001) lowerbound++;
      else if(abs(V(upperbound)-val)>0.00000001) upperbound--;
      else{
        ret = ((double)lowerbound+(double)upperbound)/2;
        break;
      }
    }else if(V(middle)>val) lowerbound = middle;
    else upperbound = middle;
  }
  if(upperbound-lowerbound==1){
    if(V(upperbound)<val) ret = (double)upperbound;
    else ret = (double)lowerbound;
  }
  return ret;
}

void FFCV(MatrixXd &A,const MatrixXi &AI, const MatrixXd &SIM_DIS, const MatrixXd &SIM_MIR){
  int rep,i,j,p,n;
  MatrixXd F(Nd,Nr);
  ArrayXd V(Nd*Nr-5430);
  //prepare index matrix
  MatrixXi idx(5430,3); //disease,mirna,set_id
  ArrayXd score_rank(5430); //score, rank
  FILE *tup = fopen("datasets/HMDD2.txt","r");
  n = 0;
  while(fscanf(tup,"%d %d\n",&j,&i)!=EOF){
    idx.row(n) << i-1,j-1,n%5;
    n++;
  }
  fclose(tup);
  FILE *ff_fp = fopen("results/ffcv.txt","w");
  for(rep=0;rep<100;rep++){
    //split the seeds to 5 random sets
    for(i=0;i<5430;i++){
      j = rand()%5430;
      p = idx(i,2);
      idx(i,2) = idx(j,2);
      idx(j,2) = p;
    }
    for(n=0;n<5;n++){
      //prepare matrix
		for (i = 0; i < 5430; i++)
			if (idx(i, 2) == n) A(idx(i, 0), idx(i, 1)) = 0;
      //calculate the score matrix
		CalcF(A, SIM_DIS, SIM_MIR, F);
      //gather and rank the scores
      p = 0;
      for(i=0;i<AI.rows();i++)
        for(j=0;j<AI.cols();j++)
          if(AI(i,j)==0) V(p++) = F(i,j);
      std::sort(V.data(),V.data()+V.size(),std::greater<double>());
      for(i=0;i<5430;i++) 
        if(idx(i,2) == n) score_rank(i) = find_rank(V,F(idx(i,0),idx(i,1)));
      //recover matrix
      for(i=0;i<5430;i++)
        if(idx(i,2) == n) A(idx(i,0),idx(i,1)) = 1;
    }
    for(i=0;i<5430;i++) fprintf(ff_fp,"%.1f ",score_rank(i)+1);
    fprintf(ff_fp,"\n");
    printf("finished %3d FFCV\n",rep+1);
  }
  fclose(ff_fp);
}



int main(){
	MatrixXd A(Nd, Nr);	read_matrix_sparse_d("datasets/HMDD2.txt", A);
	MatrixXi AI(Nd, Nr);	read_matrix_sparse_i("datasets/HMDD2.txt", AI);
	MatrixXd DIS_SIM0(Nd, Nd);	read_matrix_dense_d("datasets/DisSim0.txt", DIS_SIM0);
	MatrixXd DIS_SIM1(Nd, Nd);	read_matrix_dense_d("datasets/DisSim1.txt", DIS_SIM1);
	MatrixXd DIS_SIM(Nd, Nd);
	DIS_SIM = (DIS_SIM0 + DIS_SIM1) / 2;
	DIS_SIM.diagonal() = VectorXd::Zero(Nd);
	MatrixXd DIS_WGT(Nd, Nd);	read_matrix_dense_d("datasets/DisWgt.txt", DIS_WGT);
	DIS_WGT.diagonal() = VectorXd::Zero(Nd);
	MatrixXd DIS(Nd, Nd);
	DIS = (DIS_SIM / DIS_SIM.sum())*DIS_WGT.sum();
	MatrixXd MIR_SIM(Nr, Nr);	read_matrix_dense_d("datasets/miRSim.txt", MIR_SIM);
	MIR_SIM.diagonal() = VectorXd::Zero(Nr);
	MatrixXd MIR_WGT(Nr, Nr);	read_matrix_dense_d("datasets/miRWgt.txt", MIR_WGT);
	MIR_WGT.diagonal() = VectorXd::Zero(Nr);
	MatrixXd MIR(Nr, Nr);
	MIR = (MIR_SIM / MIR_SIM.sum())*MIR_WGT.sum();
	FFCV(A, AI, DIS, MIR);
	return 0;
}
