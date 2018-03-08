#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#define Nr 271
#define Nd 137

void read_matrix_sparse_d(const char *filename, MatrixXd &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	while (fscanf(fp, "%d %d\n", &j, &i) != EOF)
		M((i - 1), (j - 1)) = 1.0;
	fclose(fp);
}

void read_matrix_sparse_i(const char *filename, MatrixXi &M) {
	FILE *fp = fopen(filename, "r");
	int i, j;
	while (fscanf(fp, "%d %d\n", &j, &i) != EOF)
		M((i - 1), (j - 1)) = 1;
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

void read_Names(const char *filename, string *A) {
	ifstream inFile;
	inFile.open(filename, ios::in);
	int idx = 0;
	string tmpStr("");
	while (getline(inFile, tmpStr)) A[idx++] = tmpStr;
	inFile.close();
}

void read_Cared(const char *filename, int *A) {
	ifstream inFile;
	inFile.open(filename, ios::in);
	int idx = 0;
	while (inFile >> A[idx]) A[idx++]--;
	inFile.close();
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

int main(){
  ifstream inFile;
  int index;
  string tmpStr("");
  int i,j;
  ofstream outFile;
  //get names for miRNA and disease of HMDD1, check the cared disease number
  string *mir1_name = new string[495];
  read_Names("datasets/miRNA_Name.txt", mir1_name);
  string *dis1_name = new string[383];
  read_Names("datasets/Disease_Name.txt", dis1_name);
  int *cared = new int[10];
  read_Cared("datasets/cared.txt", cared);
  //get and prepare datasets
  MatrixXd A(Nd,Nr);  read_matrix_sparse_d("datasets/HMDD1.txt",A);
  MatrixXi AI(Nd,Nr);  read_matrix_sparse_i("datasets/HMDD1.txt",AI);
  MatrixXd DIS(Nd, Nd);  read_matrix_dense_d("datasets/DisSim.txt", DIS);
  DIS.diagonal() = VectorXd::Zero(Nd);
  int summa = 0;
  for (i = 0; i < Nd; i++)
	  for (j = 0; j < Nd; j++)
		  if (DIS(i, j) > 0.0000001) summa++;
  DIS = (DIS / DIS.sum()) * summa;
  MatrixXd MIR(Nr, Nr);  read_matrix_dense_d("datasets/miRSim.txt", MIR);
  MIR.diagonal() = VectorXd::Zero(Nr);
  summa = 0;
  for (i = 0; i < Nr; i++)
	  for (j = 0; j < Nr; j++)
		  if (MIR(i, j) > 0.0000001) summa++;
  MIR = (MIR / MIR.sum()) * summa;
  //calculate the scores
  MatrixXd F(Nd,Nr);
  CalcF(A, DIS, MIR, F);
  //output the result
  outFile.open("results/result.txt",ios::out);
  for(i=0;i<Nd;i++)
    for(j=0;j<Nr;j++)
      if(AI(i,j)==0){
        outFile<<dis1_name[i]<<"&"<<mir1_name[j]<<"&"<<F(i,j)<< endl;
      }
  outFile.close();
  string path("results/");
  string suffix(".txt");
  for(index=0;index<10;index++){
    i = cared[index];
    outFile.open((path+dis1_name[i]+suffix).c_str(),ios::out);
    for(j=0;j<Nr;j++){
      if(AI(i,j)==0){
		  outFile << dis1_name[i] << "&" << mir1_name[j] << "&" << F(i, j) << endl;
      }
    }
    outFile.close();
  }
  delete []mir1_name;
  delete []dis1_name;
  delete []cared;
}
