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
	int i, j, k;
	ofstream outFile;
	//get names for miRNA and disease of HMDD2, check the cared disease number
	string *mir2_name = new string[495];
	read_Names("datasets/miRNA_Name.txt", mir2_name);
	string *dis2_name = new string[383];
	read_Names("datasets/Disease_Name.txt", dis2_name);
	int *cared = new int[14];
	read_Cared("datasets/cared.txt", cared);
	//get and prepare datasets
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
	//calculate the scores
	MatrixXd F(Nd, Nr);
	string path("results/");
	string suffix(".txt");
	//output the result
	MatrixXi HMDD2(Nd, Nr);
	read_matrix_sparse_i("datasets/HMDD2.txt", HMDD2);
	for (index = 0; index<14; index++) {
		i = cared[index];
		for (k = 0; k<A.rows(); k++) {
			A(i, k) = 0;
			AI(i, k) = 0;
		}
		CalcF(A, DIS, MIR, F);
		outFile.open((path + dis2_name[i] + suffix).c_str(), ios::out);
		for (j = 0; j<Nr; j++) {
			outFile << dis2_name[i] << "&" << mir2_name[j] << "&" << F(i, j) << "&" << endl;
		}
		outFile.close();
		for (k = 0; k<A.rows(); k++) {
			A(i, k) = HMDD2(i, k);
			AI(i, k) = HMDD2(i, k);
		}
	}
	delete[]mir2_name;
	delete[]dis2_name;
	delete[]cared;
}
