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

void CalcF(const MatrixXd &A, const MatrixXd &DIS, const MatrixXd &MIR, MatrixXd &F) {
	MatrixXd GRAPH(Nd + Nr, Nd + Nr);
	GRAPH.block(0, 0, Nd, Nd) = DIS;
	GRAPH.block(0, Nd, Nd, Nr) = A;
	GRAPH.block(Nd, Nd, Nr, Nr) = MIR;
	GRAPH.block(Nd, 0, Nr, Nd) = A.transpose();
	MatrixXd WEIGHT(Nd + Nr, Nd + Nr);
	WEIGHT = MatrixXd::Zero(Nd + Nr, Nd + Nr);
	WEIGHT.diagonal() = (GRAPH * GRAPH.transpose()).diagonal();
	WEIGHT = WEIGHT.inverse();
	MatrixXd Route2E(Nd + Nr, Nd + Nr);
	Route2E = GRAPH * WEIGHT * GRAPH;
	MatrixXd Route3E(Nd + Nr, Nd + Nr);
	Route3E = Route2E * WEIGHT * GRAPH;
	F = (Route2E + Route3E).block(0, Nd, Nd, Nr);
}

void LOOCV(MatrixXd &A, const MatrixXi &AI, const MatrixXd &DIS, const MatrixXd &MIR) {
	int i, j, p, q;
	double val;
	MatrixXd F(Nd, Nr);
	MatrixXf GRes(Nd, Nr);
	MatrixXf LRes(Nd, Nr);
	float cnt_GG = 0, cnt_GE = 0, cnt_GL = 0;
	float cnt_LG = 0, cnt_LE = 0, cnt_LL = 0;
	for (i = 0; i<AI.rows(); i++)
		for (j = 0; j<AI.cols(); j++)
			if (AI(i, j) == 1) {
				//init counters
				cnt_GG = 0;  cnt_GE = 0;  cnt_GL = 0;
				cnt_LG = 0;  cnt_LE = 0;  cnt_LL = 0;
				//mask one positive example
				val = A(i, j);
				A(i, j) = 0;
				//calculate the score matrix
				CalcF(A, DIS, MIR, F);
				//statistics
				for (p = 0; p<F.rows(); p++) {
					if (p == i) {
						for (q = 0; q<F.cols(); q++)
							if (AI(p, q) == 0) {
								if (abs(F(i, j) - F(p, q))<0.000000000000001) {
									cnt_GE++;
									cnt_LE++;
								}
								else if (F(i, j)>F(p, q)) {
									cnt_GG++;
									cnt_LG++;
								}
								else {
									cnt_GL++;
									cnt_LL++;
								}
							}
					}
					else {
						for (q = 0; q<F.cols(); q++)
							if (AI(p, q) == 0) {
								if (abs(F(i, j) - F(p, q))<0.000000000000001) cnt_GE++;
								else if (F(i, j)>F(p, q)) cnt_GG++;
								else cnt_GL++;
							}
					}
				}
				//result rank
				GRes(i, j) = cnt_GL + cnt_GE / 2;
				LRes(i, j) = cnt_LL + cnt_LE / 2;
				printf("D(%3d)xR(%3d): GRank %8.1f   LRank%8.1f\n", i, j, GRes(i, j), LRes(i, j));
				//recover the positive example
				A(i, j) = val;
			}
	FILE *tup = fopen("datasets/HMDD2.txt", "r");
	FILE *gfp = fopen("results/gloocv.txt", "w");
	FILE *lfp = fopen("results/lloocv.txt", "w");
	while (fscanf(tup, "%d %d\n", &j, &i) != EOF) {
		fprintf(gfp, "%.1f ", GRes(i - 1, j - 1) + 1);
		fprintf(lfp, "%.1f ", LRes(i - 1, j - 1) + 1);
	}
	fclose(tup);
	fclose(gfp);
	fclose(lfp);
}

int main() {
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
	DIS = (DIS_SIM/ DIS_SIM.sum())*DIS_WGT.sum();
	MatrixXd MIR_SIM(Nr, Nr);	read_matrix_dense_d("datasets/miRSim.txt", MIR_SIM);
	MIR_SIM.diagonal() = VectorXd::Zero(Nr);
	MatrixXd MIR_WGT(Nr, Nr);	read_matrix_dense_d("datasets/miRWgt.txt", MIR_WGT);
	MIR_WGT.diagonal() = VectorXd::Zero(Nr);
	MatrixXd MIR(Nr, Nr);
	MIR = (MIR_SIM / MIR_SIM.sum())*MIR_WGT.sum();	
	LOOCV(A, AI, DIS, MIR);
	return 0;
}
