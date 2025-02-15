#include "C:/Users/Iain/Dev/LPG/libLPG/kernel/common.clh"

__kernel void tableau2(	__global double *Binv,
						__global double *BinvAs,
						int m, int r) 
{
	double erBinvAs = BinvAs[r];

	for (int j = 0; j < m; j++) {
		Binv[j+r*m] = Binv[j+r*m] / erBinvAs;
	}
}