#include "C:/Users/Iain/LPG/libLPG/kernel/common.clh"

__kernel void tableau2(	__global SCALAR *Binv,
						__global SCALAR *BinvAs,
						int m, int r) 
{
	SCALAR erBinvAs = BinvAs[r];

	for (int j = 0; j < m; j++) {
		Binv[j+r*m] = Binv[j+r*m] / erBinvAs;
	}
}