#include "C:/Users/Iain/LPG/libLPG/kernel/common.clh"

__kernel void tableau1(	__global SCALAR *Binv,
						__global SCALAR *BinvAs,
						int m, int r) 
{
	// Get the index of the current element to be processed
	int i = get_global_id(0);
	
	if (i < m) {
		SCALAR erBinvAs = BinvAs[r];
			
		if (i != r) {
			SCALAR eiBinvAs = BinvAs[i];
			for (int j = 0; j < m; j++) {
				Binv[j+i*m] = Binv[j+i*m] - (eiBinvAs)/(erBinvAs)*Binv[j+r*m];
			}
		}
	}
}