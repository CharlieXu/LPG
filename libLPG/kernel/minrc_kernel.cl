#include "C:/Users/Iain/Dev/LPG/libLPG/kernel/common.clh"

__kernel void minrc(__global SCALAR *rc, 
					__global int    *varStatus,
					__global int	*s,
					int n) 
{
	SCALAR minRC = -1e-7;
	int ev = -1;

	for (int i = 0; i < n; i++) {
		if ((SCALAR)varStatus[i] * rc[i] < minRC) { 
			minRC = (SCALAR)varStatus[i] * rc[i];
			ev = i;
		}
	}
	
	*s = ev;
}