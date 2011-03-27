#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tableau2(	__global double *Binv,
						__global double *BinvAs,
						int m, int r) 
{
	double erBinvAs = BinvAs[r];

	for (int j = 0; j < m; j++) {
		Binv[j+r*m] = Binv[j+r*m] / erBinvAs;
	}
}