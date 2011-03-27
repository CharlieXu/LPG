#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void minrc(__global double *rc, 
					__global int    *varStatus,
					__global int	*s,
					int n) 
{
	double minRC = -1e-7;
	int ev = -1;

	for (int i = 0; i < n; i++) {
		if ((double)varStatus[i] * rc[i] < minRC) { 
			minRC = (double)varStatus[i] * rc[i];
			ev = i;
		}
	}
	
	*s = ev;
}