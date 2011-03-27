#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void rc2(__global double *rc, 
				  __global double *c, 
				  __global double *piT, 
				  __global double *A, 
				  int m,
				  int n) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
	if (i < n) {
		// rc = c - piT A
		double piTA = 0.0;
		for (int k = 0; k < m; k++) {
			piTA += piT[k] * A[i + k*n];
		}
		
		rc[i] = c[i] - piTA;
	}
}