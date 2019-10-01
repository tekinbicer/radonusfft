#include "radonusfft.cuh"
#include "kernels.cuh"
#include <stdio.h>

radonusfft::radonusfft(size_t theta_, float center_, size_t Ntheta_, size_t Nz_, size_t N_)
{
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	center = center_;

	// USFFT parameters
	float eps = 1e-3;
	mu = -log(eps) / (2 * N * N);
	M = ceil(2 * N * 1 / PI * sqrt(-mu * log(eps) + (mu * N) * (mu * N) / 4));

	// arrays allocation on GPU
	cudaMalloc((void **)&f, N * N * Nz * sizeof(float2));
	cudaMalloc((void **)&g, N * Ntheta * Nz * sizeof(float2));
	cudaMalloc((void **)&fde, 2 * N * 2 * N * Nz * sizeof(float2));
	cudaMalloc((void **)&fdee, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2));
	cudaMalloc((void **)&x, N * Ntheta * sizeof(float));
	cudaMalloc((void **)&y, N * Ntheta * sizeof(float));
	cudaMalloc((void **)&theta, Ntheta * sizeof(float));
	cudaMalloc((void **)&shiftfwd, N * sizeof(float2));
	cudaMalloc((void **)&shiftadj, N * sizeof(float2));

	// init 2d FFTs
	int ffts[2];
	int idist;
	int odist;
	int inembed[2];
	int onembed[2];
	//fft 2d
	ffts[0] = 2 * N;
	ffts[1] = 2 * N;
	idist = 2 * N * 2 * N;
	odist = (2 * N + 2 * M) * (2 * N + 2 * M);
	inembed[0] = 2 * N;
	inembed[1] = 2 * N;
	onembed[0] = 2 * N + 2 * M;
	onembed[1] = 2 * N + 2 * M;
	cufftPlanMany(&plan2dfwd, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Nz);
	cufftPlanMany(&plan2dadj, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2C, Nz);

	// init 1d FFTs
	ffts[0] = N;
	idist = N;
	odist = N;
	inembed[0] = N;
	onembed[0] = N;
	cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta * Nz);

	//init thread blocks and block grids
	BS3d.x = 32;
	BS3d.y = 32;
	GS2d0.x = ceil(N / (float)BS3d.x);
	GS2d0.y = ceil(Ntheta / (float)BS3d.y);

	GS3d0.x = ceil(N / (float)BS3d.x);
	GS3d0.y = ceil(N / (float)BS3d.y);
	GS3d0.z = ceil(Nz / (float)BS3d.z);

	GS3d1.x = ceil(2 * N / (float)BS3d.x);
	GS3d1.y = ceil(2 * N / (float)BS3d.y);
	GS3d1.z = ceil(Nz / (float)BS3d.z);

	GS3d2.x = ceil((2 * N + 2 * M) / (float)BS3d.x);
	GS3d2.y = ceil((2 * N + 2 * M) / (float)BS3d.y);
	GS3d2.z = ceil(Nz / (float)BS3d.z);
	
	GS3d3.x = ceil(N / (float)BS3d.x);
	GS3d3.y = ceil(Ntheta / (float)BS3d.y);
	GS3d3.z = ceil(Nz / (float)BS3d.z);

	// copy angles to gpu
	cudaMemcpy(theta, (float *)theta_, Ntheta * sizeof(float), cudaMemcpyDefault);
	// compute polar coordinates
	takexy<<<GS2d0, BS3d>>>(x, y, theta, N, Ntheta);
	// compute shifts with respect to the rotation center
	takeshift<<<ceil(N/1024.0), 1024>>>(shiftfwd, -(center - N / 2.0), N);
	takeshift<<<ceil(N/1024.0), 1024>>>(shiftadj, (center - N / 2.0), N);
}

radonusfft::~radonusfft()
{
	cudaFree(f);
	cudaFree(g);
	cudaFree(fde);
	cudaFree(fdee);
	cudaFree(x);
	cudaFree(y);
	cudaFree(shiftfwd);
	cudaFree(shiftadj);
	cufftDestroy(plan2dfwd);
	cufftDestroy(plan2dadj);
	cufftDestroy(plan1d);
}

void radonusfft::fwd(size_t g_, size_t f_)
{
	// copy data, init arrays with 0
	cudaMemcpy(f, (float2 *)f_, N * N * Nz * sizeof(float2), cudaMemcpyDefault);
	cudaMemset(fde, 0, 2 * N * 2 * N * Nz * sizeof(float2));
	cudaMemset(fdee, 0, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2));
	cudaMemset(g, N * Ntheta * Nz * sizeof(float2), cudaMemcpyDefault);

	// divide by the USFFT kernel function with padding
	divphi<<<GS3d0, BS3d>>>(fde, f, mu, N, Nz);
	// 2d FFT
	fftshiftc<<<GS3d1, BS3d>>>(fde, 2 * N, Nz);
	cufftExecC2C(plan2dfwd, (cufftComplex *)fde, (cufftComplex *)&fdee[M + M * (2 * N + 2 * M)], CUFFT_FORWARD);
	fftshiftc<<<GS3d2, BS3d>>>(fdee, 2 * N + 2 * M, Nz);
	// wrap frequencies
	wrap<<<GS3d2, BS3d>>>(fdee, N, Nz, M);
	// gathering to the polar grid
	gather<<<GS3d3, BS3d>>>(g, fdee, x, y, M, mu, N, Ntheta, Nz);
	// shift with respect to given center
	shift<<<GS3d3, BS3d>>>(g,shiftfwd, N, Ntheta, Nz);
	// 1d IFFT
	fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
	cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
	fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);

	// copy result to cpu
	cudaMemcpy((float2 *)g_, g, N * Ntheta * Nz * sizeof(float2), cudaMemcpyDefault);
}

void radonusfft::adj(size_t f_, size_t g_)
{
	// copy data, init arrays with 0
	cudaMemcpy(g, (float2 *)g_, N * Ntheta * Nz * sizeof(float2), cudaMemcpyDefault);
	cudaMemset(fde, 0, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2));
	cudaMemset(fdee, 0, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2));
	cudaMemset(f, N * N * Nz * sizeof(float2), cudaMemcpyDefault);
	// 1d FFT
	fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
	cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
	fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
	// shift with respect to given center
	shift<<<GS3d3, BS3d>>>(g, shiftadj, N, Ntheta, Nz);

	// filtering 
	applyfilter<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
	// scattering from the polar grid
	scatter<<<GS3d3, BS3d>>>(fdee, g, x, y, M, mu, N, Ntheta, Nz);
	// wrap frequencies
	wrapadj<<<GS3d2, BS3d>>>(fdee, N, Nz, M);
	// 2d IFFT
	fftshiftc<<<GS3d2, BS3d>>>(fdee, 2 * N + 2 * M, Nz);
	cufftExecC2C(plan2dadj, (cufftComplex *)&fdee[M + M * (2 * N + 2 * M)], (cufftComplex *)fde, CUFFT_INVERSE);
	fftshiftc<<<GS3d1, BS3d>>>(fde, 2 * N, Nz);
	// divide by the USFFT kernel function with unpadding
	unpaddivphi<<<GS3d0, BS3d>>>(f, fde, mu, N, Nz);

	// copy result to cpu
	cudaMemcpy((float2 *)f_, f, N * N * Nz * sizeof(float2), cudaMemcpyDefault);
}
