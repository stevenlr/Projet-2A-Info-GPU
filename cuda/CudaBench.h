#ifndef CUDABENCH_H
#define CUDABENCH_H

struct _CudaBench {
	cudaEvent_t start;
	cudaEvent_t end;
};

typedef struct _CudaBench CudaBench;

CudaBench CudaBench_new()
{
	CudaBench b;

	cudaEventCreate(&b.start);
	cudaEventCreate(&b.end);

	return b;
}

void CubaBench_delete(CudaBench b)
{
	cudaEventDestroy(b.start);
	cudaEventDestroy(b.end);
}

void CudaBench_start(CudaBench b)
{
	cudaEventRecord(b.start);
}

void CudaBench_end(CudaBench b)
{
	cudaEventRecord(b.end);
}

float CudaBench_elapsedTime(CudaBench b)
{
	float dt;

	cudaEventElapsedTime(&dt, b.start, b.end);

	return dt;
}

#endif
