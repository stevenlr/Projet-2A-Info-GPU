__kernel void invert(__global uint8* data,__global const int* size, __global uint8* data2)
{
	const int idx = get_global_id(0);

	data2[idx] = data[idx] ^ 0xFF;
	if (idx < *size) {
		data2[idx] = data[idx] ^ 0xFF;
	}
}