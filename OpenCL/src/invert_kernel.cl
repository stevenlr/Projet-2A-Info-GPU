__kernel void invert(__global uchar* data, const int size)
{
	const int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);

	if (idx < size) {
		data[idx] = data[idx] ^ 0xFF;
	}
}