__kernel void invert(__global uchar16* data, const int size)
{
	int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);
	uchar x = 0xFF;
	if (idx < size) {
		data[idx] ^= x;
	}
}