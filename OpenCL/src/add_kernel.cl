__kernel void add(__global uchar* datai1, __global uchar* datai2, const int size)
{
	const int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);

	if (idx < size) {
		datai1[idx] = add_sat(datai1[idx], datai2[idx]);
	}
}