__kernel void threshold(__global uchar16* data, const int size, const float16 value16)
{
	const int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);
	float16 data16;

	if (idx < size) {
		data16 = convert_float16(data[idx]);
		data[idx] = convert_uchar16(isgreaterequal(data16, value16));
	}
}