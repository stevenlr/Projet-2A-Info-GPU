__kernel void invert(__global uchar* data, const int size)
{
	int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);
	idx *= 4;
	uchar4 v;
	uchar x = 0xFF;
	if (idx + 3 < size) {
		v =  (uchar4) (data[idx], data[idx+1], data[idx+2], data[idx+3]);
		v ^= x;
		data[idx] = v.x;
		data[idx+1] = v.y;
		data[idx+2] = v.z;
		data[idx+3] = v.w;
	}
}

//cast uchar4* le uchar