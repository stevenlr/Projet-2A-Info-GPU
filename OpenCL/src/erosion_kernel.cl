__kernel void erosion(__global uchar* datai, __global uchar* datao, const int width, const int height, const int rx, const int ry)
{
	int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);
	int idy = get_local_size(1)*get_group_id(1) + get_local_id(1);

	if (idx < width && idy < height) {
		int xmin = idx - rx, xmax = idx + rx;
		int ymin = idy - ry, ymax = idy + ry;

		if (xmin < 0)
			xmin = 0;
		else if (xmax >= width)
			xmax = width - 1;
		if (ymin < 0)
			ymin = 0;
		else if (ymax >= height)
			ymax = height - 1;

		uchar mini = 255;

		for (int y = ymin; y < ymax; ++y) {
			for (int x = xmin; x < xmax; ++x) {
				mini = min(mini, datai[width * y + x]);
			}
		}

		datao[width * idy + idx] = mini;
	}
}