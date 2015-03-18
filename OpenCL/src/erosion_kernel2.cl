#define TILE_SIZE 128

__kernel void erosion(__global uchar* datai, __global uchar* datao, const int width, const int height, const int rx, const int ry)
{
	__local uchar localData[TILE_SIZE * TILE_SIZE];

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int idx = get_local_size(0)*get_group_id(0);
	int idy = get_local_size(1)*get_group_id(1);

	if (idx+tx < width && idy+ty < height) {
		localData[TILE_SIZE * (ry + ty) + rx + tx] = datai[width * (idy + ty) + idx + tx];

		int xmin = tx, xmax = tx;
		int ymin = ty, ymax = ty;

		if (tx == 0)
			xmin = max(0, idx - rx) - idx;
		else if (tx == get_local_size(0) - 1)
			xmax = min(width - idx - 1, convert_int(get_local_size(0)) + rx - 1);
		if (ty == 0)
			ymin = max(0, idy - ry) - idy;
		else if (ty == get_local_size(1) - 1)
			ymax = min(height - idy - 1, convert_int(get_local_size(1)) + ry - 1); 
		
		for (int y = ymin; y <= ymax; ++y) {
			for (int x = xmin; x <= xmax; ++x) {
				localData[TILE_SIZE * (ry + y) + rx + x] = datai[width * (idy + y) + idx + x];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		xmin = tx - rx, xmax = tx + rx;
		ymin = ty - ry, ymax = ty + ry;

		if (xmin + idx < 0)
			xmin = 0;
		else if (xmax + idx >= width)
			xmax = width - idx - 1;
		if (ymin + idy < 0)
			ymin = 0;
		else if (ymax + idy >= height)
			ymax = height - idy - 1;

		uchar mini = 255;
		
		for (int y = ymin; y <= ymax; ++y) {
			for (int x = xmin; x <= xmax; ++x) {
				mini = min(mini, localData[TILE_SIZE * (ry + y) + rx + x]);
			}
		}
		datao[width * (idy + ty) + idx + tx] = mini;
	}
}