__kernel void convolution(__global uchar* datai, __global uchar* datao, const int width, const int height, __global float* kernel_data, const int kwidth, const int kheight, const float ksum)
{
	int idx = get_local_size(0)*get_group_id(0) + get_local_id(0);
	int idy = get_local_size(1)*get_group_id(1) + get_local_id(1);

	if (idx < width && idy < height) {
		int kx, ky, yy, xx;
		int radius_x = (kwidth - 1) /2;
		int radius_y = (kheight - 1) /2;
		float sum = 0.0;

		for (ky = 0; ky < kheight; ++ky) {
			yy = min(max(idy + ky - radius_y, 0), height - 1);

			for (kx = 0; kx < kwidth; ++kx) {
				xx = min(max(idx + kx - radius_x, 0), width - 1);
				
				sum = mad(datai[xx + yy * width], kernel_data[kx + ky * kwidth], sum);
			}
		}

		datao[idx + idy * width] = convert_uchar(sum / ksum);
	}
}