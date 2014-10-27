#include <pandore.h>
#include <iostream>
#include <limits>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s matrixfile edge_mode [im_in|-] [im_out|-]\nEdge modes: 0 = warp, 1 = extend, 2 = crop"
#define PARC 2
#define FINC 1
#define FOUTC 1

struct Kernel {
	Long width;
	Long height;
	Float sum;
	Float *data;
};

Kernel get_kernel(const char *file_name)
{
	FILE *fp = fopen(file_name, "r");
	Kernel kernel;

	kernel.data = NULL;
	kernel.sum = 0;

	if (fp == NULL) {
		std::cout << "Couldn't open kernel file " << file_name << std::endl;
		return kernel;
	}

	fscanf(fp, "%d*%d", &(kernel.width), &(kernel.height));

	if (kernel.width % 2 != 1 || kernel.height % 2 != 1) {
		std::cout << "Kernel should have odd dimensions." << std::endl;
		fclose(fp);
		return kernel;
	}

	Long size = kernel.width * kernel.height;

	try {
		kernel.data = new Float[size];
	} catch (std::bad_alloc e) {
		std::cout << "Error when allocating kernel data." << std::endl;
		fclose(fp);
		return kernel;
	}

	for (int i = 0; i < size; ++i) {
		fscanf(fp, "%f", kernel.data + i);
		kernel.sum += kernel.data[i];
	}

	if (abs(kernel.sum) <= std::numeric_limits<Float>::round_error()) {
		kernel.sum = 1;
	}

	fclose(fp);

	return kernel;
}

void delete_kernel(Kernel kernel)
{
	delete[] kernel.data;
}

namespace Convolution {
Uchar convolution(const Imc2duc &src, Kernel kernel, Long edge_mode, Long b, Long x, Long y)
{
	Long kernel_radius_w = (kernel.width - 1) / 2;
	Long kernel_radius_h = (kernel.height - 1) / 2;

	Long src_offset_x = (edge_mode == 2) ? kernel_radius_w : 0;
	Long src_offset_y = (edge_mode == 2) ? kernel_radius_h : 0;

	Long w = src.Width();
	Long h = src.Height();

	Float value = 0;

	for (Long ky = 0; ky < kernel.height; ++ky) {
		for (Long kx = 0; kx < kernel.width; ++kx) {
			Long sx = x + src_offset_x + kx - kernel_radius_w;
			Long sy = y + src_offset_y + ky - kernel_radius_h;

			if (edge_mode == 0) {
				sx = (sx + w) % w;
				sy = (sy + h) % h;
			} else if (edge_mode == 1) {
				sx = std::max(static_cast<Long>(0), std::min(sx, w - 1));
				sy = std::max(static_cast<Long>(0), std::min(sy, h - 1));
			}

			value += kernel.data[kx + kernel.width * ky] * src[b][sy][sx];
		}
	}

	value /= kernel.sum;
	value = std::max(static_cast<Float>(0), std::min(static_cast<Float>(255), value));

	return static_cast<Uchar>(value);
}

Errc Operator(const Imc2duc &src, Imc2duc &dst, Kernel kernel, const Long edge_mode)
{
	for (Long y = 0; y < dst.Height(); ++y) {
		for (Long x = 0; x < dst.Width(); ++x) {
			for (Long b = 0; b < 3; ++b) {
				dst[b][y][x] = convolution(src, kernel, edge_mode, b, x, y);
			}
		}
	}

	return SUCCESS;
}
};

int main(int argc, char *argv[])
{
	Pobject *mask;
	Pobject *objin[FINC + 1];
	Pobject *objs[FINC + 1];
	Pobject *objout[FOUTC + 1];
	Pobject *objd[FOUTC + 1];
	char *parv[PARC + 1];

	ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		 objin, objs, objout, objd, parv, USAGE);

	if (objs[0]->Type() != Po_Imc2duc) {
		std::cout << "Expected object of type Imc2duc (color, 2D, uchar)" << std::endl;
		return 1;
	}

	Long edge_mode = atoi(parv[1]);

	if (edge_mode < 0 || edge_mode > 2) {
		std::cout << "Edge mode should be either 0, 1 or 2." << std::endl;
		return 1;
	}

	Kernel kernel = get_kernel(parv[0]);

	if (kernel.data == NULL) {
		std::cout << "Error when reading kernel file." << std::endl;
		return 1;
	}

	Imc2duc* const src = (Imc2duc *) objs[0];

	Dimension2d output_size(src->Width(), src->Height());

	if (edge_mode == 2) {
		output_size.w -= kernel.width - 1;
		output_size.h -= kernel.height - 1;
	}

	objd[0] = new Imc2duc(output_size);
	Imc2duc* const dst = (Imc2duc *) objd[0];

	Errc result = Convolution::Operator(*src, *dst, kernel, edge_mode);

	delete_kernel(kernel);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
