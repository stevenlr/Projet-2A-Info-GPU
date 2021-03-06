#include <pandore.h>
#include <iostream>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s radius [im_in|-] [im_out|-]"
#define PARC 1
#define FINC 1
#define FOUTC 1

namespace Erosion {
Uchar erode(const Img2duc &src, Long x, Long y, Long radius)
{
	Long w = src.Width();
	Long h = src.Height();

	Uchar minvalue = 255;

	for (Long y2 = -radius + y; y2 <= radius + y; ++y2) {
		for (Long x2 = -radius + x; x2 <= radius + x; ++x2) {
			if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h) {
				continue;
			}

			Uchar value = src[y2][x2];

			if (minvalue > value) {
				minvalue = value;
			}
		}
	}

	return minvalue;
}

Errc Operator(const Img2duc &src, Img2duc &dst, Long radius)
{
	Long w = src.Width();
	Long h = src.Height();

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {
			dst[y][x] = erode(src, x, y, radius);
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

	if (objs[0]->Type() != Po_Img2duc) {
		std::cout << "Expected object of type Img2duc (grayscale, 2D, uchar)" << std::endl;
		return 1;
	}

	Long radius = atoi(parv[0]);

	if (radius < 0) {
		std::cout << "Expected positive radius value." << std::endl;
		return 1;
	}

	Img2duc* const src = (Img2duc *) objs[0];

	objd[0] = new Img2duc(src->Props());
	Img2duc* const dst = (Img2duc *) objd[0];

	Errc result = Erosion::Operator(*src, *dst, radius);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
