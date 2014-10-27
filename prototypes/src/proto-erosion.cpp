#include <pandore.h>
#include <iostream>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s radius [im_in|-] [im_out|-]"
#define PARC 1
#define FINC 1
#define FOUTC 1

namespace Erosion {
Uchar erode(const Imc2duc &src, Long x, Long y, Long b, Long radius)
{
	Long w = src.Width();
	Long h = src.Height();

	Uchar minvalue = 255;

	for (Long y2 = -radius + y; y2 <= radius + y; ++y2) {
		for (Long x2 = -radius + x; x2 <= radius + x; ++x2) {
			if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h) {
				continue;
			}

			Uchar value = src[b][y2][x2];

			if (minvalue > value) {
				minvalue = value;
			}
		}
	}

	return minvalue;
}

Errc Operator(const Imc2duc &src, Imc2duc &dst, Long radius)
{
	Long w = src.Width();
	Long h = src.Height();

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {
			dst[0][y][x] = erode(src, x, y, 0, radius);
			dst[1][y][x] = erode(src, x, y, 1, radius);
			dst[2][y][x] = erode(src, x, y, 2, radius);
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

	Long radius = atoi(parv[0]);

	if (radius < 0) {
		std::cout << "Expected positive radius value." << std::endl;
		return 1;
	}

	Imc2duc* const src = (Imc2duc *) objs[0];

	objd[0] = new Imc2duc(src->Props());
	Imc2duc* const dst = (Imc2duc *) objd[0];

	Errc result = Erosion::Operator(*src, *dst, radius);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
