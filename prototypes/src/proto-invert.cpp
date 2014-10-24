#include <pandore.h>
#include <iostream>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s [im_in|-] [im_out|-]"
#define PARC 0
#define FINC 1
#define FOUTC 1

namespace Invert {
Errc Operator(const Imc2duc &src, Imc2duc &dst)
{
	Long w = src.Width();
	Long h = src.Height();

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {
			dst[0][y][x] = 255 - src[0][y][x];
			dst[1][y][x] = 255 - src[1][y][x];
			dst[2][y][x] = 255 - src[2][y][x];
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

	Imc2duc* const src = (Imc2duc *) objs[0];

	objd[0] = new Imc2duc(src->Props());
	Imc2duc* const dst = (Imc2duc *) objd[0];

	Errc result = Invert::Operator(*src, *dst);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
