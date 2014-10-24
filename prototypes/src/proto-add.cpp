#include <pandore.h>
#include <iostream>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s [im_in1|-] [im_in2|-] [im_out|-]"
#define PARC 0
#define FINC 2
#define FOUTC 1

/*
 * NB : Pandore doesn't handle overflows.
 * We'll have to discuss the output format.
 */
namespace Add {
Errc Operator(const Imc2duc &src1, const Imc2duc &src2, Imc2duc &dst)
{
	Long w = src1.Width();
	Long h = src1.Height();

	if (src2.Width() != w || src2.Height() != h) {
		std::cout << "Input images have to be the same size." << std::endl;
		return FAILURE;
	}

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {
			for (Long c = 0; c < 3; ++c) {
				Long sum = src1[c][y][x] + src2[c][y][x];
				dst[c][y][x] = (sum > 255) ? 255 : sum;
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

	if (objs[0]->Type() != Po_Imc2duc || objs[1]->Type() != Po_Imc2duc) {
		std::cout << "Expected objects of type Imc2duc (color, 2D, uchar)" << std::endl;
		return 1;
	}

	Imc2duc* const src1 = (Imc2duc *) objs[0];
	Imc2duc* const src2 = (Imc2duc *) objs[1];

	objd[0] = new Imc2duc(src1->Props());
	Imc2duc* const dst = (Imc2duc *) objd[0];

	Errc result = Add::Operator(*src1, *src2, *dst);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
