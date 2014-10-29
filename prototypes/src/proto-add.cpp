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
Errc Operator(const Img2duc &src1, const Img2duc &src2, Img2duc &dst)
{
	Long w = src1.Width();
	Long h = src1.Height();

	if (src2.Width() != w || src2.Height() != h) {
		std::cout << "Input images have to be the same size." << std::endl;
		return FAILURE;
	}

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {
			Long sum = src1[y][x] + src2[y][x];
			dst[y][x] = (sum > 255) ? 255 : sum;
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

	if (objs[0]->Type() != Po_Img2duc || objs[1]->Type() != Po_Img2duc) {
		std::cout << "Expected objects of type Img2duc (grayscale, 2D, uchar)" << std::endl;
		return 1;
	}

	Img2duc* const src1 = (Img2duc *) objs[0];
	Img2duc* const src2 = (Img2duc *) objs[1];

	objd[0] = new Img2duc(src1->Props());
	Img2duc* const dst = (Img2duc *) objd[0];

	Errc result = Add::Operator(*src1, *src2, *dst);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
