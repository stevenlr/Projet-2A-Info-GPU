#include <pandore.h>
#include <iostream>

using namespace pandore;

#undef MAIN

#define USAGE "usage: %s threshold [im_in|-] [im_out|-]"
#define PARC 1
#define FINC 1
#define FOUTC 1

/*
 * To check : thresholding method : per channel or average.
 */
namespace Threshold {
Errc Operator(const Img2duc &src, Img2duc &dst, Uchar threshold)
{
	Long w = src.Width();
	Long h = src.Height();

	for (Long y = 0; y < h; ++y) {
		for (Long x = 0; x < w; ++x) {

			Uchar value = (src[y][x] >= threshold) ? 255 : 0;

			dst[y][x] = value;
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

	Uchar threshold = atoi(parv[0]);

	if (threshold < 0 || threshold > 255) {
		std::cout << "Expected threshold value. Expected value between 0 and 255." << std::endl;
		return 1;
	}

	Img2duc* const src = (Img2duc *) objs[0];

	objd[0] = new Img2duc(src->Props());
	Img2duc* const dst = (Img2duc *) objd[0];

	Errc result = Threshold::Operator(*src, *dst, threshold);

	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask,
		  objin, objs, objout, objd);

	Exit(result);
	return 0;
}
