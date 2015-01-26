.PHONY: all clean cleanall run_gpp run_sse run_tests_gpp run_tests_sse

all:
	cd libimage && $(MAKE)
	cd libimage && $(MAKE) test
	cd gpp && $(MAKE)
	cd gpp-sse && $(MAKE)
	cd gpp-tbb && $(MAKE)
	cd tests && $(MAKE)

clean:
	cd libimage && $(MAKE) clean
	cd gpp && $(MAKE) clean
	cd gpp-sse && $(MAKE) clean
	cd gpp-tbb && $(MAKE) clean
	cd tests && $(MAKE) clean

cleanall:
	cd libimage && $(MAKE) cleanall
	cd gpp && $(MAKE) cleanall
	cd gpp-sse && $(MAKE) cleanall
	cd gpp-tbb && $(MAKE) cleanall
	cd tests && $(MAKE) cleanall

run_gpp:
	cd gpp && $(MAKE) run

run_sse:
	cd gpp-sse && $(MAKE) run

run_tbb:
	cd gpp-tbb && $(MAKE) run

run_tests_gpp:
	cd tests && $(MAKE) run_gpp

run_tests_sse:
	cd tests && $(MAKE) run_sse

run_tests_tbb:
	cd tests && $(MAKE) run_tbb
