.PHONY: all clean cleanall run_gpp run_tests_gpp

all:
	cd libimage && $(MAKE)
	cd libimage && $(MAKE) test
	cd gpp && $(MAKE)
	cd tests && $(MAKE)

clean:
	cd libimage && $(MAKE) clean
	cd gpp && $(MAKE) clean
	cd tests && $(MAKE) clean

cleanall:
	cd libimage && $(MAKE) cleanall
	cd gpp && $(MAKE) cleanall
	cd tests && $(MAKE) cleanall

run_gpp:
	cd gpp && $(MAKE) run

run_tests_gpp:
	cd tests && $(MAKE) run_gpp

run_tests_sse:
	cd tests && $(MAKE) run_sse