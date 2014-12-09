/**
 * @file benchmark.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "benchmark.h"

static uint64_t _clock_frequency = 0;

static uint64_t _timestamp()
{
	uint32_t h, l;

	__asm__ (
		"rdtsc\n\t"
		: "=a" (l), "=d" (h)
	);

	return (((uint64_t) h) << 32) | l;
}

static void _measure_clock_frequency()
{
	uint64_t total = 0;
	uint64_t s, e;
	clock_t cs;
	int i;

	for (i = 0; i < 10; ++i) {
		s = _timestamp();
		cs = clock();
		while (cs == clock())
			s = _timestamp();

		e = _timestamp();
		while (clock() <= cs + 10)
			e = _timestamp();

		total += (e - s) * CLOCKS_PER_SEC / 10;
	}

	_clock_frequency = total / 10;
}

void start_benchmark(Benchmark *benchmark)
{
	benchmark->_start_time = _timestamp();
}

void end_benchmark(Benchmark *benchmark)
{
	benchmark->_end_time = _timestamp();
	benchmark->elapsed_ticks = benchmark->_end_time - benchmark->_start_time;

	if (_clock_frequency == 0) {
		_measure_clock_frequency();
	}

	benchmark->elapsed_time = (double) benchmark->elapsed_ticks / _clock_frequency;
}