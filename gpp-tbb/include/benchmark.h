/**
 * @file benchmark.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <stdint.h>

typedef struct {
	uint64_t _start_time;	/**< Start time, in ticks. */
	uint64_t _end_time;	/**< End time, in ticks. */
	uint64_t elapsed_ticks;	/**< Elapsed time in ticks. */
	double elapsed_time;	/**< Elapsed time in seconds. */
} Benchmark;

/**
 * Starts a benchmark.
 * @param benchmark Structure storing the timer.
 */
void start_benchmark(Benchmark *benchmark);

/**
 * Ends a benchmark.
 * @param benchmark Structure storing the timer and result.
 */
void end_benchmark(Benchmark *benchmark);

#endif
