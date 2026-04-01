// timer.h — High-resolution timing via mach_absolute_time
#ifndef TIMER_H
#define TIMER_H

#include <mach/mach_time.h>
#include <stdint.h>

static mach_timebase_info_data_t _timebase_info;

static inline void timer_init(void) {
    mach_timebase_info(&_timebase_info);
}

static inline uint64_t timer_now(void) {
    return mach_absolute_time();
}

// Returns elapsed time in milliseconds
static inline double timer_ms(uint64_t start, uint64_t end) {
    uint64_t elapsed = end - start;
    return (double)elapsed * _timebase_info.numer / _timebase_info.denom / 1e6;
}

// Returns elapsed time in microseconds
static inline double timer_us(uint64_t start, uint64_t end) {
    uint64_t elapsed = end - start;
    return (double)elapsed * _timebase_info.numer / _timebase_info.denom / 1e3;
}

#endif // TIMER_H
