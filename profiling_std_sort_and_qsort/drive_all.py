#!/usr/bin/env python

import datetime
import subprocess
import numpy

def get_results(algo):
    start = datetime.datetime.now()
    subprocess.check_call(["./test_std_sort_qsort", algo])
    end = datetime.datetime.now()
    duration = end - start
    return duration.seconds + duration.microseconds / 1e6

def bootstrap(algo, runs = 1):
    times = []
    for i in range(runs):
        time = get_results(algo)
        times.append(time)
    mean = numpy.mean(times)
    return mean

def main():
    for algo in ['q', 's']:
        print(algo, bootstrap(algo))

if __name__ == "__main__":
    main()
