#!/usr/bin/env python3
# Boilerplate for launching conveniently parallel jobs.
# Good for exhaustive search of cmd options to yield a desirable result
import subprocess
import sys
import time
import argparse
import shlex
import numpy as np
import random

cmd_template = "grep cmd_template -A NUM_LINES_AFTER -B NUM_LINES_BEFORE grid_search.py"

work_queue = []
for nlines_after in np.arange(1,5,2):
  for nlines_before in np.arange(2,6,3):
      nlines_after_str = str(nlines_after)
      nlines_before_str = str(nlines_before)
      
      the_cmd = cmd_template.replace("NUM_LINES_AFTER", nlines_after_str).replace("NUM_LINES_BEFORE", nlines_before_str)
      work_queue.append(the_cmd)

class JobInfo(object):
    def __init__(self, job, start_time, cmd):
        self.job = job
        self.start_time = start_time
        self.cmd = cmd

MAX_CONCURRENT_JOBS = 8
current_running_jobs = []
while work_queue:
    if len(current_running_jobs) > MAX_CONCURRENT_JOBS:
        # Monitor existing jobs
        while current_running_jobs:
            for the_job in current_running_jobs:
                p1 = the_job.job
                if p1.poll() is None: # the process still running
                    if (time.time() - the_job.start_time) > 1800: # job time out in 30 minutes
                        print('  [TIME OUT] ' + the_job.cmd)
                        p1.kill()
                        current_running_jobs.remove(the_job)
                else: # the process completed
                    current_running_jobs.remove(the_job)
                    outs, errs = p1.communicate()
                    rc = p1.poll()
                    print('  [JOB STATUS] ' + the_job.cmd)
                    print('               %s seconds' % (time.time() - start_time))
                    print('               Exit Status: ' + rc.__str__())
                    if rc == 0: # job success
                        print('[SUCCESS] ' + the_job.cmd)
                        print(outs)
                        sys.exit(0) # stop the parameter searching
                    else:
                        print('               Error Message: ' + errs)
            time.sleep(5)
    else:
        pos = random.randint(0, len(work_queue) - 1)
        the_cmd = work_queue.pop(pos) # randomly pick a job to run
        x = shlex.split(the_cmd)

        print('[Job Launched]: ')
        print(x)
        p1 = subprocess.Popen(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_running_jobs.append(JobInfo(job=p1, start_time=time.time(), cmd=the_cmd))

