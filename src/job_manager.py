#!/usr/bin/env python3
import collections
import threading
import shlex
import subprocess
import time
import tabular


QueuedJob = collections.NamedTuple(
    'QueuedJob', ['job_id', 'args', 'queued_time'])
RunningJob = collections.NamedTuple(
    'RunningJob', ['job_id', 'args', 'start_time', 'popen'])
CompletedJob = collections.NamedTuple(
        'CompletedJob', ['job_id', 'args', 'start_time', 'end_time', 'return_code'])
Queues = collections.NamedTuple(
        'Queues', ['queued_jobs', 'running_jobs', 'completed_jobs'])


lock = threading.Lock()
keep_running = True
job_id = 0


def scheduler_loop(max_running_jobs, queues):
    global keep_running, lock
    while True:
        lock.acquire()
        if not keep_running:
            for job in queues.running_jobs:
                job.popen.kill()
            lock.release()
            break

        new_running_jobs = list()
        for job in queues.running_jobs:
            return_code = job.popen.poll()
            if return_code is not None:
                queues.completed_jobs.append(CompletedJob(
                    job_id=job.job_id,
                    args=job.args,
                    start_time=job.start_time,
                    end_time=time.time(),
                    return_code=return_code))
                print('Job {} completed'.format(job.job_id))
            else:
                new_running_jobs.append(running_job)
        queues.running_jobs = new_running_jobs

        while len(queues.running_jobs) < max_running_jobs and len(queues.queued_jobs) > 0:
            queued_job = queues.queued_jobs.pop(0)
            try:
                popen = subprocess.Popen(queued_job.args)
            except OSError as err:
                print('OS error: {}'.format(err))
                continue
            print('Job {} started'.format(job.job_id))
            queues.running_jobs.append(RunningJob(
                job_id=queued_job.job_id,
                args=queued_job.args,
                start_time=time.time(),
                popen=popen))
        lock.release()
        time.sleep(30)


def list_jobs(queues):
    global lock
    lock.acuqure()
    now = time.time()
    print_list = list()
    for job in queues.running_jobs:
        print_list.append((
            job.job_id,
            ' '.join(job.args),
            job.start_time,
            now - job.start_time))
    print("Running:")
    print(tabulate.tabulate(
        print_list, headings=['job_id', 'args', 'start_time', 'elapsed_time']))

    for job in queues.queued_jobs:
        print_list.append((
            job.job_id,
            ' '.join(job.args),
            job.queued_time,
            now - job.queued_time))
    print("Queued:")
    print(tabulate.tabulate(
        print_list, headings=['job_id', 'args', 'queued_time', 'time_in_queue']))
    lock.release()


def submit_job(args_str, queues):
    global lock, job_id
    lock.acquire()
    queues.queued_jobs.append(QueuedJob(
        job_id=job_id,
        args=shlex.split(args_str),
        queued_time=time.time()))
    job_id += 1
    lock.release()


def cancel_job(job_id, queues):
    lock.acquire()
    for job in queues.running_jobs:
        if job.job_id == job_id:
            job.popen.kill()
            running_jobs.remove(job)
            break

    for job in queues.queued_jobs:
        if job.job_id == job_id:
            queues.queued_jobs.remove(job)
            break
    lock.release()


def user_interaction_loop(queues):
    global keep_running, lock
    while True:
        command = input('>')
        pair = command.split(' ', 2)
        if len(pair) == 0:
            print('Cannot parse "{}"'.format(command))
            continue
        command = pair[0]
        if command == 'ls':
            list_jobs(queues)
        elif command == 'sub':
            if len(pair) != 2:
                print('Cannot parse "{}"'.format(command))
                continue
            submit_job(pair[1], queues)
        elif command == 'cancel':
            if len(pair) != 2:
                print('Cannot parse "{}"'.format(command))
                continue
            try:
                job_id = int(pair[1])
            except ValueError:
                print('Cannot parse "{}"'.format(command))
                continue
            cancel_job(job_id, queues)
        elif command == 'stop':
            lock.acquire()
            keep_running = False
            lock.release()
            break
        else:
            print('Cannot parse "{}"'.format(command))
            continue


def main():
    if len(sys.argv) != 2):
        print('job_manager.py max_processes')
        sys.exit(-1)
    try:
        max_running_jobs=int(sys.argv[2])
    except ValueError:
        print('job_manager.py max_processes')
        sys.exit(-1)

    queues=Queues(
            queued_jobs = list(),
            running_jobs = list(),
            completed_jobs = list())
    scheduler_thread=threading.Thread(
        targed = scheduler, args = (max_running_jobs, queues))
    user_interaction_loop(queues)
    scheduler_thread.join()
    sys.exit(0)

if __name__ == '__main__':
    main()
