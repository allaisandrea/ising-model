#!/usr/bin/env python3
import collections
import signal
import threading
import shlex
import subprocess
import time
import tabulate
import sys
import readline


QueuedJob = collections.namedtuple(
    'QueuedJob', ['job_id', 'args', 'queued_time'])
RunningJob = collections.namedtuple(
    'RunningJob', ['job_id', 'args', 'start_time', 'popen'])
FinishedJob = collections.namedtuple(
    'FinishedJob', ['job_id', 'args', 'start_time', 'end_time', 'status'])
State = collections.namedtuple(
    'State', ['lock', 'queued_jobs', 'running_jobs',
              'finished_jobs', 'next_job_id', 'keep_running'])


def clear_finished_jobs(state):
    old_running_jobs = list(state.running_jobs)
    state.running_jobs.clear()
    for job in old_running_jobs:
        return_code = job.popen.poll()
        if return_code is None:
            state.running_jobs.append(job)
        else:
            state.finished_jobs.append(FinishedJob(
                job_id=job.job_id,
                args=job.args,
                start_time=job.start_time,
                end_time=time.time(),
                status="completed"))


def start_queued_jobs(max_running_jobs, state):
    while len(state.running_jobs) < max_running_jobs and len(state.queued_jobs) > 0:
        job = state.queued_jobs.pop(0)
        tokens = shlex.split(job.args)
        if len(tokens) == 0:
            return
        if tokens[0].startswith('CWD='):
            cwd = tokens[0][4:]
            tokens = tokens[1:]
        else:
            cwd = None
        try:
            popen = subprocess.Popen(tokens, cwd=cwd)
        except OSError as err:
            print('OS error: {}'.format(err))
            continue
        state.running_jobs.append(RunningJob(
            job_id=job.job_id,
            args=job.args,
            start_time=time.time(),
            popen=popen))


def process_queues(max_running_jobs, state):
    with state.lock:
        if not state.keep_running[0]:
            for job in state.running_jobs:
                job.popen.kill()
            return False

        clear_finished_jobs(state)
        start_queued_jobs(max_running_jobs, state)
        return True


def scheduler_loop(max_running_jobs, state):
    while process_queues(max_running_jobs, state):
        time.sleep(5)


def list_jobs(state):
    with state.lock:
        now = time.time()

        print_list = list()
        for job in state.finished_jobs:
            print_list.append((
                job.job_id,
                job.args,
                time.asctime(time.gmtime(job.start_time)),
                time.asctime(time.gmtime(job.end_time)),
                job.status))
        print("Completed:")
        print(tabulate.tabulate(
            print_list, headers=['job_id', 'args', 'start_time', 'end_time', 'status']))

        print_list = list()
        for job in state.running_jobs:
            print_list.append((
                job.job_id,
                job.args,
                time.asctime(time.gmtime(job.start_time)),
                now - job.start_time))
        print("Running:")
        print(tabulate.tabulate(
            print_list, headers=['job_id', 'args', 'start_time', 'elapsed_time']))

        print_list = list()
        for job in state.queued_jobs:
            print_list.append((
                job.job_id,
                job.args,
                time.asctime(time.gmtime(job.queued_time)),
                now - job.queued_time))
        print("Queued:")
        print(tabulate.tabulate(
            print_list, headers=['job_id', 'args', 'queued_time', 'time_in_queue']))


def submit_job(args, state):
    with state.lock:
        state.queued_jobs.append(QueuedJob(
            job_id=state.next_job_id[0],
            args=args,
            queued_time=time.time()))
        state.next_job_id[0] += 1


def cancel_job(job_id, state):
    with state.lock:
        for job in state.running_jobs:
            if job.job_id == job_id:
                job.popen.kill()
                state.running_jobs.remove(job)
                state.finished_jobs.append(FinishedJob(
                    job_id=job.job_id,
                    args=job.args,
                    start_time=job.start_time,
                    end_time=time.time(),
                    status="canceled"))
                break

        for job in state.queued_jobs:
            if job.job_id == job_id:
                state.queued_jobs.remove(job)
                state.finished_jobs.append(FinishedJob(
                    job_id=job.job_id,
                    args=job.args,
                    start_time=time.time(),
                    end_time=time.time(),
                    status="canceled"))
                break


def load_jobs_from_file(file_name, state):
    try:
        with open(file_name, 'r') as stream:
            for line in stream.readlines():
                if not (line.startswith('#') or line.startswith('\n')):
                    submit_job(line, state)
    except FileNotFoundError:
        print("Cannot open file \"{}\"".format(file_name))


def quit(state):
    with state.lock:
        state.keep_running[0] = False


def process_user_interaction(state):
    try:
        command = input('> ')
    except EOFError:
        print('Type quit to exit')
        return True
    tokens = command.split(' ', 1)
    if len(tokens) == 0:
        print('Cannot parse "{}"'.format(command))
        return True
    verb = tokens[0]
    if verb == 'ls':
        if len(tokens) != 1:
            print('Cannot parse "{}"'.format(command))
            return True
        list_jobs(state)
    elif verb == 'sub':
        if len(tokens) != 2:
            print('Cannot parse "{}"'.format(command))
            return True
        submit_job(tokens[1], state)
    elif verb == 'cancel':
        if len(tokens) != 2:
            print('Cannot parse "{}"'.format(command))
            return True
        try:
            job_id = int(tokens[1])
        except ValueError:
            print('Cannot parse "{}"'.format(command))
            return True
        cancel_job(job_id, state)
    elif verb == 'load':
        if len(tokens) != 2:
            print('Cannot parse "{}"'.format(command))
            return True
        load_jobs_from_file(tokens[1], state)
    elif verb == 'quit':
        if len(tokens) != 1:
            print('Cannot parse "{}"'.format(command))
            return True
        quit(state)
        return False
    else:
        print('Cannot parse "{}"'.format(command))
    return True


def user_interaction_loop(state):
    while process_user_interaction(state):
        pass


def main():
    if len(sys.argv) != 2:
        print('job_manager.py max_processes')
        sys.exit(-1)
    try:
        max_running_jobs = int(sys.argv[1])
    except ValueError:
        print('job_manager.py max_processes')
        sys.exit(-1)

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    state = State(
        lock=threading.Lock(),
        queued_jobs=list(),
        running_jobs=list(),
        finished_jobs=list(),
        next_job_id=[0],
        keep_running=[True])
    scheduler_thread = threading.Thread(
        target=scheduler_loop, args=(max_running_jobs, state))
    scheduler_thread.start()
    user_interaction_loop(state)
    scheduler_thread.join()
    sys.exit(0)


if __name__ == '__main__':
    main()
