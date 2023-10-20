import ast
import json
import multiprocessing
import numpy as np
import signal
import time
import io
import shutil
import os
import warnings
import argparse
from tqdm import tqdm
from numbers import Number
import itertools
import sys

import multiprocessing as mp
from multiprocessing import Queue, Process

from constants import *


# Suppress syntax warnings
warnings.filterwarnings('ignore')


def get_samples(
    sid, qid, num_samples, dtype, test_args=[], pos=0, 
    early_stopping=10000, max_timeouts=5, append_samples=False
):
    """
    Sample `num_samples` samples from `prog`. 
    dtype is either scalar or list.
    """
    if 'solution' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'solution', sid, 'response.txt')
    else:
        sample_path = os.path.join(DATA_DIR, qid, 'students', sid, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'students', sid, 'response.txt')
    with open(prog_path) as f:
        prog = f.read()
    
    if append_samples and os.path.isfile(sample_path):
        samples = list(np.load(sample_path))
        # ADD
        # samples = list(np.load(sample_path))
        # curr_index = len(samples) 
        # allocate = np.zeros(500000)
        # allocate[:len(samples)] = samples
        # samples = allocate
    else:
        samples = []
        # ADD
        # samples = np.zeros(500000)
        # curr_index = 0
    
    if dtype == 'scalar':
        sample_fn = scalar_sample 
    elif dtype == 'list':
        sample_fn = list_sample
        
    pid = os.getpid()
    pbar = tqdm(range(num_samples), leave=False, position=pos,
                dynamic_ncols=True, nrows=20, postfix=f'{pid}')
    def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
        pbar.close()
        return None, []
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    timeout_cnt = 0
    samples_remaining = num_samples
    
    while samples_remaining > 0:
        start = time.time()
        val = sample_fn(qid, prog, test_args=test_args)
        end = time.time()
        
        if dtype == 'scalar':
            if val is None:
                timeout_cnt += 1
            else:
                samples.append(val)
                # ADD
                # samples[curr_index] = val
                # curr_index += 1
                
        elif dtype == 'list':
            if val is None or len(val) == 0:
                timeout_cnt += 1
            if not isinstance(val[0], Number):
                break
            val = list(val)
            samples.extend(val)
            # ADD
            # samples[curr_index : curr_index + len(val)] = val
            # curr_index += len(val)
                
        # Check for early stopping (timeouts or degenerate distributions)
        if timeout_cnt > max_timeouts:
            samples = []
            break
        if len(samples) >= early_stopping and samples.count(samples[0]) == len(samples):
        # ADD
        # if len(samples[:curr_index]) >= early_stopping and samples.count(samples[0]) == len(samples[:curr_index]):
            samples = [samples[0]]
            break
            
        if val is not None:
            if dtype == 'scalar':
                pbar.update(1)
                samples_remaining -= 1
            if dtype == 'list':
                pbar.update(min(len(val), samples_remaining))
                samples_remaining -= len(val)
    
    pbar.close()
    return samples


def scalar_sample(qid, prog, test_args=[]):
    return exec_program(qid, prog, test_args=test_args)


def list_sample(qid, prog, test_args=[]):
    return exec_program(qid, prog, test_args=test_args, allowed_types=[list])


def evaluate_student_code(qid, prog, test_args, test_agent_name='__test_agent'):
    """
    Evaluate the student code in the context of the associated problem
    environment. Suppress stdout.
    """
    
    import sys
    import io
    test_path = os.path.join(DATA_DIR, qid, 'test_agent.py')
    with open(test_path) as f:
        test_agent = f.read()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(prog, locals(), locals())
        exec(test_agent, locals(), locals())
        val = locals()[test_agent_name](*test_args)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return val


def _alarm_handler(signum, frame):
    raise TimeoutError
    

def exec_program(qid, prog, timeout=1,
                 test_args=[], allowed_types=[]):
    """
    Evaluate the student program and return its return value.
    Return None if student program cannot be evaluated.
    """

    val = None
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    try:  # Attempt to run the program
        start = time.time()
        val = evaluate_student_code(qid, prog, test_args)
        end = time.time()
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:  # Caught an error
        return None
    finally:  # Reset alarm
        signal.alarm(0)

    if not isinstance(val, tuple([Number] + allowed_types)):
        return None

    return val


def get_student_info_single(
        sid, qid, num_samples, dtype, test_label=None, test_args=[],
        pos=None, proc_queue=None, append_samples=False
):
    
    dir_lst = [DATA_DIR, qid]
    dir_lst.append('solution' if 'solution' in sid else 'students')
    if test_label is not None:  # For different test cases
        dir_lst.append(test_label)
    dir_lst += [sid]
    dirname = os.path.join(*dir_lst)

    sample_path = f'{dirname}/samples.npy'
    
    # If samples already exist or we don't want to append, don't do anything
    if not os.path.isfile(sample_path) or append_samples:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        samples = get_samples(
            sid, qid, num_samples, dtype, test_args, pos, 
            append_samples=append_samples
        )

        if samples is not None:
            np.save(sample_path, np.array(samples))

    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))


def get_student_info_multi(
        sids, qid, num_samples, dtype, max_parallel,
        test_suites={None: []}, clear_dir=False, append_samples=False
):

    dirname = os.path.join(SAMPLE_DIR, qid, 'students')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif clear_dir:
        print('removing existing content...')
        shutil.rmtree(dirname)
        os.makedirs(dirname)

    proc_dict = dict()
    proc_queue = Queue(max_parallel)
    def _deque_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        return old_pos

    # Find ast parsable submissions
    error_idx = []
    parsable_submissions = []
    for sid in sids:
        student_path = os.path.join(DATA_DIR, qid, 'students', sid)
        with open(os.path.join(student_path, 'response.txt')) as f:
            prog = f.read()
        submission = {'sid': sid, 'prog': prog}

        try:
            ast.parse(prog)
        except SyntaxError:
            error_idx.append(i)
        else:
            parsable_submissions.append(submission)

    filled_pos = 0
    for i, (test_label, test_args) in enumerate(test_suites.items()):

        pbar = tqdm(parsable_submissions, total=len(parsable_submissions),
                    leave=(i==len(test_suites)-1),
                    dynamic_ncols=True, nrows=20,
                    postfix=f'case {i+1}/{len(test_suites)}')

        error_idx = []
        for submission in pbar:
            sid = submission['sid']
            prog = submission['prog']

            if filled_pos >= max_parallel:
                pos = _deque_proc()
            else:
                pos = filled_pos+1
            filled_pos += 1
            p = Process(target=get_student_info_single,
                        args=[sid, qid, num_samples, dtype,
                              test_label, test_args,
                              pos, proc_queue, append_samples])
            p.start()
            proc_dict[p.pid] = p

    for _ in range(max_parallel):
        _deque_proc()

    with open(f'{dirname}/error_prog_idx.json', 'w') as f:
        json.dump(error_idx, f)
        
        
def sample_soln(
    qid, num_samples, dtype, test_suites={None: []}
):
    dirname = os.path.join(SAMPLE_DIR, qid)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(DATA_DIR, qid, 'solution/solution.py')) as f:
        soln_prog = f.read()

    for test_label, test_args in test_suites.items():
        get_student_info_single(
            'solution', qid, num_samples, dtype,
            test_label, test_args
        )
            
    
def get_test_suite(qid, test_suite_dir, num_tests=20, all_test_cases=False):
    
    suite_fname = os.path.join(test_suite_dir, f'{qid}.pkl')

    if not os.path.isfile(suite_fname):
        gen_fname = os.path.join(test_suite_dir, f'{qid}.py')
        with open(gen_fname) as f:
            script = f.read()
        exec(script, locals(), locals())

        test_suites = {}
        for i in range(num_tests):
            test_case = locals()['generate_test_suites']()
            test_label = f'case_{i}'
            test_suites[test_label] = test_case

        with open(suite_fname, 'wb') as f:
            pickle.dump(test_suites, f)
    else:
        print('loading existing test cases...')
        with open(suite_fname, 'rb') as f:
            test_suites = pickle.load(f)

    if not all_test_cases:
        labels_fname = os.path.join(test_suite_dir, f'{qid}.labels.json')
        chosen_labels = json.load(open(labels_fname))
        print('testing a subset of test cases: ', chosen_labels)
        test_suites = dict((k,v) for k,v in test_suites.items()
                           if k in chosen_labels)
    return test_suites


def main():

    parser = argparse.ArgumentParser(description=('parallel student code sampler.'))
    parser.add_argument('qid', type=str)
    parser.add_argument('dtype', type=str)
    parser.add_argument('--num-samples', type=int, default=500000)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--solution', action='store_true', default=False)
    parser.add_argument('--num-parallel', type=int, default=16)
    parser.add_argument('--clear-existing', action='store_true', default=False)
    parser.add_argument('--test-suite-dir', type=str, default=None)
    parser.add_argument('--all-test-cases', action='store_true', default=False)
    parser.add_argument('--sid', type=str, default=None)
    args = parser.parse_args()

    # Load the test suites, if there are any
    if args.test_suite_dir:
        test_suites = get_test_suite(args.qid, args.test_suite_dir, num_tests=20, 
                                     all_test_cases=args.all_test_cases)
    else:
        test_suites = {None: []}

    if args.solution:
        print('sampling solution...')
        sample_soln(args.qid, args.num_samples, test_suites=test_suites)
        return

    if args.sid is not None:
        with open(os.path.join(DATA_DIR, args.qid, 'students', args.sid)) as f:
            prog = f.read()
        get_student_info_single(args.sid, args.qid, args.num_samples, args.dtype)
        return
    
    student_dir = os.path.join(DATA_DIR, args.qid, 'students')
    sids = os.listdir(student_dir)
    get_student_info_multi(sids, args.qid,
                           args.num_samples, args.dtype, args.max_parallel,
                           test_suites=test_suites,
                           clear_dir=args.clear_dir, append_samples=args.append)


if __name__ == '__main__':
    main()
                                    