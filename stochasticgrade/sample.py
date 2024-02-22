import argparse
import ast
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import numpy as np
import os
import pickle
import shutil
import signal
import sys
import time
import warnings

from tqdm import tqdm
from multiprocessing import Process, Queue
from numbers import Number

from stochasticgrade.constants import *


# Suppress syntax warnings
warnings.filterwarnings('ignore')


def get_samples(
    sid, qid, num_samples, dtype, test_label='', test_args=[],
    early_stopping=10000, max_timeouts=5, append_samples=False, pos=0
):
    """
    Sample `num_samples` samples from `prog`. 
    dtype is either scalar or list.
    """
    
    # Load the program from which we will be sampling. If samples already exist,
    # load the samples in as well. 
    if 'solution' in sid:
        sample_path = os.path.join(DATA_DIR, qid, 'solution', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'solution', 'solution', 'response.txt')
    else:
        sample_path = os.path.join(DATA_DIR, qid, 'students', sid, test_label, 'samples.npy')
        prog_path = os.path.join(DATA_DIR, qid, 'students', sid, 'response.txt')
    with open(prog_path) as f:
        prog = f.read()
        
    if append_samples and os.path.isfile(sample_path):
        samples = list(np.load(sample_path, allow_pickle=True))
    else:
        samples = []
    
    if dtype == 'scalar':
        sample_fn = scalar_sample 
    elif dtype == 'list':
        sample_fn = list_sample
    elif 'array_shape_' in dtype:
        sample_fn = multidim_sample
        
    # Set up progress bar and error handling
    pid = os.getpid()
    pbar = tqdm(range(num_samples), leave=False, position=pos,
                dynamic_ncols=True, nrows=20, postfix=f'{pid}')
    def sigterm_handler(_signo, _stack_frame):
        pbar.close()
        return None, []
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    timeout_cnt = 0
    samples_remaining = num_samples
    
    # Begin sampling
    while samples_remaining > 0:
        
        # Check for early stopping (timeouts or degenerate distributions)
        if timeout_cnt > max_timeouts:
            samples = []
            break
        if len(samples) == early_stopping and len(np.unique(samples)) == 1:
            samples = [samples[0]]
            break
        
        # Proceed with sampling as normal
        start = time.time()
        val = sample_fn(qid, prog, dtype, test_args=test_args)
        end = time.time()
        
        # Add the newly sampled value(s) to the list of samples
        if dtype == 'scalar':
            if val is None:
                timeout_cnt += 1
                break
            samples.append(val)
                
        if dtype == 'list':
            if val is None or len(val) == 0:
                timeout_cnt += 1
                break
            if not isinstance(val[0], Number):
                break
            val = list(val)
            samples.extend(val)
            
        if 'array_shape_' in dtype: 
            shape = tuple([int(i) for i in dtype.split('array_shape_')[1][1:-1].split(',')])
            if val is None or len(val) == 0:
                timeout_cnt += 1
                break
            if np.array(val).shape != shape:
                break
            samples.append(val)
         
        # Update the progress bar if the sample is valid
        if val is not None:
            if dtype == 'scalar':
                pbar.update(1)
                samples_remaining -= 1
            if dtype == 'list':
                pbar.update(min(len(val), samples_remaining))
                samples_remaining -= len(val)
            if 'array_shape_' in dtype:
                pbar.update(1)
                samples_remaining -= 1
    
    pbar.close()
    return samples


def scalar_sample(qid, prog, dtype, test_args=[]):
    """
    Execution function for scalar data types.
    """
    return exec_program(qid, prog, test_args=test_args)


def list_sample(qid, prog, dtype, test_args=[]):
    """
    Execution function for list data type.
    """
    return exec_program(qid, prog, test_args=test_args, allowed_types=[list])


def multidim_sample(qid, prog, dtype, test_args=[]):
    """
    Execution function for multidimensional data types.
    """
    return exec_program(qid, prog, test_args=test_args, allowed_types=[list])


def evaluate_student_code(qid, prog, test_args=[], test_agent_name='__test_agent'):
    """
    Evaluate the student code in the context of the associated problem
    environment. Suppress stdout.
    """
    
    # Load in the testing program 
    test_path = os.path.join(DATA_DIR, qid, 'test_agent.py')
    with open(test_path) as f:
        test_agent = f.read()
        
    # Redirect output to prevent extraneous printing
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            # Execute the program and record the output
            exec(prog, locals(), locals())
            exec(test_agent, locals(), locals())
            val = locals()[test_agent_name](*test_args)
        except:
            pass

    return val


def _alarm_handler(signum, frame):
    raise TimeoutError

    
def is_valid_program(code):
    # NOT WORKING?
    import signal
    import builtins
    from unittest.mock import patch

    # Redirect output to prevent extraneous printing
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    val = False
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(1)

    try:
        # Check for syntactic validity
        ast.parse(code)
        # Check for runtime errors
        with patch.object(builtins, 'input', lambda _: 'mocked_input'):
            exec(code)
        val = True
    except (SyntaxError, AttributeError, Exception):
        val = False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        signal.alarm(0)
    return val
    

def exec_program(qid, prog, timeout=1, test_args=[], allowed_types=[]):
    """
    Evaluate the student program and return its return value.
    Return None if student program cannot be evaluated.
    """

    val = None
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    try:  # Attempt to run the program
        val = evaluate_student_code(qid, prog, test_args=test_args)
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
    sid, qid, num_samples, dtype, 
    test_label='', test_args=[], append_samples=False, pos=0, proc_queue=None
):
    """
    Sampling process for singlethreaded sampling. 
    """
    
    # Create the appropriate directory name where samples will be stored
    dir_lst = [DATA_DIR, qid]
    dir_lst.append('solution' if 'solution' in sid else 'students')
    if 'mc_solution_' in sid:
        dir_lst.append('mc_solutions')
    dir_lst += [sid]
    dir_lst += [test_label]
    dirname = os.path.join(*dir_lst)
    
    sample_path = os.path.join(dirname, 'samples.npy')
    
    # If samples already exist or we don't want to append, don't do anything
    if not os.path.isfile(sample_path) or append_samples:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        samples = get_samples(
            sid, qid, num_samples, dtype, test_label=test_label, 
            test_args=test_args, append_samples=append_samples, pos=pos
        )
        
        np.save(sample_path, np.array(samples))   
        
    if proc_queue is not None:
        proc_queue.put((mp.current_process().pid, pos))
        
        
def get_student_info_multi(
        sids, qid, num_samples, dtype, max_parallel,
        test_label='', test_args=[], append_samples=False, clear_dir=False
):
    # Remove existing directory if requested / create directory if needed
    dirname = os.path.join(DATA_DIR, qid, 'students')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif clear_dir:
        print('Removing existing content...')
        shutil.rmtree(dirname)
        os.makedirs(dirname)

    # Set up parallelization
    proc_dict = dict()
    proc_queue = Queue(max_parallel)
    def _deque_proc():
        (old_pid, old_pos) = proc_queue.get()
        old_proc = proc_dict[old_pid]
        old_proc.join()
        old_proc.close()
        return old_pos
    filled_pos = 0
    pbar = tqdm(sids, total=len(sids), dynamic_ncols=True, nrows=20)

    # Begin sampling for each sid
    for sid in pbar:
        if filled_pos >= max_parallel:
            getpos = _deque_proc()
        else:
            pos = filled_pos + 1 
        filled_pos += 1
        p = Process(target=get_student_info_single,
                    args=[sid, qid, num_samples, dtype, test_label, test_args,
                          append_samples, pos, proc_queue])
        p.start()
        proc_dict[p.pid] = p

    for _ in range(max_parallel):
        _deque_proc()
            
    
def get_test_suite(qid, num_tests=20, all_test_suites=False):
    """
    Create test suite cases for problems in which multiple sample sets
    from the same program are useful.
    """
    
    test_suite_dir = os.path.join(DATA_DIR, qid, 'test_suites')
    if not os.path.isdir(test_suite_dir):
        os.mkdir(test_suite_dir)
        print(f'Must initialize the {qid}.py file containing the', 
              'generate_test_suites() function in the test suite directory.')
        print('If you would like to specify a subset of cases, do so', 
              f'in a {qid}.labels.json file in the test suite directory.')
        raise Exception
        
    
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

    if not all_test_suites:
        labels_fname = os.path.join(test_suite_dir, f'{qid}.labels.json')
        chosen_labels = json.load(open(labels_fname))
        print('testing a subset of test cases: ', chosen_labels)
        test_suites = dict((k,v) for k,v in test_suites.items()
                           if k in chosen_labels)
    return test_suites
                                    