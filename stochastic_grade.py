import argparse
import configparser
import json
import math
import multiprocessing
import numpy as np
import os
import pickle
import subprocess
import time
from constants import *
from sampling import *
from scorer import *
from scipy import stats
from tqdm import tqdm


class StochasticGrade():
    """
    Implements the StochasticGrade algorithm.
    """
    
    def __init__(self, qid, scorer, sample_sizes, false_rejection_rate, dtype, num_soln_samples):
        self.qid = qid
        self.scorer = scorer
        self.sample_sizes = sample_sizes
        self.frr = false_rejection_rate
        self.dtype = dtype
        self.num_soln_samples = 500000
        
        soln_sample_path = os.path.join(DATA_DIR, qid, 'solution/solution/samples.npy')
        if os.path.isfile(soln_sample_path):
            self.soln_samples = np.load(soln_sample_path)
        else:
            self.sample_solution()
        
    def sample_solution(self):
        """
        Sample the solution program.
        """
        get_student_info_single('solution', self.qid, self.num_soln_samples, self.dtype, append_samples=False)
        
        file_path = os.path.join(DATA_DIR, self.qid, 'solution', 'solution', 'samples.npy')
        self.soln_samples = np.load(file_path)
        print(f'Generated {self.num_soln_samples} solution samples.')
        
    def sample(self, n, sid):
        """
        Expand the collection of total student samples to n.
        """
        file_path = os.path.join(DATA_DIR, self.qid, 'students', sid, 'samples.npy')
        if not os.path.isfile(file_path):
            num_samples = n
        else:
            self.stud_samples = np.load(file_path)
            num_samples = n - len(self.stud_samples)
            
        get_student_info_single(sid, self.qid, num_samples, self.dtype, append_samples=True)
        self.stud_samples = np.load(file_path)
        
    def grade(self, sid):
        """
        Grade the given student program. 
        """
        self.stud_samples = []
        for i in range(len(self.sample_sizes)):
            sample_size = self.sample_sizes[i]
            self.sample(sample_size, sid)
            frr = self.frr / 2 ** (len(self.sample_sizes) - i)
            score = self.scorer.compute_score(self.stud_samples, self.soln_samples)
            epsilon = self.scorer.rejection_threshold(self.frr, len(self.stud_samples), len(self.soln_samples))
            if score >= epsilon:
                return False, score
        return True, score
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('--config-file', required=True) 
    parser.add_argument('--data-file')
    parser.add_argument('--mode', type=str, choices=['grade'], default='grade')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    if not os.path.isfile(args.config_file):
        print('Invalid config file.')
        raise Exception
    config.read(args.config_file)
    
    qid = args.qid
    mode = args.mode
    dtype = config['Parameters']['dtype']
    sample_sizes = config['Parameters']['sample_sizes']
    frr = config['Parameters']['false_rejection_rate']
    num_soln_samples = config['Parameters']['num_soln_samples']
    
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    
    # Load in the data for sampling
    if args.data_file:
        with open(args.data_file) as f:
            data = json.load(f)
        for sid in data:
            sid_dir = os.path.join(DATA_DIR, qid, sid)
            if not os.path.isdir(sid_dir):
                os.mkdir(sid_dir)
                with open(os.path.join(sid_dir, 'response.txt'), 'w') as f:
                    f.write(data[sid])
        sids = [sid for sid in data]
    else:
        sids = os.listdir(os.path.join(DATA_DIR, qid))
        
    if mode == 'grade':
        labels, scores = {}, {}
        algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, num_soln_samples)
        for sid in sids:
            label, score = algorithm.grade(sid)
            labels[sid] = label
            scores[sid] = score
        with open(os.path.join(DATA_DIR, qid, 'labels.json'), 'w') as f:
            json.dump(labels)
        with open(os.path.join(DATA_DIR, qid, 'scores.json'), 'w') as f:
            json.dump(scores)
   