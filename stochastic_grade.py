import argparse
import configparser
import json
import math
import multiprocessing
import numpy as np
import os
import pickle
import shutil
import subprocess
import time
from cluster import *
from constants import *
from sample import *
from score import *
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
            self.soln_samples = np.load(soln_sample_path, allow_pickle=True)
        else:
            self.sample_solution()
        
    def sample_solution(self, case=None):
        """
        Sample the solution program.
        """
        print('Sampling the solution program.')
        sid = 'solution' if case is None else f'case_{case}/solution'
        get_student_info_single('solution', self.qid, self.num_soln_samples, self.dtype, append_samples=False)
        file_path = os.path.join(DATA_DIR, self.qid, 'solution', sid, 'samples.npy')
        self.soln_samples = np.load(file_path, allow_pickle=True)
        print(f'Generated {self.num_soln_samples} solution samples.')
        
    def sample(self, n, sid, case=None):
        """
        Expand the collection of total student samples to n.
        """
        file_path = os.path.join(DATA_DIR, self.qid, 'students', sid, 'samples.npy')
        if not os.path.isfile(file_path):
            num_samples = n
        else:
            self.stud_samples = np.load(file_path, allow_pickle=True)
            num_samples = n - len(self.stud_samples)
            
        get_student_info_single(sid, self.qid, num_samples, self.dtype, append_samples=True)
        self.stud_samples = np.load(file_path, allow_pickle=True)
        
    def monte_carlo(self, num_samples, M=1000, overwrite=False):
        """
        Sample M solution program sample sets of size `num_samples`.
        """
        monte_carlo_dir = os.path.join(DATA_DIR, self.qid, 'solution', 'mc_solutions')
        if overwrite and os.path.isdir(monte_carlo_dir):
            shutil.rmtree(monte_carlo_dir)
            
        if not os.path.isdir(monte_carlo_dir):
            os.makedirs(monte_carlo_dir)
        for i in tqdm(range(M)):
            sid = f'mc_solution_{i+1}'
            if sid not in os.listdir(monte_carlo_dir) or overwrite:
                get_student_info_single(sid, self.qid, num_samples, self.dtype, append_samples=True)                
        
    def grade(self, sid, case=None):
        """
        Grade the given student program. 
        """
        self.stud_samples = []
        start = time.time()
        
        for i in range(len(self.sample_sizes)):
            sample_size = self.sample_sizes[i]
            self.sample(sample_size, sid)
            if len(self.stud_samples) < sample_size:
                return False, 1e7, sample_size, time.time() - start
            frr = self.frr / 2 ** (len(self.sample_sizes) - i)
            score = self.scorer.compute_score(self.stud_samples[:sample_size], self.soln_samples)
            monte_carlo_path = os.path.join(DATA_DIR, self.qid, 'results', str(self.scorer), 'monte_carlo_scores.json')
            epsilon = self.scorer.rejection_threshold(
                self.frr, len(self.stud_samples[:sample_size]), len(self.soln_samples), monte_carlo_path=monte_carlo_path
            )
            if score >= epsilon:
                return False, score, sample_size, time.time() - start
            
        # Return the correctness, score, samples needed, and runtime
        return True, score, sample_size, time.time() - start
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('qid', type=str)
    parser.add_argument('--config-file') 
    parser.add_argument('--data-file')
    parser.add_argument('--mode', type=str, 
                        choices=['single-grade', 'batch-grade', 'cluster', 'monte-carlo', 'hand-label'], 
                        default='batch-grade')
    parser.add_argument('--sid', type=str)
    parser.add_argument('--case', type=str)
    args = parser.parse_args()
    
    # Load the command line arguments
    single_grade_sid = args.sid
    case = args.case
    qid = args.qid
    mode = args.mode
    
    # Load the config file
    config = configparser.ConfigParser()
    if args.config_file:
        if not os.path.isfile(args.config_file):
            print('Invalid config file.')
            raise Exception
        config.read(args.config_file)
    else:
        if os.path.isfile(os.path.join(DATA_DIR, qid, 'config.ini')):   
            config.read(os.path.join(DATA_DIR, qid, 'config.ini'))
        else:
            print('Please create a config.ini file in the qid directory.')
            raise Exception
            
    # Load the model parameters from the config file
    dtype = config['Parameters']['dtype']
    sample_sizes = [int(num) for num in config['Parameters']['sample_sizes'].split(',')]
    frr = float(config['Parameters']['false_rejection_rate'])
    num_soln_samples = int(config['Parameters']['num_soln_samples'])
    
    scorer_name = config['Parameters']['scorer']
    scorer_map = make_scorer_map()
    scorer = scorer_map[scorer_name]
    
    # Load the clustering parameters
    n_clusters = int(config['Clustering']['n_clusters'])
    bandwidth = float(config['Clustering']['bandwidth'])
    cluster_scorers = [scorer for scorer in config['Clustering']['cluster_scorers'].split(',')]
    cluster_method = config['Clustering']['cluster_method']
    
    # Load the Monte Carlo sampling parameters
    overwrite = False if config['Monte Carlo']['overwrite'] == 'False' else True
    M = int(config['Monte Carlo']['M'])
    mc_num_samples = int(config['Monte Carlo']['mc_num_samples'])
    mc_num_anchor_samples = int(config['Monte Carlo']['mc_num_anchor_samples'])
    
    # Load in the data for sampling
    if args.data_file:
        with open(args.data_file) as f:
            data = json.load(f)
        for sid in data:
            sid_dir = os.path.join(DATA_DIR, qid, 'students', sid)
            if not os.path.isdir(sid_dir):
                os.mkdir(sid_dir)
                with open(os.path.join(sid_dir, 'response.txt'), 'w') as f:
                    f.write(data[sid])
        sids = [sid for sid in data]
    else:
        sids = os.listdir(os.path.join(DATA_DIR, qid, 'students'))
        
        
    # single-grade: grades a single program specified by an input sid.
    # batch-grade: grades all programs within a given sid: program mapping, or
    # grades all programs within the 'students' directory. 
    if mode == 'single-grade' or mode == 'batch-grade':
        results_path = os.path.join(DATA_DIR, qid, 'results', scorer_name, str(frr))
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        
        if os.path.isdir(os.path.join(results_path, 'labels.json')):  
            with open(os.path.join(results_path, 'labels.json')) as f:
                labels = json.load(f)
        else:
            labels = {}
        if os.path.isdir(os.path.join(results_path, 'scores.json')):
            with open(os.path.join(results_path, 'scores.json')) as f:
                scores = json.load(f)
        else:
            scores = {}
        if os.path.isdir(os.path.join(results_path, 'samples_needed.json')):
            with open(os.path.join(results_path, 'samples_needed.json')) as f:
                samples_needed = json.load(f)
        else:
            samples_needed = {}
        if os.path.isdir(os.path.join(results_path, 'runtimes.json')):
            with open(os.path.join(results_path, 'runtimes.json')) as f:
                runtimes = json.load(f)
        else:
            runtimes = {}
            
        algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, num_soln_samples)
        if mode == 'single-grade':
            sids_to_grade = [single_grade_sid]
        else:
            sids_to_grade = sids
        
        for i in tqdm(range(len(sids_to_grade))):
            sid = sids_to_grade[i]
            label, score, need, run = algorithm.grade(sid, case=case)
            labels[sid] = label
            scores[sid] = score
            samples_needed[sid] = need
            runtimes[sid] = run
            
            if i % 5 == 0:
                with open(os.path.join(results_path, 'labels.json'), 'w') as f:
                    json.dump(labels, f)
                with open(os.path.join(results_path, 'scores.json'), 'w') as f:
                    json.dump(scores, f)
                with open(os.path.join(results_path, 'samples_needed.json'), 'w') as f:
                    json.dump(samples_needed, f)
                with open(os.path.join(results_path, 'runtimes.json'), 'w') as f:
                    json.dump(runtimes, f)
                    
        with open(os.path.join(results_path, 'labels.json'), 'w') as f:
            json.dump(labels, f)
        with open(os.path.join(results_path, 'scores.json'), 'w') as f:
            json.dump(scores, f)
        with open(os.path.join(results_path, 'samples_needed.json'), 'w') as f:
            json.dump(samples_needed, f)
        with open(os.path.join(results_path, 'runtimes.json'), 'w') as f:
            json.dump(runtimes, f)
    
    
    # cluster: cluster programs based on their scores. 
    if mode == 'cluster':
        score_list = []
        results_path = os.path.join(DATA_DIR, qid, 'results', scorer_name, str(frr))
        for scorer in cluster_scorers:
            with open(os.path.join(results_path, 'scores.json')) as f:
                scores_dict = json.load(f)
            scores = [scores_dict[sid] for sid in scores_dict]
            score_list.append(scores)
        print(score_list)
            
        if cluster_method == 'agglomerative':
            cluster_labels = agglomerative(score_list, n_clusters = n_clusters)    
        else:
            cluster_labels = mean_shift(score_list, bandwidth=bandwidth)
        
        with open(os.path.join(results_path, 'clusters.json')) as f:
            json.dump(cluster_labels, f)
            

    # hand-label: label the programs by hand
    if mode == 'hand-label':
        pass
        
    
    # monte-carlo: perform Monte-Carlo sampling and precompute scores for 
    # the rejection threshold.
    if mode == 'monte-carlo':
        algorithm = StochasticGrade(qid, scorer, sample_sizes, frr, dtype, num_soln_samples)
        num_samples = mc_num_anchor_samples if mc_num_anchor_samples > max(sample_sizes) else max(sample_sizes)
        print(f'Sampling {num_samples} samples from the correct distribution...')
        algorithm.monte_carlo(num_samples, M=M, overwrite=overwrite)
            
        # Precompute scores
        monte_carlo_dir = os.path.join(DATA_DIR, qid, 'solution', 'mc_solutions')
        scores = {}
        print('Computing scores...')
        for sid in tqdm(os.listdir(monte_carlo_dir)):
            samples = np.load(os.path.join(monte_carlo_dir, sid, 'samples.npy'), allow_pickle=True)
            for sample_size in sample_sizes:
                if sample_size not in scores:
                    scores[sample_size] = []
                score = algorithm.scorer.compute_score(
                    samples[:sample_size], algorithm.soln_samples[:mc_num_anchor_samples]
                )
                scores[sample_size].append(score)
            if mc_num_samples not in scores:
                scores[mc_num_samples] = []
            score = algorithm.scorer.compute_score(
                samples[:mc_num_samples], algorithm.soln_samples[:mc_num_anchor_samples]
            )
            scores[mc_num_samples].append(score)
                
        results_path = os.path.join(DATA_DIR, qid, 'results', scorer_name)
        scores_path = os.path.join(results_path, 'monte_carlo_scores.json')
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(scores_path, 'w') as f:
            json.dump(scores, f)
                
