import inspect
import json
import numpy as np
import os
import pickle
import scipy.stats as stats
import time
from abc import ABC, abstractmethod
from constants import *
from copy import deepcopy
from tqdm import tqdm


class Scorer(ABC):
    """
    Scores are only based on the samples being evaluated.
    They are independent of scores assigned to other samples.

    Inheriting classes should implement the following functions:
        - __str__
        - compute_score
    """
    def __init__(self):
        self.score_list = None
        self.single_num_samples = None
        # num_samples can be useful when adjusting sample sizes for scoring
        self.single_score_dict = None
        self.runtime = None
        
    @abstractmethod
    def __str__(self):
        """
        Returns:
            Name of the class string (to be used for out_dirname)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_score(self, samples):
        """
        Returns:
            score: the calculated score (double)
        """
        raise NotImplementedError
        
    def get_score_list(self, qid, sid_list=None, overwrite=False):
        """
        Compute scores for all submissions (or submissions specified by sid_list) 
        belonging to a question (qid)
        """
        
        score_list_path = os.path.join(DATA_DIR, qid, 'scores', str(self), 'score_list.pkl')
        score_dict_path = os.path.join(DATA_DIR, qid, 'scores', str(self), 'score_dict.pkl')
        
        if os.path.isfile(score_list_path) and not overwrite:
            with open(score_list_path, 'rb') as f:
                score_list = pickle.load(f)
        else:
            sids = sid_list
            if sid_list is None:
                sids = os.listdir(DATA_DIR, qid, 'students')
            score_list, score_dict = [], {}
            soln_sample_path = os.path.join(DATA_DIR, qid, 'solution/samples.pkl')
            soln_samples = pickle.load(soln_sample_path)
            
            # Compute the scores for each student
            for sid in sids: 
                stud_sample_path = os.path.join(DATA_DIR, qid, 'students', sid, 'samples.pkl')
                stud_samples = pickle.load(stud_sample_path)
                score = self.compute_score(stud_samples, soln_samples)
                score_list.append(score)
                score_dict[sid] = score
            
            # Save the scores
            with open(score_list_path, 'wb') as f:
                pickle.dump(score_list, f)
            with open(score_dict_path, 'wb') as f:
                pickle.dump(score_dict, f)
    
        return score_list
    
    
class SkeletonScorer(Scorer):
    """
    Allows for the definition of a new scoring method.
    name: the name of the scorer as a string
    score_fn: a function accepting stud_samples and soln_samples
    """
    def __init__(self, name='', score_fn=None, rejection_threshold_fn=None):
        super().__init__()
        self.name = name
        self.score_fn = score_fn
        self.rejection_threshold_fn = rejection_threshold_fn

    def __str__(self):
        return self.name

    def compute_score(self, stud_samples, soln_samples):
        return self.score_fn(stud_samples, soln_samples)
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        return self.rejection_threshold_fn(frr, num_stud_samples, num_soln_samples, monte_carlo_path=None)
        

class AndersonDarlingScorer(Scorer):
    """
    Anderson-Darling statistic based scoring method.
    Calculates the Anderson-Darling test statistic between the
    student and solution samples.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'AndersonDarlingScorer'

    def compute_score(self, stud_samples, soln_samples):
        score = 1 - stats.anderson_ksamp([soln_samples, stud_samples]).pvalue
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        return 1 - frr
    
    
class MeanScorer(Scorer):
    """
    Mean-based scoring method.
    Measures the difference between the student and solution means.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'MeanScorer'

    def compute_score(self, stud_samples, soln_samples):
        student_mean = np.mean(stud_samples)
        solution_mean = np.mean(soln_samples)
        score = abs(student_mean - solution_mean)
        if not np.isfinite(score):
            score = 1e7 
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        threshold = stats.norm.ppf(1 - frr / 2)
        return threshold
    
    
class MSDScorer(Scorer):
    """
    MSD-based scoring method.
    Measures the difference in spread between the student and
    solution samples.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'MSDScorer'

    def compute_score(self, stud_samples, soln_samples):
        score = (np.mean(stud_samples) - np.mean(soln_samples)) ** 2 + np.var(stud_samples)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
        scores = scores[str(num_stud_samples)]
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 
    

class WassersteinScorer(Scorer):
    """
    Wasserstein distance based scoring method.
    Calculates the Wasserstein distance between the student and 
    solution samples.
    """
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return 'WassersteinScorer'
    
    def compute_score(self, stud_samples, soln_samples):
        score = stats.wasserstein_distance(stud_samples, soln_samples)
        if not np.isfinite(score):
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
        scores = scores[str(num_stud_samples)]
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 


def make_scorer_map():
    scorer_map = {}

    # Get all classes defined in this module
    classes = inspect.getmembers(__import__(__name__), inspect.isclass)

    for name, cls in classes:
        if issubclass(cls, Scorer) and cls is not Scorer:
            scorer_map[cls.__name__] = cls()

    return scorer_map