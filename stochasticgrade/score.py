import inspect
import json
import numpy as np
import os
import pickle
import scipy
import scipy.stats as stats
import time
from abc import ABC, abstractmethod
from stochasticgrade.constants import *
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
    
    
class SkeletonScorer(Scorer):
    """
    Allows for the definition of a new scoring method.
    name: the name of the scorer as a string
    score_fn: a function accepting stud_samples and soln_samples
    rejection_threshold_fn: a function that calculates the rejection threshold
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

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        if len(stud_samples.shape) == 1:
            samples = np.array([soln_samples.flatten(), stud_samples.flatten()])
            np.random.shuffle(samples)
            score = stats.anderson_ksamp(samples).statistic
        else:
            stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid)
            dists = np.array([stud_dists.flatten(), soln_dists.flatten()])
            np.random.shuffle(dists)
            score = stats.anderson_ksamp(dists).statistic
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        frr_string = np.format_float_positional(frr, trim='-')
        table = {
            '0.000046875': 8.726,
            '0.00009375': 8.150,
            '0.00015625': 7.695,
            '0.0001875': 7.528,
            '0.0003125': 7.054,
            '0.000375': 6.884,
            '0.000625': 6.405,
            '0.00075': 6.235,
            '0.00078125': 6.197,
            '0.00125': 5.760,
            '0.0015': 5.592,
            '0.0015625': 5.555,
            '0.0025': 5.124,
            '0.003': 4.958,
            '0.003125': 4.921,
            '0.005': 4.497,
            '0.00625': 4.297,
            '0.01': 3.878,
            '0.0125': 3.681,
            '0.025': 3.077,
            '0.05': 2.492,
            '0.1': 1.933
        }
        
        if frr_string in table:
            threshold = table[frr_string]
        else:
            print('The provided FRR is not accepted.')
            print('For the AndersonDarlingScorer, you must select 0.003, 0.01, 0.05, or 0.1.')
            raise Exception
        return threshold
    
    
class CovarianceScorer(Scorer):
    """
    Covariance-based scoring method.
    Measures the difference between covariance matrices of the
    student and solution samples using the Frobenius norm.
    Used for multidimensional data.
    """
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return 'CovarianceScorer'
    
    def compute_score(self, stud_samples, soln_samples, sid, qid):
        student_cov = np.cov(stud_samples.T)
        score = np.sum(np.mean(stud_samples, axis=0) ** 2) + np.sum(student_cov)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 
    
    
class MahalanobisScorer(Scorer):
    """
    Squared Mahalanobis distance scoring method.
    """
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return 'MahalanobisScorer'
    
    def compute_score(self, stud_samples, soln_samples, sid, qid):
        try:
            mean_diff = np.mean(stud_samples - np.mean(soln_samples, axis=0), axis=0)
            if len(stud_samples.shape) > 1:   
                cov = np.linalg.inv(np.cov(stud_samples.T))
                score = mean_diff.T @ cov @ mean_diff
            else:
                cov = 1 / np.std(stud_samples) ** 2
                score = mean_diff * cov * mean_diff
        except:
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 
    
    
class MahalanobisScorerV2(Scorer):
    """
    Squared Mahalanobis distance scoring method.
    Measuring the squared average distance of the student sample points to 
    the solution sample distribution. Uses a two-tailed rejection threshold.
    """
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return 'MahalanobisScorerV2'
    
    def compute_score(self, stud_samples, soln_samples, sid, qid):
        try:
            mean_diff = stud_samples - np.mean(soln_samples, axis=0)
            if len(stud_samples.shape) > 1:   
                cov = np.linalg.inv(np.cov(soln_samples.T))
                score = np.mean((mean_diff @ cov) * mean_diff)
            else:
                cov = 1 / np.std(soln_samples) ** 2
                score = np.mean(mean_diff * cov * mean_diff)
        except:
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        lower_index = int(len(scores) * (1 - frr / 2))
        upper_index = int(len(scores) * frr / 2)
        threshold = {'lower_threshold': scores[lower_index], 'upper_threshold': scores[upper_index]}
        return threshold 
    
    
class MeanScorer(Scorer):
    """
    Mean-based scoring method.
    Measures the difference between the student and solution sample means
    by the t-test. Used with one-dimensional data.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'MeanScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        if len(stud_samples.shape) == 1:  # 1-D
            ttest = stats.ttest_ind(stud_samples, soln_samples)
            score = 1 - ttest.pvalue
        else:
            stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid)
            ttest = stats.ttest_ind(stud_dists, soln_dists)
            score = 1 - ttest.pvalue
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        return 1 - frr
    
    
class MeanClusterScorer(Scorer):
    """
    Mean-based scoring method used only for clustering.
    Measures the difference between the student and solution sample means. 
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'MeanClusterScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        score = np.mean(stud_samples) - np.mean(soln_samples)
        if not np.isfinite(score):
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 
    
    
class TClusterScorer(Scorer):
    """
    T-test-based scoring method used only for clustering.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'TClusterScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        if len(stud_samples.shape) == 1:  # 1-D
            ttest = stats.ttest_ind(stud_samples, soln_samples)
            score = ttest.statistic
        else:
            stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid)
            ttest = stats.ttest_ind(stud_dists, soln_dists)
            score = ttest.statistic
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        df = num_stud_samples + num_soln_samples - 2
        t_stat = scipy.special.stdtrit(df, 1 - frr)
        threshold = {'lower_threshold': -t_stat, 'upper_threshold': t_stat}
        return threshold 
    
    
class MSDScorer(Scorer):
    """
    MSD-based scoring method.
    Measures the difference in spread between the student and
    solution samples.
    """
    def __init__(self):
        super().__init__()
        self.soln_var_term = None

    def __str__(self):
        return 'MSDScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        if len(stud_samples.shape) == 1:  # 1-D
            mean_term = np.linalg.norm(np.mean(stud_samples, axis=0) - np.mean(soln_samples, axis=0)) ** 2
            stud_var_term = np.std(stud_samples) ** 2
            if self.soln_var_term is None:
                self.soln_var_term = np.std(soln_samples) ** 2
        else:
            stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid)
            mean_term = np.linalg.norm(np.mean(stud_dists, axis=0) - np.mean(soln_dists, axis=0)) ** 2
            stud_var_term = np.std(stud_dists) ** 2
            if self.soln_var_term is None:
                self.soln_var_term = np.std(soln_dists) ** 2
        score = abs(mean_term + stud_var_term - self.soln_var_term)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold    
    
    
class MultiDimMeanScorer(Scorer):
    """
    Multi-dimensional mean-based scoring method.
    Measures the difference between the student and solution sample means
    by the t-test. Used with multi-dimensional data.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'MultiDimMeanScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        ttest = stats.ttest_ind(stud_samples, soln_samples)
        score = np.sum(1 - ttest.pvalue)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 
    
    
class OldAndersonDarlingScorer(Scorer):
    """
    Old Anderson-Darling p-value based scoring method.
    Calculates the Anderson-Darling test statistic between the
    student and solution samples.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'OldAndersonDarlingScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        samples = np.array([soln_samples.flatten(), stud_samples.flatten()])
        np.random.shuffle(samples)
        score = 1 - stats.anderson_ksamp(samples).pvalue
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        return 1 - frr
    
    
class OldMSDScorer(Scorer):
    """
    MSD-based scoring method.
    Measures the difference in spread between the student and
    solution samples.
    """
    def __init__(self):
        super().__init__()
        self.soln_var_term = None

    def __str__(self):
        return 'MSDScorer'

    def compute_score(self, stud_samples, soln_samples, sid, qid):
        mean_term = np.linalg.norm(np.mean(stud_samples, axis=0) - np.mean(soln_samples, axis=0)) ** 2
        stud_var_term = np.trace(np.cov(stud_samples.T)) if len(stud_samples.shape) > 1 else np.std(stud_samples) ** 2
        if self.soln_var_term is None:
            if len(soln_samples.shape) > 1:
                self.soln_var_term = np.trace(np.cov(soln_samples.T)) 
            else:
                self.soln_var_term = np.std(soln_samples) ** 2
        score = abs(mean_term + stud_var_term - self.soln_var_term)
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
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
    
    def compute_score(self, stud_samples, soln_samples, sid, qid):
        if len(stud_samples.shape) == 1:  # 1-D
            score = stats.wasserstein_distance(stud_samples.flatten(), soln_samples.flatten())
        else:
            stud_dists = get_euclidean_distances(stud_samples, soln_samples, sid, qid)
            soln_dists = get_euclidean_distances(soln_samples, soln_samples, sid, qid)
            score = stats.wasserstein_distance(stud_dists.flatten(), soln_dists.flatten())
        if not np.isfinite(score):
            score = 1e7
        return score
    
    def rejection_threshold(self, frr, num_stud_samples, num_soln_samples, monte_carlo_path=None):
        if not os.path.isfile(monte_carlo_path):
            print('\nERROR: Must perform Monte Carlo sampling!\n')
            raise Exception
        with open(monte_carlo_path) as f:
            scores = json.load(f)
            if str(num_stud_samples) not in scores.keys():
                print('\nERROR: Must perform Monte Carlo sampling! (Uncomputed sample size)\n')
                raise Exception
        scores = scores[str(num_stud_samples)]
        scores.sort()
        scores.reverse()
        index = int(len(scores) * frr)
        threshold = scores[index]
        return threshold 


def make_scorer_map():
    import __main__
    __name__ = "__main__"
    scorer_map = {}

    # Get all classes defined in this module
    classes = inspect.getmembers(__import__(__name__), inspect.isclass)

    for name, cls in classes:
        if issubclass(cls, Scorer) and cls is not Scorer:
            scorer_map[cls.__name__] = cls()

    return scorer_map


def get_euclidean_distances(stud_samples, soln_samples, sid, qid, dilation=1):
    """
    Measures the Euclidean distance between a randomly selected 
    anchor point (near the solution distribution) and the student samples.
    """
    anchor_path = os.path.join(DATA_DIR, qid, 'results', 'random_anchor.json')
    
    # Sample the random anchor point from a bounding box of the solution samples
    if not os.path.isfile(anchor_path):
        dirname = '/'.join(anchor_path.split('/')[:-1])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        minima = np.amin(soln_samples, axis=0)
        maxima = np.amax(soln_samples, axis=0)
        midpoints = (minima + maxima) / 2
        adjusted_minima = midpoints - (midpoints - minima) * (1 + dilation)
        adjusted_maxima = midpoints + (maxima - midpoints) * (1 + dilation)
        
        sample_anchor = True
        while sample_anchor:
            anchor = [np.random.uniform(x[0], x[1]) for x in zip(adjusted_minima, adjusted_maxima)]
            for i in range(len(anchor)):
                if anchor[i] < minima[i] or anchor[i] > maxima[i]:
                    sample_anchor = False
                    
        with open(os.path.join(anchor_path), 'w') as f:
            json.dump(anchor, f)
        soln_dists = np.sqrt(np.sum((soln_samples - anchor) * (soln_samples - anchor), axis=1)) 
        path = os.path.join(DATA_DIR, qid, 'solution', 'solution', 'euclidean_dists.npy')
        np.save(path, soln_dists)
        
    with open(os.path.join(anchor_path)) as f:
        anchor = json.load(f)
            
    # Compute the distances of student samples points to the anchor point
    dists = np.sqrt(np.sum((stud_samples - anchor) * (stud_samples - anchor), axis=1))
    
    if 'solution' in sid:
        id_type = 'solution/mc_solutions' if 'mc_solution' in sid else 'solution'
    else:
        id_type = 'students'
    path = os.path.join(DATA_DIR, qid, id_type, sid, 'euclidean_dists.npy')
    np.save(path, dists)
    
    return dists


def get_projection_distances(stud_samples, soln_samples, sid, qid):
    """
    Calculates the projection of the student samples onto a unit vector
    from the corresponding n-dimensional hypersphere. 
    """
    vector_path = os.path.join(DATA_DIR, qid, 'results', 'random_unit_vector.json')
    
    # Sample the random unit vector from the hypersphere
    if not os.path.isfile(vector_path):
        dirname = '/'.join(vector_path.split('/')[:-1])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        n = soln_samples.shape[1:]
        point = np.random.normal(size=n)
        norm = np.linalg.norm(point)
        unit_vector = point / norm

        with open(os.path.join(vector_path), 'w') as f:
            json.dump(unit_vector.tolist(), f)
        soln_dists = np.dot(soln_samples, unit_vector)
        path = os.path.join(DATA_DIR, qid, 'solution', 'solution', 'projection_dists.npy')
        np.save(path, soln_dists)
        
    with open(os.path.join(vector_path)) as f:
        unit_vector = json.load(f)
        
    # Compute the distances of student samples points to the anchor point
    dists = np.dot(stud_samples, unit_vector)
    
    if 'solution' in sid:
        id_type = 'solution/mc_solutions' if 'mc_solution' in sid else 'solution'
    else:
        id_type = 'students'
    path = os.path.join(DATA_DIR, qid, id_type, sid, 'projection_dists.npy')
    np.save(path, dists)
    
    return dists