"""
╔═════════════════════════════════════════════════════╗
║                openswath_scoring.py                 ║
╠═════════════════════════════════════════════════════╣
║       Description: Caculate OpenSWATH scores        ║
╠═════════════════════════════════════════════════════╣
║                 Author: Mingxuan Gao                ║
║         Contact: mingxuan.gao@utoronto.ca           ║
╚═════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.special import erf

def normalize_sum(x):
    sum_x = sum(x)
    if sum_x == 0.0:
        return x
    inverse_sum = 1 / sum_x
    return [i * inverse_sum for i in x]

def xcorr_score(chrom1, chrom2):
    n = len(chrom1)
    mean_chrom1 = np.mean(chrom1)
    mean_chrom2 = np.mean(chrom2)

    crosscorr = [(chrom1[i] - mean_chrom1) * (chrom2[i] - mean_chrom2) for i in range(n)]

    sum_num = sum(crosscorr)
    sum_den1 = sum((chrom1[i] - mean_chrom1) ** 2 for i in range(n))
    sum_den2 = sum((chrom2[i] - mean_chrom2) ** 2 for i in range(n))

    denominator = np.sqrt(sum_den1 * sum_den2)
    return sum_num / denominator if denominator != 0 else 0

def xcorr_shape_score(chrom1, chrom2, max_lag=10):
    n = len(chrom1)
    mean_chrom1 = np.mean(chrom1)
    mean_chrom2 = np.mean(chrom2)
    max_score = -np.inf
    
    for lag in range(-max_lag, max_lag + 1):
        score = 0.0
        count = 0
        
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                score += (chrom1[i] - mean_chrom1) * (chrom2[j] - mean_chrom2)
                count += 1
        
        if count > 0:
            score /= count
            max_score = max(max_score, score)
    
    return max_score

def calculate_xcorr_scores(matrix):
    num_chromatograms = len(matrix)
    mean_xcorr_scores = np.zeros(num_chromatograms)
    mean_xcorr_shape_scores = np.zeros(num_chromatograms)
    
    for i in range(num_chromatograms):
        xcorr_scores = []
        xcorr_shape_scores = []
        for j in range(num_chromatograms):
            if i != j:
                xcorr_scores.append(xcorr_score(matrix[i], matrix[j]))
                xcorr_shape_scores.append(xcorr_shape_score(matrix[i], matrix[j]))
        
        mean_xcorr_scores[i] = np.mean(xcorr_scores) if xcorr_scores else 0
        mean_xcorr_shape_scores[i] = np.mean(xcorr_shape_scores) if xcorr_shape_scores else 0
    
    return mean_xcorr_scores, mean_xcorr_shape_scores

def emg_score(chromatogram, mu = 2.0, sigma = 1.0, lambda_ = 0.5):
    score = 0.0
    for i in range(len(chromatogram)):
        t = float(i)
        exp_term = np.exp((lambda_ / 2) * (2 * mu + lambda_ * sigma ** 2 - 2 * t))
        erf_term = erf((mu + lambda_ * sigma ** 2 - t) / (np.sqrt(2) * sigma))
        model_value = (lambda_ / 2) * exp_term * (1 + erf_term)
        score += (chromatogram[i] - model_value) ** 2
    return score

def calculate_emg_scores(matrix, mu = 2.0, sigma = 1.0, lambda_ = 0.5):
    num_chromatograms = matrix.shape[0]
    emg_scores = np.zeros(num_chromatograms)
    
    for i in range(num_chromatograms):
        emg_scores[i] = emg_score(matrix[i], mu, sigma, lambda_)
    
    return emg_scores