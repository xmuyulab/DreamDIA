"""
╔═════════════════════════════════════════════════════╗
║               statistical_analysis.py               ║
╠═════════════════════════════════════════════════════╣
║     Description: Utility functions for statistical  ║
║                      analysis                       ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os
import random

import click
import scipy as sp
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.kde import KDEUnivariate

from utils import get_precursor_indices, flatten_list

def sample_scoring_table(scoring_table, sample_rate, random_seed):
    """
    Samples a given scoring table at a specified rate using a random seed for reproducibility.
    
    Parameters:
    scoring_table (pd.DataFrame): A DataFrame containing the scoring table data, which includes a column 'transition_group_id'.
    sample_rate (float): The rate at which to sample the scoring table, expressed as a fraction (e.g., 0.1 for 10%).
    random_seed (int): The seed for the random number generator to ensure reproducibility.
    
    Returns:
    pd.DataFrame: A DataFrame containing the sampled rows from the original scoring table.
    
    Notes:
    - The function relies on a `get_precursor_indices` function to get the indices of precursor rows based on the 'transition_group_id' column.
    - The function also relies on a `flatten_list` function to flatten the sampled indices.
    """

    random.seed(random_seed)
    precursor_indices = get_precursor_indices(scoring_table["transition_group_id"])
    n_samples = int(len(precursor_indices) * sample_rate)
    samples = flatten_list(random.sample(precursor_indices, n_samples))
    
    return scoring_table.iloc[samples, :]

def pemp(stat, stat0):
    """ 
    Computes empirical p-values identically to the bioconductor/qvalue empPvals.
    
    Parameters:
    stat (list or array-like): The test statistics for which p-values are to be computed.
    stat0 (list or array-like): The null distribution statistics.
    
    Returns:
    numpy.ndarray: An array of empirical p-values corresponding to the input test statistics.
    """

    assert len(stat0) > 0
    assert len(stat) > 0

    stat = np.array(stat)
    stat0 = np.array(stat0)

    m = len(stat)
    m0 = len(stat0)

    statc = np.concatenate((stat, stat0))
    v = np.array([True] * m + [False] * m0)
    perm = np.argsort(-statc, kind="mergesort")  # reversed sort, mergesort is stable
    v = v[perm]

    u = np.where(v)[0]
    p = (u - np.arange(m)) / float(m0)

    # ranks can be fractional, we round down to the next integer, ranking returns values starting
    # with 1, not 0:
    ranks = np.floor(sp.stats.rankdata(-stat)).astype(int) - 1
    p = p[ranks]
    p[p <= 1.0 / m0] = 1.0 / m0

    return p

def pi0est(p_values, lambda_ = np.arange(0.05,1.0,0.05), pi0_method = "smoother", smooth_df = 3, smooth_log_pi0 = False):
    """ Estimate pi0 according to bioconductor/qvalue """

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    p = np.array(p_values)

    rm_na = np.isfinite(p)
    p = p[rm_na]
    m = len(p)
    ll = 1
    if isinstance(lambda_, np.ndarray ):
        ll = len(lambda_)
        lambda_ = np.sort(lambda_)

    if (min(p) < 0 or max(p) > 1):
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (ll > 1 and ll < 4):
        raise click.ClickException("If lambda_ is not predefined (one value), at least four data points are required.")
    elif (np.min(lambda_) < 0 or np.max(lambda_) >= 1):
        raise click.ClickException("Lambda must be within [0,1)")

    if (ll == 1):
        pi0 = np.mean(p >= lambda_)/(1 - lambda_)
        pi0_lambda = pi0
        pi0 = np.minimum(pi0, 1)
        pi0Smooth = False
    else:
        pi0 = []
        for l in lambda_:
            pi0.append(np.mean(p >= l)/(1 - l))
        pi0_lambda = pi0

        if (pi0_method == "smoother"):
            if smooth_log_pi0:
                pi0 = np.log(pi0)
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = np.exp(spi0(lambda_))
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = np.exp(predict(spi0, x = lambda_).rx2('y')) # R reference function
            else:
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = spi0(lambda_)
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = predict(spi0, x = lambda_).rx2('y')  # R reference function
            pi0 = np.minimum(pi0Smooth[ll-1],1)
        elif (pi0_method == "bootstrap"):
            minpi0 = np.percentile(pi0,0.1)
            W = []
            for l in lambda_:
                W.append(np.sum(p >= l))
            mse = (np.array(W) / (np.power(m,2) * np.power((1 - lambda_),2))) * (1 - np.array(W) / m) + np.power((pi0 - minpi0),2)
            pi0 = np.minimum(pi0[np.argmin(mse)],1)
            pi0Smooth = False
        else:
            raise click.ClickException("pi0_method must be one of 'smoother' or 'bootstrap'.")
    if (pi0<=0):
        #plot_hist(p, f"p-value density histogram used during pi0 estimation", "p-value", "density histogram", "pi0_estimation_error_pvalue_histogram_plot.pdf")
        raise click.ClickException(f"The estimated pi0 <= 0. Check that you have valid p-values or use a different range of lambda. Current lambda range: {lambda_}")

    return {'pi0': pi0, 'pi0_lambda': pi0_lambda, 'lambda_': lambda_, 'pi0_smooth': pi0Smooth}

def qvalue(p_values, pi0, pfdr = False):
    p = np.array(p_values)

    qvals_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise click.ClickException("pi0 not in valid range [0,1].")

    m = len(p)
    u = np.argsort(p)
    v = sp.stats.rankdata(p,"max")

    if pfdr:
        qvals = (pi0 * m * p) / (v * (1 - np.power((1 - p), m)))
    else:
        qvals = (pi0 * m * p) / v
    
    qvals[u[m-1]] = np.minimum(qvals[u[m-1]], 1)
    for i in list(reversed(range(0,m-2,1))):
        qvals[u[i]] = np.minimum(qvals[u[i]], qvals[u[i + 1]])

    qvals_out[rm_na] = qvals
    return qvals_out

def bw_nrd0(x):
    if len(x) < 2:
        raise click.ClickException("bandwidth estimation requires at least two data points.")

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)
    lo = lo or hi or abs(x[0]) or 1

    return 0.9 * lo *len(x)**-0.2

def lfdr(p_values, pi0, trunc = True, monotone = True, transf = "probit", adj = 1.5, eps = np.power(10.0,-8)):
    """ Estimate local FDR / posterior error probability from p-values according to bioconductor/qvalue """
    p = np.array(p_values)

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # density=robjects.r('density')
    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    # Check inputs
    lfdr_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise click.ClickException("pi0 not in valid range [0,1].")

    # Local FDR method for both probit and logit transformations
    if (transf == "probit"):
        p = np.maximum(p, eps)
        p = np.minimum(p, 1-eps)
        x = sp.stats.norm.ppf(p, loc=0, scale=1)

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)
        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        lfdr = pi0 * sp.stats.norm.pdf(x) / y
    elif (transf == "logit"):
        x = np.log((p + eps) / (1 - p + eps))

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)

        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        dx = np.exp(x) / np.power((1 + np.exp(x)),2)
        lfdr = (pi0 * dx) / y
    else:
        raise click.ClickException("Invalid local FDR method.")

    if (trunc):
        lfdr[lfdr > 1] = 1
    if (monotone):
        lfdr = lfdr[p.ravel().argsort()]
        for i in range(1,len(x)):
            if (lfdr[i] < lfdr[i - 1]):
                lfdr[i] = lfdr[i - 1]
        lfdr = lfdr[sp.stats.rankdata(p,"min")-1]

    lfdr_out[rm_na] = lfdr
    return lfdr_out

def stats(scoring_table, epsilon):
    scoring_table["norm_dr_score"] = scoring_table["dr_score"].apply(lambda x : np.log((x + epsilon) / (1 - x + epsilon)))
    scoring_table_decoy = scoring_table[scoring_table["decoy"] == 1]
    stat = scoring_table["norm_dr_score"].values
    stat0 = scoring_table_decoy["norm_dr_score"].values
    p = pemp(stat, stat0)
    scoring_table["pvalue"] = p

    pi0 = pi0est(scoring_table["pvalue"].values)
    qvalues = qvalue(scoring_table["pvalue"].values, pi0["pi0"])
    pep = lfdr(scoring_table["pvalue"].values, pi0["pi0"])
    scoring_table["qvalue"] = qvalues
    scoring_table["pep"] = pep
    
    return scoring_table

def dream_stats(scoring_table_list, output_dir, n_threads, logger, 
                all_result_name, precursor_result_name, peptide_result_name, protein_result_name, 
                fdr_precursor = 0.01, fdr_peptide = 0.01, fdr_protein = 0.01, 
                disc_model = "xgboost", sample_rate = None, random_seed = 6258721, tag = "dreamdialignr"):

    cutoff_epochs = 10
    cutoff_delta_mse = 3e-4
    init_ranking_score = "SCORE_lib_cosine"
    heuristic_depth_xgboost = 6
    heuristic_depth_rf = 12
    heuristic_n_estimators = 200
    epsilon = 1e-8
        
    if sample_rate is None:
        scoring_table = pd.concat(scoring_table_list)   
    else:
        scoring_table = pd.concat([sample_scoring_table(df, sample_rate, random_seed) for df in scoring_table_list])
    scoring_table.reset_index(drop = True, inplace = True)

    if disc_model == "xgboost":
        cl = XGBClassifier(n_jobs = n_threads, max_depth = heuristic_depth_xgboost, random_state = random_seed)
    else:
        cl = RandomForestClassifier(n_estimators = heuristic_n_estimators, max_depth = heuristic_depth_rf, n_jobs = n_threads, random_state = random_seed)

    score_columns = [i for i in list(scoring_table.columns) if (i.startswith("SCORE"))]
    if tag == "dreamdialignr":
        score_columns.remove("SCORE_RT_mean")
        score_columns.remove("SCORE_RT_std")
        score_columns.remove("SCORE_deltaRT_mean")
        score_columns.remove("SCORE_deltaRT_std")
    if "SCORE_drf0" in score_columns:
        score_columns = [i for i in list(scoring_table.columns) if (i.startswith("SCORE") and i != "SCORE_DREAM")]
    
    scoring_table["precursor_run_id"] = scoring_table["transition_group_id"] + "_" + scoring_table["filename"]
    
    scoring_table_target = scoring_table[scoring_table["decoy"] == 0]
    scoring_table_decoy = scoring_table[scoring_table["decoy"] == 1]
    scoring_table_target = scoring_table_target.sort_values(by = ["precursor_run_id", init_ranking_score], ascending = [True, False])
    scoring_table_best = scoring_table_target.drop_duplicates(subset = "precursor_run_id", keep = "first")
    scoring_table_best = pd.concat([scoring_table_best, scoring_table_decoy])
    scoring_table_best.reset_index(inplace = True, drop = True)   
    
    cl.fit(scoring_table_best.loc[:, score_columns].values, scoring_table_best["decoy"])
    scoring_table["dr_score0"] = cl.predict_proba(scoring_table.loc[:, score_columns].values)[:, 0]
    
    last_mse = 0
    last_cl = cl
    for epoch in range(cutoff_epochs):
        scoring_table_target = scoring_table[scoring_table["decoy"] == 0]
        scoring_table_decoy = scoring_table[scoring_table["decoy"] == 1]
        scoring_table_target = scoring_table_target.sort_values(by = ["precursor_run_id", "dr_score%d" % epoch], ascending = [True, False])
        scoring_table_best = scoring_table_target.drop_duplicates(subset = "precursor_run_id", keep = "first")
        scoring_table_best = pd.concat([scoring_table_best, scoring_table_decoy])
        scoring_table_best.reset_index(inplace = True, drop = True)
        
        if disc_model == "xgboost":
            cl = XGBClassifier(n_jobs = n_threads, max_depth = heuristic_depth_xgboost, random_state = random_seed)
        else:
            cl = RandomForestClassifier(n_estimators = heuristic_n_estimators, max_depth = heuristic_depth_rf, n_jobs = n_threads, random_state = random_seed)
        
        cl.fit(scoring_table_best.loc[:, score_columns].values, scoring_table_best["decoy"])
        prediction = cl.predict_proba(scoring_table.loc[:, score_columns].values)[:, 0]   
        mse = mean_squared_error(prediction, scoring_table["dr_score%d" % (epoch)]) 
        delta_mse = abs(last_mse - mse)
        logger.info("Semi-supervised model prediction error (epoch %d): %s" % (epoch + 1, delta_mse))
        
        if delta_mse < cutoff_delta_mse:
            break
        else:
            scoring_table["dr_score%d" % (epoch + 1)] = prediction
            last_mse = mse
            last_cl = cl
    
    scoring_table = pd.concat(scoring_table_list)
    scoring_table.reset_index(drop = True, inplace = True)
    scoring_table["precursor_run_id"] = scoring_table["transition_group_id"] + "_" + scoring_table["filename"]
    scoring_table["dr_score"] = last_cl.predict_proba(scoring_table.loc[:, score_columns].values)[:, 0] 
    scoring_table = stats(scoring_table, epsilon)
        
    scoring_table = scoring_table.sort_values(by = ["precursor_run_id", "dr_score"], ascending = [True, False])
    scoring_table.reset_index(drop = True, inplace = True)
    precursor_indice = get_precursor_indices(scoring_table["precursor_run_id"])
    peak_group_ranks = flatten_list([list(range(1, len(i) + 1)) for i in precursor_indice])
    scoring_table["peak_group_rank"] = peak_group_ranks
    
    # precursor level
    scoring_table_precursor = scoring_table.drop_duplicates(subset = "precursor_run_id", keep = "first")
    scoring_table_precursor = scoring_table_precursor.sort_values(by = "transition_group_id")
    scoring_table_precursor.reset_index(drop = True, inplace = True)
    
    # peptide level
    scoring_table["run_peptide_id"] = scoring_table["filename"] + "_" + scoring_table["FullPeptideName"]
    scoring_table_peptide = scoring_table.sort_values(by = ["run_peptide_id", "dr_score"], ascending = [True, False])
    scoring_table_peptide = scoring_table_peptide.drop_duplicates(subset = "run_peptide_id", keep = "first")
    scoring_table_peptide.drop("run_peptide_id", axis = 1, inplace = True)
    scoring_table_peptide.reset_index(drop = True, inplace = True)
    scoring_table.drop("run_peptide_id", axis = 1, inplace = True)
    
    # protein level
    scoring_table["run_protein_id"] = scoring_table["filename"] + "_" + scoring_table["ProteinName"].astype(str)
    scoring_table_protein = scoring_table.sort_values(by = ["run_protein_id", "dr_score"], ascending = [True, False])
    scoring_table_protein.reset_index(drop = True, inplace = True)
    protein_ids = get_precursor_indices(scoring_table_protein["run_protein_id"])
    protein_intens = []
    for protein in protein_ids:
        if len(protein) < 3:
            protein_intens.append(scoring_table_protein.iloc[protein, :]["Intensity"].sum())
        else:
            protein_intens.append(scoring_table_protein.iloc[protein[:3], :]["Intensity"].sum())
    scoring_table_protein = scoring_table_protein.drop_duplicates(subset = "run_protein_id", keep = "first")
    scoring_table_protein["Intensity"] = protein_intens
    scoring_table_protein.drop("run_protein_id", axis = 1, inplace = True)
    scoring_table_protein.reset_index(drop = True, inplace = True)
    scoring_table.drop("run_protein_id", axis = 1, inplace = True)

    scoring_table_precursor = scoring_table_precursor[scoring_table_precursor["qvalue"] < fdr_precursor]
    scoring_table_peptide = scoring_table_peptide[scoring_table_peptide["qvalue"] < fdr_peptide]
    scoring_table_protein = scoring_table_protein[scoring_table_protein["qvalue"] < fdr_protein]

    scoring_table.to_csv(os.path.join(output_dir, all_result_name), sep = "\t", index = False)
    scoring_table_precursor.to_csv(os.path.join(output_dir, precursor_result_name), sep = "\t", index = False)
    scoring_table_peptide.to_csv(os.path.join(output_dir, peptide_result_name), sep = "\t", index = False)
    scoring_table_protein.to_csv(os.path.join(output_dir, protein_result_name), sep = "\t", index = False)