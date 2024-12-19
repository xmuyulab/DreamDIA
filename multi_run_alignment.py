"""
╔═════════════════════════════════════════════════════╗
║               multi_run_alignment.py                ║
╠═════════════════════════════════════════════════════╣
║   Description: Utility functions of DreamDIAlignR   ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import statsmodels.api as sm

from file_io import load_batch_chromatograms, load_batch_scoring_profiles
from utils import get_precursor_indices, normalize_single_trace
from dream_prophet_utils import load_chromatograms_of_one_precursor_from_memory, Scoring_profile, merge_score_packages

def calc_nc_distance_matrix(runs_under_analysis, dream_dir, rt_normalization_dir_suffix):
    """
    Calculate the normalized N_COMMON distance matrix for a set of runs based on identified precursors.

    Parameters:
    runs_under_analysis (list of str): List of run names to be analyzed.
    dream_dir (str): Directory where the sqDream files are stored.
    rt_normalization_dir_suffix (str): Suffix for the directory containing the identified precursors.

    Returns:
    pd.DataFrame: A DataFrame representing the normalized complement distance matrix for the runs.
    """

    # Load identified IDs
    identified_precursors = {}
    for run_name in runs_under_analysis:
        identified_precursors[run_name] = list(pd.read_csv(os.path.join(dream_dir, run_name + rt_normalization_dir_suffix, "preidentified_ids.txt"), header = None)[0])

    # Calculate distance matrix
    distances = []
    for run_name1 in runs_under_analysis:
        distances_run = []
        for run_name2 in runs_under_analysis:
            N1 = len(identified_precursors[run_name1])
            N2 = len(identified_precursors[run_name2])
            N_common = len(set(identified_precursors[run_name1]) & set(identified_precursors[run_name2]))
            distances_run.append(1 - 2 * N_common / (N1 + N2))
        distances.append(distances_run) 

    distances = pd.DataFrame(np.array(distances), index = runs_under_analysis, columns = runs_under_analysis)
    
    return distances

def calc_euc_distance_matrix(runs_under_analysis, dream_dir, rt_normalization_dir_suffix):   
    """
    Calculate the Euclidean distance matrix for a set of runs based on shared precursors.

    Parameters:
    runs_under_analysis (list of str): List of run names to be analyzed.
    dream_dir (str): Directory where the dream files are stored.
    rt_normalization_dir_suffix (str): Suffix for the directory containing the time points.

    Returns:
    pd.DataFrame: A DataFrame representing the Euclidean distance matrix for the runs.
    """

    time_points_all_runs = {}
    for run_name in runs_under_analysis:
        time_points_all_runs[run_name] = pd.read_csv(os.path.join(dream_dir, run_name + rt_normalization_dir_suffix, "time_points.tsv"), sep = "\t", header = None)
    
    for i, run_name in enumerate(runs_under_analysis):
        if i == 0:
            shared_precursors = set(list(time_points_all_runs[run_name][2]))
        else:
            shared_precursors = shared_precursors & set(list(time_points_all_runs[run_name][2]))
    
    score_matrices = {}
    for run_name in runs_under_analysis:
        time_points = time_points_all_runs[run_name]
        sorted_time_points = time_points[time_points[2].isin(shared_precursors)]
        sorted_time_points = sorted_time_points.sort_values(by = 2)
        score_matrices[run_name] = list(sorted_time_points[3]) + list(sorted_time_points[4])
        
    score_matrices = pd.DataFrame(score_matrices).T
    
    distances = pd.DataFrame(sp.spatial.distance_matrix(score_matrices.values, score_matrices.values), 
                             index = score_matrices.index, columns = score_matrices.index)
        
    return distances

def calc_run_weights(distances):
    """
    Calculate run weights based on the normalized distance matrix.

    Parameters:
    distances (pd.DataFrame): A DataFrame representing the distance matrix for the runs.

    Returns:
    pd.DataFrame: A DataFrame with the same dimensions as the input, where each entry
                  represents the weight of a run relative to the others.
    """

    run_weight_matrix = distances.copy()
    
    for i, run_name in enumerate(list(run_weight_matrix.columns)):
        run_weights_single_run = run_weight_matrix[run_name].values
        run_weights_single_run = np.hstack([run_weights_single_run[:i], run_weights_single_run[i+1:]])
        run_weights_single_run = 1 - (run_weights_single_run - run_weights_single_run.min()) / (run_weights_single_run.max() - run_weights_single_run.min())
        run_weights_single_run = list(run_weights_single_run)
        run_weights_single_run.insert(i, 1)
        run_weight_matrix.loc[:, run_name] = run_weights_single_run
    
    return run_weight_matrix

def build_mst(distance_matrix, r_home, build_mst_r_script):
    """
    Build a Minimum Spanning Tree (MST) from the given distance matrix using an R script.

    Parameters:
    distance_matrix (pd.DataFrame): A DataFrame representing the distance matrix for the runs.
    r_home (str): The path to the R installation directory.
    build_mst_r_script (str): The path to the R script that builds the MST.

    Returns:
    pd.DataFrame: A DataFrame representing the MST with columns ['ref', 'exp', 'weight'],
                  where 'ref' and 'exp' are the nodes connected by an edge and 'weight' is the distance.
    """

    os.environ['R_HOME'] = r_home
    #os.environ["R_LIBS_USER"] = r_library_home

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    with ro.default_converter + pandas2ri.converter:
        distance_matrix_r = ro.conversion.get_conversion().py2rpy(distance_matrix)
    
    r = ro.r
    r['source'](build_mst_r_script)
    build_mst_r = ro.globalenv["getMST"]
    
    mst_r = build_mst_r(distance_matrix_r)
    
    with ro.default_converter + pandas2ri.converter:
        mst_python = ro.conversion.get_conversion().rpy2py(mst_r)

    mst_python = pd.DataFrame(mst_python).values.reshape((2, -1)).T
    mst_python = pd.DataFrame(mst_python, columns = ["ref", "exp"])

    weights = []
    for i in range(mst_python.shape[0]):
        weights.append(distance_matrix.loc[mst_python.iloc[i, 0], mst_python.iloc[i, 1]])
    
    mst_python["weight"] = weights
    
    return mst_python

def traverse_mst(mst):
    """
    Traverse the Minimum Spanning Tree (MST) to determine the order of merging nodes and key runs.

    Parameters:
    mst (pd.DataFrame): A DataFrame representing the MST with columns ['ref', 'exp', 'weight'],
                        where 'ref' and 'exp' are the nodes connected by an edge and 'weight' is the distance.

    Returns:
    tuple: A tuple containing two lists:
           - merge_order: A list of tuples representing the order of merging nodes (ref, exp).
           - key_runs: A list of nodes that are key runs in the order they are traversed.
    """

    all_nodes = list(mst["ref"]) + list(mst["exp"])
    starting_point = list(pd.Series(all_nodes).value_counts().index)[0]
    all_combos = [(idx, i, j) for idx, (i, j) in enumerate(zip(mst["ref"], mst["exp"]))]
    
    merge_order = []  
    key_runs = []  
    next_level = []
    new_combos = []
    for combo in all_combos:
        if starting_point in combo:
            merge_order.append(combo[0])
            if combo[1] == starting_point:
                next_level.append(combo[2])
                key_runs.append(combo[1])
            else:
                next_level.append(combo[1])
                key_runs.append(combo[2])
        else:
            new_combos.append(combo)
    all_combos = new_combos
    
    while len(all_combos) > 0:
        new_combos = []
        new_next_level = []
        for combo in all_combos:
            if combo[1] in next_level:
                merge_order.append(combo[0])
                new_next_level.append(combo[2])
                key_runs.append(combo[1])
            elif combo[2] in next_level:
                merge_order.append(combo[0])
                new_next_level.append(combo[1])
                key_runs.append(combo[2])
            else:
                new_combos.append(combo)
        all_combos = new_combos
        next_level = new_next_level

    merge_order = [(mst.iloc[i, :]["ref"], mst.iloc[i, :]["exp"]) for i in merge_order]
    
    return merge_order, key_runs

def get_linear_global_fit(mst, dream_dir, rt_normalization_dir_suffix):
    global_fit = {}
    for ref_run_name, exp_run_name in zip(mst["ref"], mst["exp"]):
        identified_ids_ref = list(pd.read_csv(os.path.join(dream_dir, ref_run_name + rt_normalization_dir_suffix, "preidentified_ids.txt"), header = None)[0])
        identified_ids_exp = list(pd.read_csv(os.path.join(dream_dir, exp_run_name + rt_normalization_dir_suffix, "preidentified_ids.txt"), header = None)[0])

        shared_identified_ids = list(set(identified_ids_ref) & set(identified_ids_exp))

        time_points_ref = pd.read_csv(os.path.join(dream_dir, ref_run_name + rt_normalization_dir_suffix, "time_points.tsv"), sep = "\t", header = None)
        time_points_exp = pd.read_csv(os.path.join(dream_dir, exp_run_name + rt_normalization_dir_suffix, "time_points.tsv"), sep = "\t", header = None)
        
        inlier_rts_ref = time_points_ref[time_points_ref[2].isin(shared_identified_ids)][1].values
        inlier_rts_exp = time_points_exp[time_points_exp[2].isin(shared_identified_ids)][1].values
        
        lr_RAN = RANSACRegressor(LinearRegression(), random_state = 12580)
        lr_RAN.fit(inlier_rts_ref.reshape(-1, 1), inlier_rts_exp)
        new_lr = LinearRegression()
        new_lr.fit(inlier_rts_ref.reshape(-1, 1)[lr_RAN.inlier_mask_], inlier_rts_exp[lr_RAN.inlier_mask_])
        slope, intercept = new_lr.coef_[0], new_lr.intercept_

        residuals = np.abs(inlier_rts_ref * slope + intercept - inlier_rts_exp)
        rse = np.sqrt(sum(residuals ** 2) / (len(residuals) - 2))
        adaptive_rt = 1.5 * rse

        global_fit[(ref_run_name, exp_run_name)] = [slope, intercept, adaptive_rt]

    return global_fit

def get_lowess_global_fit(mst, dream_dir, rt_normalization_dir_suffix, span_value = 0.1):
    global_fit = {}
    for ref_run_name, exp_run_name in zip(mst["ref"], mst["exp"]):
        identified_ids_ref = list(pd.read_csv(os.path.join(dream_dir, ref_run_name + rt_normalization_dir_suffix, "preidentified_ids.txt"), header = None)[0])
        identified_ids_exp = list(pd.read_csv(os.path.join(dream_dir, exp_run_name + rt_normalization_dir_suffix, "preidentified_ids.txt"), header = None)[0])

        shared_identified_ids = list(set(identified_ids_ref) & set(identified_ids_exp))

        time_points_ref = pd.read_csv(os.path.join(dream_dir, ref_run_name + rt_normalization_dir_suffix, "time_points.tsv"), sep = "\t", header = None)
        time_points_exp = pd.read_csv(os.path.join(dream_dir, exp_run_name + rt_normalization_dir_suffix, "time_points.tsv"), sep = "\t", header = None)
        
        inlier_rts_ref = time_points_ref[time_points_ref[2].isin(shared_identified_ids)][1].values
        inlier_rts_exp = time_points_exp[time_points_exp[2].isin(shared_identified_ids)][1].values

        drop_duplicate_df = pd.DataFrame({"ref" : inlier_rts_ref, "exp" : inlier_rts_exp})
        drop_duplicate_df = drop_duplicate_df.sort_values(by = "ref")
        drop_duplicate_df = drop_duplicate_df.drop_duplicates(subset = "ref", keep = "first")
        drop_duplicate_df = drop_duplicate_df.sort_values(by = "exp")
        drop_duplicate_df = drop_duplicate_df.drop_duplicates(subset = "exp", keep = "first")
        inlier_rts_ref = drop_duplicate_df["ref"].values
        inlier_rts_exp = drop_duplicate_df["exp"].values

        smoothed_points = sm.nonparametric.lowess(inlier_rts_exp, inlier_rts_ref, frac = span_value, it = 3, delta = 0)
        lowess_x = list(zip(*smoothed_points))[0]
        lowess_y = list(zip(*smoothed_points))[1]
        lowess_res = pd.DataFrame({"x" : lowess_x, "y" : lowess_y})
        
        interpolate_function = interp1d(lowess_x, lowess_y, bounds_error = False, fill_value = "extrapolate")
        predicted_y = interpolate_function(inlier_rts_ref)
        
        residuals = np.abs(predicted_y - inlier_rts_exp)
        rse = np.sqrt(sum(residuals ** 2) / (len(residuals) - 2))

        # Similarity matrix is not penalized within adaptive RT.
        # Small: global; Large: local.
        adaptive_rt = 3.5 * rse

        global_fit[(ref_run_name, exp_run_name)] = [lowess_res, adaptive_rt]

    return global_fit

def build_global_constraint(
    out, 
    runs_under_analysis, 
    dream_dir, 
    rt_normalization_dir_suffix, 
    r_home, 
    distance_metric, 
    exp_decay, 
    build_mst_r_script, 
    global_constraint_type, 
    span_value
):
    if distance_metric == "nc":
        distance_matrix = calc_nc_distance_matrix(runs_under_analysis, dream_dir, rt_normalization_dir_suffix)
    else:
        distance_matrix = calc_euc_distance_matrix(runs_under_analysis, dream_dir, rt_normalization_dir_suffix)
        
    run_weights = calc_run_weights(distance_matrix)
    run_weights.to_csv(os.path.join(out, "run_weights.csv"))
    run_weights = (exp_decay ** run_weights - 1) / (exp_decay - 1)
        
    mst = build_mst(distance_matrix, r_home, build_mst_r_script)
    
    merge_order, key_runs = traverse_mst(mst)
    
    if global_constraint_type == "lowess":
        global_fit = get_lowess_global_fit(mst, dream_dir, rt_normalization_dir_suffix, span_value)
    else:
        global_fit = get_linear_global_fit(mst, dream_dir, rt_normalization_dir_suffix)
    
    with open(os.path.join(out, "global_fit.pkl"), "wb") as f:
        pickle.dump(global_fit, f)

    return run_weights, merge_order, key_runs, global_fit

def do_align_linear(rt_list_ref, ms2_inten_lists_ref, rt_list_exp, ms2_inten_lists_exp, 
                    getAlignedTimesFast_function, global_fit_arguments):
    """DIAlignR alignment with linear global constraint."""
    
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    xic_df_ref = pd.DataFrame(np.array([rt_list_ref] + ms2_inten_lists_ref).T, 
                              columns = ["time"] + ["frag_%d" % i for i in range(len(ms2_inten_lists_ref))])
    xic_df_exp = pd.DataFrame(np.array([rt_list_exp] + ms2_inten_lists_exp).T, 
                              columns = ["time"] + ["frag_%d" % i for i in range(len(ms2_inten_lists_exp))])

    with ro.default_converter + pandas2ri.converter:
        xic_r_df_ref = ro.conversion.get_conversion().py2rpy(xic_df_ref)
    with ro.default_converter + pandas2ri.converter:
        xic_r_df_exp = ro.conversion.get_conversion().py2rpy(xic_df_exp)
    
    global_fit_arguments_r = ro.FloatVector(global_fit_arguments)
    
    align_res_r = getAlignedTimesFast_function(xic_r_df_ref, xic_r_df_exp, global_fit_arguments_r)
    
    with ro.default_converter + pandas2ri.converter:
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(align_res_r)

    return pd_from_r_df

def do_align_lowess(rt_list_ref, ms2_inten_lists_ref, rt_list_exp, ms2_inten_lists_exp, 
                    getAlignedTimesFast_function, global_fit_arguments):
    """DIAlignR alignment with lowess global constraint."""
    
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    xic_df_ref = pd.DataFrame(np.array([rt_list_ref] + ms2_inten_lists_ref).T, 
                              columns = ["time"] + ["frag_%d" % i for i in range(len(ms2_inten_lists_ref))])
    xic_df_exp = pd.DataFrame(np.array([rt_list_exp] + ms2_inten_lists_exp).T, 
                              columns = ["time"] + ["frag_%d" % i for i in range(len(ms2_inten_lists_exp))])

    with ro.default_converter + pandas2ri.converter:
        xic_r_df_ref = ro.conversion.get_conversion().py2rpy(xic_df_ref)
    with ro.default_converter + pandas2ri.converter:
        xic_r_df_exp = ro.conversion.get_conversion().py2rpy(xic_df_exp)

    with ro.default_converter + pandas2ri.converter:
        lowess_coef_r = ro.conversion.get_conversion().py2rpy(global_fit_arguments[0])
    
    adaptive_rt_r = ro.FloatVector([global_fit_arguments[1]])
    
    align_res_r = getAlignedTimesFast_function(xic_r_df_ref, xic_r_df_exp, lowess_coef_r, adaptive_rt_r)
    
    with ro.default_converter + pandas2ri.converter:
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(align_res_r)

    return pd_from_r_df

def get_global_alignment_times_linear(rt_list_ref, rt_list_exp, slope, intercept, tol = 3.3):
    """Linear global alignment."""
    
    rt_list_ref_to_exp = rt_list_ref * slope + intercept

    aligned_rt_list_ref, aligned_rt_list_exp = [], []

    for pointer_ref, (time_point_ref, time_point_ref_to_exp) in enumerate(zip(rt_list_ref, rt_list_ref_to_exp)):
        subtract = np.abs(time_point_ref_to_exp - rt_list_exp)
        if min(subtract) < tol:            
            aligned_index = np.argmin(subtract)
            if pointer_ref == 0 and aligned_index > 0:
                aligned_rt_list_ref.extend([-1] * aligned_index)
                aligned_rt_list_exp.extend(rt_list_exp[:aligned_index])
            aligned_rt_list_ref.append(time_point_ref)
            aligned_rt_list_exp.append(rt_list_exp[aligned_index])
        else:
            aligned_rt_list_ref.append(time_point_ref)
            aligned_rt_list_exp.append(-1)

    return aligned_rt_list_ref, aligned_rt_list_exp

def get_global_alignment_times_lowess(rt_list_ref, rt_list_exp, lowess_params, tol = 3.3):
    """Lowess global alignment."""
    
    interpolate_function = interp1d(list(lowess_params["x"]), list(lowess_params["y"]), bounds_error = False, fill_value = "extrapolate")
    rt_list_ref_to_exp = interpolate_function(np.array(rt_list_ref))

    aligned_rt_list_ref, aligned_rt_list_exp = [], []

    for pointer_ref, (time_point_ref, time_point_ref_to_exp) in enumerate(zip(rt_list_ref, rt_list_ref_to_exp)):
        subtract = np.abs(time_point_ref_to_exp - rt_list_exp)
        if min(subtract) < tol:            
            aligned_index = np.argmin(subtract)
            if pointer_ref == 0 and aligned_index > 0:
                aligned_rt_list_ref.extend([-1] * aligned_index)
                aligned_rt_list_exp.extend(rt_list_exp[:aligned_index])
            aligned_rt_list_ref.append(time_point_ref)
            aligned_rt_list_exp.append(rt_list_exp[aligned_index])
        else:
            aligned_rt_list_ref.append(time_point_ref)
            aligned_rt_list_exp.append(-1)

    return aligned_rt_list_ref, aligned_rt_list_exp

def merge_aligned_times(aligned_times, merge_order, key_runs):
    time_vectors = aligned_times[merge_order[0]]
    time_vectors = time_vectors.fillna(-1)
    for merge_id, key_run in zip(merge_order[1:], key_runs[1:]):
        time_vectors = pd.merge(time_vectors, aligned_times[merge_id], on = key_run, how = "left")
        time_vectors = time_vectors.fillna(-1)
    time_vectors[time_vectors == -1] = pd.NA
    
    return time_vectors

def align_a_batch_of_precursors(
    alignment_queue, 
    dream_dir, 
    runs_under_analysis, 
    sqdream_files, 
    batch_precursor, 
    global_fit, 
    mra_algorithm, 
    global_constraint_type, 
    rt_tolerance, 
    merge_order, 
    key_runs, 
    getAlignedTimesFast_function
):

    batch_chromatograms = {}
    batch_scoring_profiles = {}

    for run_name, sqdream_file in zip(runs_under_analysis, sqdream_files):
        batch_chromatograms[run_name] = load_batch_chromatograms(os.path.join(dream_dir, sqdream_file), batch_precursor)
        batch_scoring_profiles[run_name] = load_batch_scoring_profiles(os.path.join(dream_dir, sqdream_file), batch_precursor)
    
    aligned_time_vectors = {}
    
    precursor_row_indices = get_precursor_indices(list(batch_chromatograms[runs_under_analysis[0]]["PRECURSOR_ID"]))

    for precursor_row_idx in precursor_row_indices:
        precursor_id = list(batch_chromatograms[runs_under_analysis[0]].iloc[precursor_row_idx, :]["PRECURSOR_ID"])[0]
        aligned_res_precursor = {}

        for (ref_run, exp_run) in global_fit:
            rt_list_ref, ms2_inten_lists_ref, ms1_inten_ref = load_chromatograms_of_one_precursor_from_memory(batch_chromatograms[ref_run], precursor_row_idx)
            rt_list_exp, ms2_inten_lists_exp, ms1_inten_exp = load_chromatograms_of_one_precursor_from_memory(batch_chromatograms[exp_run], precursor_row_idx)

            if mra_algorithm == "dialignr":
                if global_constraint_type == "lowess":
                    aligned_vector = do_align_lowess(rt_list_ref, ms2_inten_lists_ref,
                                                     rt_list_exp, ms2_inten_lists_exp,
                                                     getAlignedTimesFast_function, global_fit[(ref_run, exp_run)])
                else:
                    aligned_vector = do_align_linear(rt_list_ref, ms2_inten_lists_ref,
                                                     rt_list_exp, ms2_inten_lists_exp,
                                                     getAlignedTimesFast_function, global_fit[(ref_run, exp_run)])
                aligned_vector = pd.DataFrame(aligned_vector, columns=[ref_run, exp_run])

            else:
                if global_constraint_type == "lowess":
                    aligned_time_vector_ref, aligned_time_vector_exp = get_global_alignment_times_lowess(rt_list_ref,
                                                                                                         rt_list_exp,
                                                                                                         global_fit[(ref_run, exp_run)][0],
                                                                                                         tol=rt_tolerance)
                else:
                    aligned_time_vector_ref, aligned_time_vector_exp = get_global_alignment_times_linear(rt_list_ref,
                                                                                                         rt_list_exp,
                                                                                                         global_fit[(ref_run, exp_run)][0],
                                                                                                         global_fit[(ref_run, exp_run)][1],
                                                                                                         tol=rt_tolerance)

                aligned_vector = pd.DataFrame({ref_run: aligned_time_vector_ref, exp_run: aligned_time_vector_exp}, dtype=np.float64)
                aligned_vector[aligned_vector == -1] = np.nan

            aligned_res_precursor[(ref_run, exp_run)] = aligned_vector

        aligned_res_precursor = merge_aligned_times(aligned_res_precursor, merge_order, key_runs)
        aligned_time_vectors[precursor_id] = aligned_res_precursor

    alignment_queue.put([aligned_time_vectors, batch_scoring_profiles])

def aligned_time_to_index(aligned_time, middle_rts, rt_tolerance = 1.65):
    middle_rt_middle_points = [(middle_rts[i] + middle_rts[i + 1]) / 2 for i in range(len(middle_rts) - 1)]
    middle_rt_middle_points = list(np.round(middle_rt_middle_points, 2))
    
    middle_rts = list(np.round(middle_rts, 2))
    
    aligned_index = []
    for time_point in aligned_time:
        if pd.isna(time_point):
            aligned_index.append(pd.NA)
        else:
            try:
                aligned_index.append(middle_rts.index(time_point))
            except ValueError:
                try:
                    aligned_index.append(middle_rt_middle_points.index(time_point))
                except ValueError:
                    bias = np.abs(time_point - np.array(middle_rts))
                    if min(bias) > rt_tolerance:
                        aligned_index.append(pd.NA)
                    else:
                        aligned_index.append(np.argmin(bias))
    return aligned_index

def time_vector_to_index_vector(time_vector, precursor_sp_info_list, rt_tolerance):
    index_vector = {}
    for run_name in list(time_vector.columns):
        aligned_time = list(time_vector[run_name])
        middle_rts = list(precursor_sp_info_list[run_name].scores["middle_rts"])
        index_vector[run_name] = aligned_time_to_index(aligned_time, middle_rts, rt_tolerance / 2)
        
    return pd.DataFrame(index_vector)

def interpolate_index_vector(index_vector):
    interpolated_index_vector = {}
    for run_name in list(index_vector.columns):
        interpolated_index_vector[run_name] = []
        
        last_not_na = -1
        na_count = 0
        
        for time_point in index_vector[run_name]:
            if pd.isna(time_point):
                na_count += 1
            else:
                if last_not_na == -1:
                    if time_point == 0:
                        interpolated_index_vector[run_name].extend([pd.NA] * na_count + [0])                        
                    else:
                        padding = list(range(time_point))
                        if len(padding) <= na_count:
                            interpolated_index_vector[run_name].extend([pd.NA] * (na_count - len(padding)) + padding)
                        else:
                            interpolated_index_vector[run_name].extend(padding[time_point - na_count:])
                        interpolated_index_vector[run_name].append(time_point)
                                        
                    na_count = 0    
                
                else:
                    if na_count == 0:
                        interpolated_index_vector[run_name].append(time_point)
                    else:
                        padding = [int(ii) for ii in np.linspace(last_not_na, time_point, na_count + 2)[1:]]                        
                        interpolated_index_vector[run_name].extend(padding)
                        na_count = 0                       
                    
                last_not_na = time_point
        
        if na_count != 0:
            interpolated_index_vector[run_name].extend([pd.NA] * na_count)
                
    return pd.DataFrame(interpolated_index_vector).dropna().astype(int)

def index_vector_to_score_vector(index_vector, precursor_sp_info_list, score_name):
    score_vector = {}
    for run_name in list(index_vector.columns):        
        score_vector[run_name] = precursor_sp_info_list[run_name].scores[score_name][index_vector[run_name].values]
    return pd.DataFrame(score_vector)

def calc_weighted_means(scoring_df, weights):
    weighted_scoring_df = {}
    for run_name in list(scoring_df.columns):
        weighted_scoring_df[run_name] = scoring_df[run_name].values * weights[run_name]
    return pd.DataFrame(weighted_scoring_df).mean(axis = 1).values

def pick_peaks_and_score_multi_run(
    precursor_time_vector, 
    precursor_sp_info_list, 
    top_k, 
    delta_rt_weight, 
    run_weights, 
    rt_tolerance, 
    peak_picking_mode = "average"
):
    aligned_runs = list(precursor_time_vector.columns)

    if precursor_time_vector.shape[0] < 15:
        precursor_score_package_list = []
        for run in precursor_sp_info_list:
            scoring_profile = precursor_sp_info_list[run]
            scoring_profile.pick_peaks_and_score_single_run(top_k)
            precursor_score_package_list.append(scoring_profile.picked_scores)
        precursor_score_package = merge_score_packages(precursor_score_package_list)
        return precursor_score_package

    index_vector = interpolate_index_vector(time_vector_to_index_vector(precursor_time_vector, precursor_sp_info_list, rt_tolerance))
    if index_vector.shape[0] < 15:
        precursor_score_package_list = []
        for run in precursor_sp_info_list:
            scoring_profile = precursor_sp_info_list[run]
            scoring_profile.pick_peaks_and_score_single_run(top_k)
            precursor_score_package_list.append(scoring_profile.picked_scores)
        precursor_score_package = merge_score_packages(precursor_score_package_list)
    else:
        dream_score_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "dream_scores")
        delta_rt_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "delta_rt")
        quant_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "quantification")
        lib_cos_score_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "lib_cos_scores")
        ms1_area_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "ms1_area")
        ms2_area_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "ms2_area")
        middle_rt_vector = index_vector_to_score_vector(index_vector, precursor_sp_info_list, "middle_rts")

        aligned_dream_score = {}
        aligned_lib_cos_score = {}
        aligned_ms1_area = {}
        aligned_ms2_area = {}
        aligned_delta_rt = {}

        for run_name in aligned_runs:
            aligned_dream_score[run_name] = calc_weighted_means(dream_score_vector, run_weights[run_name])
            aligned_lib_cos_score[run_name] = calc_weighted_means(lib_cos_score_vector, run_weights[run_name])
            aligned_ms1_area[run_name] = calc_weighted_means(ms1_area_vector, run_weights[run_name])
            aligned_ms2_area[run_name] = calc_weighted_means(ms2_area_vector, run_weights[run_name])
            aligned_delta_rt[run_name] = calc_weighted_means(delta_rt_vector, run_weights[run_name])
        
        aligned_dream_score = pd.DataFrame(aligned_dream_score)
        aligned_lib_cos_score = pd.DataFrame(aligned_lib_cos_score)
        aligned_ms1_area = pd.DataFrame(aligned_ms1_area)
        aligned_ms2_area = pd.DataFrame(aligned_ms2_area)
        aligned_delta_rt = pd.DataFrame(aligned_delta_rt)

        if peak_picking_mode == "average":
            peak_picking_base = dream_score_vector.mean(axis = 1).values - delta_rt_weight * normalize_single_trace(delta_rt_vector.mean(axis = 1).values)
            picked_indice = np.argsort(-peak_picking_base)[:top_k]
        
            picked_dream_score_vector = dream_score_vector.iloc[picked_indice, :]
            picked_delta_rt_vector = delta_rt_vector.iloc[picked_indice, :]
            picked_quant_vector = quant_vector.iloc[picked_indice, :]
            picked_lib_cos_score_vector = lib_cos_score_vector.iloc[picked_indice, :]
            picked_ms1_area_vector = ms1_area_vector.iloc[picked_indice, :]
            picked_ms2_area_vector = ms2_area_vector.iloc[picked_indice, :]   
            picked_middle_rt_vector = middle_rt_vector.iloc[picked_indice, :]

            picked_aligned_dream_score = aligned_dream_score.iloc[picked_indice, :]
            picked_aligned_lib_cos_score = aligned_lib_cos_score.iloc[picked_indice, :]
            picked_aligned_ms1_area = aligned_ms1_area.iloc[picked_indice, :]
            picked_aligned_ms2_area = aligned_ms2_area.iloc[picked_indice, :]

        else:
            picked_dream_score_vector = {}
            picked_delta_rt_vector = {}
            picked_quant_vector = {}
            picked_lib_cos_score_vector = {}
            picked_ms1_area_vector = {}
            picked_ms2_area_vector = {}
            picked_middle_rt_vector = {}

            picked_aligned_dream_score = {}
            picked_aligned_lib_cos_score = {}
            picked_aligned_ms1_area = {}
            picked_aligned_ms2_area = {}

            for run_name in aligned_runs:
                peak_picking_base = aligned_dream_score[run_name].values - delta_rt_weight * normalize_single_trace(aligned_delta_rt[run_name].values)
                picked_indice = np.argsort(-peak_picking_base)[:top_k]

                picked_dream_score_vector[run_name] = dream_score_vector[run_name].values[picked_indice]
                picked_delta_rt_vector[run_name] = delta_rt_vector[run_name].values[picked_indice]
                picked_quant_vector[run_name] = quant_vector[run_name].values[picked_indice]
                picked_lib_cos_score_vector[run_name] = lib_cos_score_vector[run_name].values[picked_indice]
                picked_ms1_area_vector[run_name] = ms1_area_vector[run_name].values[picked_indice]
                picked_ms2_area_vector[run_name] = ms2_area_vector[run_name].values[picked_indice]
                picked_middle_rt_vector[run_name] = middle_rt_vector[run_name].values[picked_indice]

                picked_aligned_dream_score[run_name] = aligned_dream_score[run_name].values[picked_indice]
                picked_aligned_lib_cos_score[run_name] = aligned_lib_cos_score[run_name].values[picked_indice]
                picked_aligned_ms1_area[run_name] = aligned_ms1_area[run_name].values[picked_indice]
                picked_aligned_ms2_area[run_name] = aligned_ms2_area[run_name].values[picked_indice]

            picked_dream_score_vector = pd.DataFrame(picked_dream_score_vector)
            picked_delta_rt_vector = pd.DataFrame(picked_delta_rt_vector)
            picked_quant_vector = pd.DataFrame(picked_quant_vector)
            picked_lib_cos_score_vector = pd.DataFrame(picked_lib_cos_score_vector)
            picked_ms1_area_vector = pd.DataFrame(picked_ms1_area_vector)
            picked_ms2_area_vector = pd.DataFrame(picked_ms2_area_vector)
            picked_middle_rt_vector = pd.DataFrame(picked_middle_rt_vector)

            picked_aligned_dream_score = pd.DataFrame(picked_aligned_dream_score)
            picked_aligned_lib_cos_score = pd.DataFrame(picked_aligned_lib_cos_score)
            picked_aligned_ms1_area = pd.DataFrame(picked_aligned_ms1_area)
            picked_aligned_ms2_area = pd.DataFrame(picked_aligned_ms2_area)

        rt_mean = picked_middle_rt_vector.mean(axis = 0)
        rt_std = picked_middle_rt_vector.std(axis = 0)
        delta_rt_mean = picked_delta_rt_vector.mean(axis = 0)
        delta_rt_std = picked_delta_rt_vector.std(axis = 0)
        dream_score_mean = picked_dream_score_vector.mean(axis = 0)
        dream_score_std = picked_dream_score_vector.std(axis = 0)
        lib_cos_score_mean = picked_lib_cos_score_vector.mean(axis = 0)
        lib_cos_score_std = picked_lib_cos_score_vector.std(axis = 0)

        precursor_score_package = {"alignment_boosted" : 1, 
                                   "dream_scores" : picked_dream_score_vector, 
                                   "delta_rt" : picked_delta_rt_vector, 
                                   "lib_cos_scores" : picked_lib_cos_score_vector, 
                                   "ms1_area" : picked_ms1_area_vector, 
                                   "ms2_area" : picked_ms2_area_vector, 
                                   "middle_rts" : picked_middle_rt_vector, 
                                   "quant" : picked_quant_vector, 
                                   "aligned_dream_score" : picked_aligned_dream_score, 
                                   "aligned_lib_cos_score" : picked_aligned_lib_cos_score, 
                                   "aligned_ms1_area" : picked_aligned_ms1_area, 
                                   "aligned_ms2_area" : picked_aligned_ms2_area, 
                                   "rt_mean" : rt_mean, 
                                   "rt_std" : rt_std, 
                                   "delta_rt_mean" : delta_rt_mean, 
                                   "delta_rt_std" : delta_rt_std, 
                                   "dream_score_mean" : dream_score_mean, 
                                   "dream_score_std" : dream_score_std, 
                                   "lib_cos_score_mean" : lib_cos_score_mean, 
                                   "lib_cos_score_std" : lib_cos_score_std}

    return precursor_score_package

def get_peak_picking_cross_run_results(
    alignment_queue, 
    feature_queue, 
    runs_under_analysis, 
    logger, 
    top_k, 
    rt_tolerance = 3.3,
    delta_rt_weight = 0, 
    run_weights = None, 
    peak_picking_mode = "average"
):
    """
    Get aligned time verctors and metadata from `alignment_queue`, and output aligned features to `feature_queue`.

    Parameters:
        alignment_queue (multiprocessing.Manager.JoinableQueue): Queue to get aligned data from.
        feature_queue (multiprocessing.Manager.JoinableQueue): Queue to put calculated features into.
        runs_under_analysis (list): List of run names to analyze.
        logger (logging.Logger): Logger for logging information.
        top_k (int): Top K peaks to consider.
        rt_tolerance (float): Retention time tolerance.
        delta_rt_weight (float): Weight for delta retention time.
        run_weights (dict): Weights for different runs.
        peak_picking_mode (str): Mode for peak picking.
    """
    
    while 1:
        aligned_data = alignment_queue.get()
        if aligned_data is None:
            alignment_queue.task_done()
            logger.info("Cross-run peak picking done!")
            break
        
        aligned_time_vectors, batch_scoring_profiles = aligned_data
        alignment_queue.task_done()
        
        # Iterate over precursors
        for i in range(batch_scoring_profiles[runs_under_analysis[0]].shape[0]):
            precursor_sp_info_list = {
                run_name: Scoring_profile(batch_scoring_profiles[run_name].iloc[i, :], run_name)
                for run_name in runs_under_analysis
            }   

            #logger.info(precursor_sp_info_list[runs_under_analysis[0]].get_static_info())      

            # Aligned time vector DataFrame of this precursor
            # pd.DataFrame: col = run_names, row = rt, value = rt values
            precursor_time_vector = aligned_time_vectors[precursor_sp_info_list[runs_under_analysis[0]].precursor_id]

            # Start calculating multi-run score
            precursor_score_package = pick_peaks_and_score_multi_run(precursor_time_vector, precursor_sp_info_list, top_k, delta_rt_weight, run_weights, rt_tolerance, peak_picking_mode)

            feature_queue.put([precursor_sp_info_list[runs_under_analysis[0]].get_static_info(), precursor_score_package])

    feature_queue.put(None)

def initiate_scoring_table_multi_run():
    scoring_table = {"transition_group_id" : [], 
                     "PeptideSequence" : [], 
                     "FullPeptideName" : [], 
                     "irt" : [], 
                     "ProteinName" : [], 
                     "filename" : [], 
                     "alignment_boosted" : [], 
                     "rank" : [], 
                     "SCORE_RT" : [], "SCORE_RT_mean" : [], "SCORE_RT_std" : [], 
                     "SCORE_dream" : [], "SCORE_dream_mean" : [], "SCORE_dream_std" : [], 
                     "SCORE_lib_cosine" : [], "SCORE_lib_cosine_mean" : [], "SCORE_lib_cosine_std" : [], 
                     "SCORE_deltaRT" : [], "SCORE_deltaRT_mean" : [], "SCORE_deltaRT_std" : [], 
                     "SCORE_ALIGN_DREAM" : [], 
                     "SCORE_ALIGN_LIBCOS" : [], 
                     "SCORE_ALIGN_MS1_AREA" : [], 
                     "SCORE_ALIGN_MS2_AREA" : [],                     
                     "SCORE_MS1_area" : [], "SCORE_MS2_area" : [], 
                     "SCORE_charge" : [], "SCORE_pep_len" : [], "SCORE_mz" : [], "decoy" : [], 
                     "Intensity" : []}
    return scoring_table

def collect_scoring_table_multi_run(feature_queue, runs_under_analysis, out, top_k, n_writting_batch, logger):
    scoring_table = initiate_scoring_table_multi_run()

    output_count = 0

    while 1:
        scoring_data = feature_queue.get()

        if scoring_data is None:
            feature_queue.task_done()
            break

        precursor_static_info, precursor_score_package = scoring_data
        
        if precursor_static_info is None:
            feature_queue.task_done()
            continue

        #logger.info(precursor_static_info)

        for run_name in runs_under_analysis:
            scoring_table["transition_group_id"].extend([precursor_static_info["precursor_id"]] * top_k)
            scoring_table["PeptideSequence"].extend([precursor_static_info["pure_sequence"]] * top_k)
            scoring_table["FullPeptideName"].extend([precursor_static_info["peptide_sequence"]] * top_k)
            scoring_table["irt"].extend([precursor_static_info["irt"]] * top_k)
            scoring_table["ProteinName"].extend([precursor_static_info["protein_name"]] * top_k)           
            scoring_table["filename"].extend([run_name] * top_k)
            scoring_table["alignment_boosted"].extend([precursor_score_package["alignment_boosted"]] * top_k)
            scoring_table["rank"].extend(list(range(top_k)))
            
            score_rt = precursor_score_package["middle_rts"][run_name].values
            scoring_table["SCORE_RT"].extend(score_rt)
            scoring_table["SCORE_RT_mean"].extend([precursor_score_package["rt_mean"][run_name]] * top_k)
            scoring_table["SCORE_RT_std"].extend([precursor_score_package["rt_std"][run_name]] * top_k)
            
            score_dream = precursor_score_package["dream_scores"][run_name].values
            scoring_table["SCORE_dream"].extend(score_dream)
            scoring_table["SCORE_dream_mean"].extend([precursor_score_package["dream_score_mean"][run_name]] * top_k)
            scoring_table["SCORE_dream_std"].extend([precursor_score_package["dream_score_std"][run_name]] * top_k)
            
            score_libcos = precursor_score_package["lib_cos_scores"][run_name].values
            scoring_table["SCORE_lib_cosine"].extend(score_libcos)
            scoring_table["SCORE_lib_cosine_mean"].extend([precursor_score_package["lib_cos_score_mean"][run_name]] * top_k)
            scoring_table["SCORE_lib_cosine_std"].extend([precursor_score_package["lib_cos_score_std"][run_name]] * top_k)

            score_deltart = precursor_score_package["delta_rt"][run_name].values
            scoring_table["SCORE_deltaRT"].extend(score_deltart)
            scoring_table["SCORE_deltaRT_mean"].extend([precursor_score_package["delta_rt_mean"][run_name]] * top_k)
            scoring_table["SCORE_deltaRT_std"].extend([precursor_score_package["delta_rt_std"][run_name]] * top_k)
            
            try:
                score_align = precursor_score_package["aligned_dream_score"][run_name].values
                scoring_table["SCORE_ALIGN_DREAM"].extend(score_align)
            except:
                logger.info(precursor_score_package["aligned_dream_score"])

            score_align = precursor_score_package["aligned_lib_cos_score"][run_name].values
            scoring_table["SCORE_ALIGN_LIBCOS"].extend(score_align)

            score_align = precursor_score_package["aligned_ms1_area"][run_name].values
            scoring_table["SCORE_ALIGN_MS1_AREA"].extend(score_align)

            score_align = precursor_score_package["aligned_ms2_area"][run_name].values
            scoring_table["SCORE_ALIGN_MS2_AREA"].extend(score_align)
            
            score_ms1_area = precursor_score_package["ms1_area"][run_name].values
            score_ms2_area = precursor_score_package["ms2_area"][run_name].values
            scoring_table["SCORE_MS1_area"].extend(score_ms1_area)
            scoring_table["SCORE_MS2_area"].extend(score_ms2_area)
            
            scoring_table["SCORE_charge"].extend([precursor_static_info["precursor_charge"]] * top_k)
            scoring_table["SCORE_pep_len"].extend([len(precursor_static_info["pure_sequence"])] * top_k)
            scoring_table["SCORE_mz"].extend([precursor_static_info["precursor_mz"]] * top_k)
            scoring_table["decoy"].extend([precursor_static_info["decoy"]] * top_k)

            quant_res = precursor_score_package["quant"][run_name].values
            scoring_table["Intensity"].extend(quant_res)

        feature_queue.task_done()
        
        if len(scoring_table["transition_group_id"]) >= n_writting_batch:
            with open(os.path.join(out, "FT_cache%d.pkl" % output_count), "wb") as ff:
                pickle.dump(scoring_table, ff)
            output_count += 1
            scoring_table = initiate_scoring_table_multi_run()
            
    if len(scoring_table["transition_group_id"]) != 0:
        with open(os.path.join(out, "FT_cache%d.pkl" % output_count), "wb") as ff:
            pickle.dump(scoring_table, ff)

def output_scoring_table(out, multi_run_scoring_table_name):
    ft_cache_files = [i for i in os.listdir(out) if i.startswith("FT_cache")]
    scoring_table = []
    for ft_cache_file in ft_cache_files:
        ft_cache = pickle.load(open(os.path.join(out, ft_cache_file), "rb"))
        ft_cache = pd.DataFrame(ft_cache)
        scoring_table.append(ft_cache)
    scoring_table = pd.concat(scoring_table)
    scoring_table.to_csv(os.path.join(out, multi_run_scoring_table_name), sep = "\t", index = False)
    for ft_cache_file in ft_cache_files:
        os.remove(os.path.join(out, ft_cache_file))
    
    return scoring_table