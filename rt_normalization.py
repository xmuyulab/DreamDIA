"""
╔═════════════════════════════════════════════════════╗
║                 rt_normalization.py                 ║
╠═════════════════════════════════════════════════════╣
║     Description: Utility functions for retention    ║
║                time normalization                   ║
╠═════════════════════════════════════════════════════╣
║         Author: Mingxuan Gao, Wenxian Yang          ║
║         Contact: mingxuan.gao@utoronto.ca           ║
╚═════════════════════════════════════════════════════╝
"""

import re
import os
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import statsmodels.api as sm
from scipy.interpolate import interp1d

# from sklearn.metrics.pairwise import cosine_similarity
# Warning: the multiprocessing code will be killed silently if sklearn cosine_similarity is used.
# Warning: very hard to debug!
from scipy import spatial

from mz_calculator import calc_all_fragment_mzs
from gpu_settings import set_gpu_memory
import tools_cython as tools
from utils import calc_win_id, find_rt_pos, calc_XIC, filter_matrix, calc_pearson_sums, adjust_size, adjust_cycle, flatten_list, tukey_inliers
from third_party.calib_rt import Calib_RT

def generate_endoIRT(lib_cols, library, n_irts, seed):
    """
    Generate endogenous iRT peptide library.
    
    Args:
        lib_cols (dict): Dictionary of library columns.
        library (pd.DataFrame): The spectral library DataFrame.
        n_irts (int): Number of iRT peptides to generate.
        seed (int): Seed for random number generation.
        
    Returns:
        pd.DataFrame: Subsampled library DataFrame with selected iRT peptides.
    """
    random.seed(seed)

    # Filter out decoys
    target_library = library[library[lib_cols["DECOY_OR_NOT_COL"]] == 0]

    # number of layers
    n_layers = 10  
    
    # number of precursors at each layer
    n_each_layer = n_irts // n_layers  
    compensation = 0.2  # must be less than 0.25
    n_compen = int(n_each_layer * compensation)

    if n_layers < 4:
        n_samples_theoretical = [n_each_layer] * n_layers
    else:
        n_samples_theoretical = [n_each_layer + n_compen, n_each_layer + n_compen] + [n_each_layer - n_compen] * (n_layers - 4) + [n_each_layer + n_compen, n_each_layer + n_compen]
    
    # iRT range of each layer
    irt_min = target_library[lib_cols["IRT_COL"]].min()
    irt_max = target_library[lib_cols["IRT_COL"]].max()
    each_layer = (irt_max - irt_min) / n_layers

    # Define the layers
    layers = [
        (irt_min + each_layer * i, irt_min + each_layer * (i + 1)) 
        for i in range(n_layers)
    ]
     
    # Subsample library for each layer
    lib_part = [
        target_library[(target_library[lib_cols["IRT_COL"]] > layer[0]) & 
                       (target_library[lib_cols["IRT_COL"]] < layer[1])]
        for layer in layers
    ]
    
    # Number of precursors available and to be sampled in each layer
    counts = [len(np.unique(part[lib_cols["PRECURSOR_ID_COL"]])) for part in lib_part]
    n_samples = [min(i, j) for i, j in zip(n_samples_theoretical, counts)]
    
    # Select the precursors
    precursor_chosen = []
    for lib, n in zip(lib_part, n_samples):
        precursor_chosen.extend(
            random.sample(list(np.unique(lib[lib_cols["PRECURSOR_ID_COL"]])), n)
        )

    # Return the subsampled library
    irt_library = target_library[target_library[lib_cols["PRECURSOR_ID_COL"]].isin(precursor_chosen)]

    return irt_library

class IRT_Precursor:
    def __init__(self, precursor_id, full_sequence, charge, precursor_mz, iRT, protein_name):
        self.precursor_id = precursor_id
        self.full_sequence = full_sequence
        self.sequence = re.sub(r"\(UniMod:\d+\)", "", full_sequence)
        self.charge = charge
        self.precursor_mz = precursor_mz
        self.iRT = iRT
        self.protein_name = protein_name 
    
    def __eq__(self, obj):
        return (self.full_sequence == obj.full_sequence) and (self.charge == obj.charge)
    
    def filter_frags(self, frag_list, mz_min, mz_max, padding = False, padding_value = -1):
        if padding:
            return list(map(lambda x : x if (mz_min <= x < mz_max) else padding_value, frag_list))
        return [i for i in frag_list if mz_min <= i < mz_max]
    
    def __calc_self_frags(self, mz_min, mz_max):
        self.self_frags, self.self_frag_charges, self.self_frag_series = calc_all_fragment_mzs(self.full_sequence, 
                                                                                               self.charge, 
                                                                                               (mz_min, mz_max),  
                                                                                               return_annotations = True)
    
    def __calc_qt3_frags(self, mz_max, iso_range):
        iso_shift_max = int(min(iso_range, (mz_max - self.precursor_mz) * self.charge)) + 1
        self.qt3_frags = [self.precursor_mz + iso_shift / self.charge for iso_shift in range(iso_shift_max)]
    
    def __calc_lib_frags(self, frag_mz_list, frag_charge_list, frag_intensity_list, mz_min, mz_max):
        valid_fragment_indice = [i for i, frag in enumerate(frag_mz_list) if mz_min <= frag < mz_max]
        self.lib_frags = [frag_mz_list[i] for i in valid_fragment_indice]
        self.lib_frag_charges = [frag_charge_list[i] for i in valid_fragment_indice]
        self.lib_intensities = [frag_intensity_list[i] for i in valid_fragment_indice]
    
    def __calc_iso_frags(self, mz_min, mz_max):
        self.iso_frags = self.filter_frags([mz + 1 / c for mz, c in zip(self.lib_frags, self.lib_frag_charges)], 
                                           mz_min, mz_max, padding = True)
    
    def __calc_light_frags(self, mz_min, mz_max):
        self.light_frags = self.filter_frags([mz - 1 / c for mz, c in zip(self.lib_frags, self.lib_frag_charges)], 
                                             mz_min, mz_max, padding = True)
    
    def calc_frags(self, frag_mz_list, frag_charge_list, frag_intensity_list, mz_min, mz_max, iso_range):
        self.__calc_self_frags(mz_min, mz_max)
        self.__calc_qt3_frags(mz_max, iso_range)
        self.__calc_lib_frags(frag_mz_list, frag_charge_list, frag_intensity_list, mz_min, mz_max)
        self.__calc_iso_frags(mz_min, mz_max)
        self.__calc_light_frags(mz_min, mz_max)

def load_irt_precursors(irt_library, lib_cols, mz_min, mz_max, iso_range, n_threads):
    """
    Load iRT precursors from the given iRT library, calculate their fragments and tear the iRT library into chunks.

    Parameters:
    - irt_library (pd.DataFrame): The iRT library containing precursor information.
    - lib_cols (dict): A dictionary mapping column types to their respective column names in the library.
    - mz_min (float): Minimum m/z value for filtering fragments.
    - mz_max (float): Maximum m/z value for filtering fragments.
    - iso_range (float): Isotope range for calculating QT3 fragments.
    - n_threads (int): Number of threads for parallel processing.

    Returns:
    - irt_precursors (list): List of IRT_Precursor objects with calculated fragments.
    - chunk_indices (list): List of chunk indices for parallel processing.
    """

    irt_precursors = []
    precursor_ids = irt_library[lib_cols["PRECURSOR_ID_COL"]].unique()

    for precursor in precursor_ids:
        library_part = irt_library[irt_library[lib_cols["PRECURSOR_ID_COL"]] == precursor]
        precursor_obj = IRT_Precursor(
            list(library_part.loc[:, lib_cols["PRECURSOR_ID_COL"]])[0], 
            list(library_part.loc[:, lib_cols["FULL_SEQUENCE_COL"]])[0], 
            list(library_part.loc[:, lib_cols["PRECURSOR_CHARGE_COL"]])[0], 
            list(library_part.loc[:, lib_cols["PRECURSOR_MZ_COL"]])[0], 
            list(library_part.loc[:, lib_cols["IRT_COL"]])[0], 
            list(library_part.loc[:, lib_cols["PROTEIN_NAME_COL"]])[0]
        )
        precursor_obj.calc_frags(
            list(library_part[lib_cols["FRAGMENT_MZ_COL"]]), 
            list(library_part[lib_cols["FRAGMENT_CHARGE_COL"]]),
            list(library_part[lib_cols["LIB_INTENSITY_COL"]]), 
            mz_min, mz_max, iso_range
        )
        irt_precursors.append(precursor_obj)

    n_precursors = len(irt_precursors)
    n_each_chunk = n_precursors // n_threads
    chunk_indices = [[k + i * n_each_chunk for k in range(n_each_chunk)] for i in range(n_threads)]
    
    for i, idx in enumerate(range(chunk_indices[-1][-1] + 1, n_precursors)):
        chunk_indices[i].append(idx)
    
    return irt_precursors, chunk_indices

def get_rt_norm_searching_space(rt_list, irt, k = 40, b = 2000, irt_searching_cycles = 200):       
    central_rt = irt * k + b
    return find_rt_pos(central_rt, rt_list, irt_searching_cycles)

def build_irt_RSMs(precursor, ms1, ms2, win_range, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, model_cycles, iso_range,
                   apex_indices, feature_dimension, irt_k, irt_b, irt_searching_cycles, 
                   n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags): 
    precursor_win_id = calc_win_id(precursor.precursor_mz, win_range)
    
    rt_pos_ms1 = get_rt_norm_searching_space(ms1.rt_list, precursor.iRT, irt_k, irt_b, irt_searching_cycles)
    rt_pos_ms2 = get_rt_norm_searching_space(ms2[precursor_win_id].rt_list, precursor.iRT, irt_k, irt_b, irt_searching_cycles) 
    
    precursor_rt_list = [ms1.rt_list[i] for i in rt_pos_ms1]
    precursor_ms1_spectra = [ms1.spectra[i] for i in rt_pos_ms1]
    precursor_ms2_spectra = [ms2[precursor_win_id].spectra[i] for i in rt_pos_ms2]
    
    lib_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.lib_frags])
    lib_xics_1 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.2 * mz_tol_ms2) for frag in precursor.lib_frags])
    lib_xics_2 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.45 * mz_tol_ms2) for frag in precursor.lib_frags])
    self_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.self_frags])
    qt3_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.qt3_frags])
    
    if self_xics.shape[1] < 1 or qt3_xics.shape[1] < 1:
        return
    
    ms1_xics = [calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, mz_tol_ms1), 
                calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, 0.2 * mz_tol_ms1), 
                calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, 0.45 * mz_tol_ms1)]
    ms1_iso_frags = [precursor.precursor_mz - 1 / precursor.charge] + [precursor.precursor_mz + iso_shift / precursor.charge for iso_shift in range(1, iso_range + 1)]
    ms1_iso_frags = [i for i in ms1_iso_frags if mz_min <= i < mz_max]
    ms1_xics.extend([calc_XIC(precursor_ms1_spectra, frag, mz_unit, mz_tol_ms1) for frag in ms1_iso_frags])
    ms1_xics = np.array(ms1_xics)
    
    iso_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.iso_frags])
    light_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.light_frags])
    
    precursor_matrices, middle_rt_list = [], []
    lib_cos_scores = []
    
    for start_cycle in range(irt_searching_cycles - model_cycles + 1):
        end_cycle = start_cycle + model_cycles
        middle_rt_list.append(precursor_rt_list[start_cycle + model_cycles // 2])
        lib_matrix = lib_xics[:, start_cycle : end_cycle]
        lib_matrix_1 = lib_xics_1[:, start_cycle : end_cycle]
        lib_matrix_2 = lib_xics_2[:, start_cycle : end_cycle]
        self_matrix = self_xics[:, start_cycle : end_cycle]
        qt3_matrix = qt3_xics[:, start_cycle : end_cycle]
        ms1_matrix = ms1_xics[:, start_cycle : end_cycle]
        iso_matrix = iso_xics[:, start_cycle : end_cycle]
        light_matrix = light_xics[:, start_cycle : end_cycle]
        
        try:
            self_matrix = filter_matrix(self_matrix)
        except:
            return
        
        qt3_matrix = filter_matrix(qt3_matrix)
        lib_matrix = tools.smooth_array(lib_matrix.astype(float))
        lib_matrix_1 = tools.smooth_array(lib_matrix_1.astype(float))
        lib_matrix_2 = tools.smooth_array(lib_matrix_2.astype(float))
        self_matrix = tools.smooth_array(self_matrix.astype(float))
        qt3_matrix = tools.smooth_array(qt3_matrix.astype(float))
        ms1_matrix = tools.smooth_array(ms1_matrix.astype(float))
        iso_matrix = tools.smooth_array(iso_matrix.astype(float))
        light_matrix = tools.smooth_array(light_matrix.astype(float))
        
        apex_intensities = lib_matrix[:, apex_indices].mean(axis = 1)
        if sum(apex_intensities) != 0:
            lib_cos_scores.append(1 - spatial.distance.cosine(apex_intensities, precursor.lib_intensities))
        else:
            lib_cos_scores.append(0)
        
        if lib_matrix.shape[0] > 0:
            std_indice, pearson_sums = calc_pearson_sums(lib_matrix)
            sort_order = np.argsort(-np.array(pearson_sums))
            lib_matrix = lib_matrix[sort_order, :]
            lib_matrix_1 = lib_matrix_1[sort_order, :]
            lib_matrix_2 = lib_matrix_2[sort_order, :]
            iso_matrix = iso_matrix[sort_order, :]
            light_matrix = light_matrix[sort_order, :]
            if self_matrix.shape[0] > 1 and len(std_indice) >= 1:
                self_pearson = np.array([tools.calc_pearson(self_matrix[i, :], lib_matrix[0, :]) for i in range(self_matrix.shape[0])])
                self_matrix = self_matrix[np.argsort(-self_pearson), :]
            if qt3_matrix.shape[0] > 1 and len(std_indice) >= 1:
                qt3_pearson = np.array([tools.calc_pearson(qt3_matrix[i, :], lib_matrix[0, :]) for i in range(qt3_matrix.shape[0])])
                qt3_matrix = qt3_matrix[np.argsort(-qt3_pearson), :]
        
        lib_matrix = adjust_size(lib_matrix, n_lib_frags)
        lib_matrix_1 = adjust_size(lib_matrix_1, n_lib_frags)
        lib_matrix_2 = adjust_size(lib_matrix_2, n_lib_frags)
        self_matrix = adjust_size(self_matrix, n_self_frags)
        qt3_matrix = adjust_size(qt3_matrix, n_qt3_frags)
        ms1_matrix = adjust_size(ms1_matrix, n_ms1_frags)
        iso_matrix = adjust_size(iso_matrix, n_iso_frags)
        light_matrix = adjust_size(light_matrix, n_light_frags)
        
        training_matrix = np.zeros((feature_dimension, model_cycles))
        
        part1_indice = (0, 
                        lib_matrix.shape[0])
        part2_indice = (n_lib_frags, 
                        n_lib_frags + self_matrix.shape[0])
        part3_indice = (n_lib_frags + n_self_frags, 
                        n_lib_frags + n_self_frags + qt3_matrix.shape[0])
        part4_indice = (n_lib_frags + n_self_frags + n_qt3_frags, 
                        n_lib_frags + n_self_frags + n_qt3_frags + ms1_matrix.shape[0])
        part5_indice = (n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags, 
                        n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + iso_matrix.shape[0])
        part6_indice = (n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags, 
                        n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + light_matrix.shape[0])
        part7_indice = (n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags, 
                        n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags + lib_matrix_1.shape[0])
        part8_indice = (n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags + n_lib_frags, 
                        n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags + n_lib_frags + lib_matrix_2.shape[0])

        if lib_matrix.shape[1] != model_cycles:
            lib_matrix = adjust_cycle(lib_matrix, model_cycles)
        if self_matrix.shape[1] != model_cycles:
            self_matrix = adjust_cycle(self_matrix, model_cycles)
        if qt3_matrix.shape[1] != model_cycles:
            qt3_matrix = adjust_cycle(qt3_matrix, model_cycles)
        if ms1_matrix.shape[1] != model_cycles:
            ms1_matrix = adjust_cycle(ms1_matrix, model_cycles)
        if iso_matrix.shape[1] != model_cycles:
            iso_matrix = adjust_cycle(iso_matrix, model_cycles)
        if light_matrix.shape[1] != model_cycles:
            light_matrix = adjust_cycle(light_matrix, model_cycles)
        if lib_matrix_1.shape[1] != model_cycles:
            lib_matrix_1 = adjust_cycle(lib_matrix_1, model_cycles)
        if lib_matrix_2.shape[1] != model_cycles:
            lib_matrix_2 = adjust_cycle(lib_matrix_2, model_cycles)

        training_matrix[part1_indice[0] : part1_indice[1], :] = lib_matrix
        training_matrix[part2_indice[0] : part2_indice[1], :] = self_matrix
        training_matrix[part3_indice[0] : part3_indice[1], :] = qt3_matrix
        training_matrix[part4_indice[0] : part4_indice[1], :] = ms1_matrix
        training_matrix[part5_indice[0] : part5_indice[1], :] = iso_matrix
        training_matrix[part6_indice[0] : part6_indice[1], :] = light_matrix
        training_matrix[part7_indice[0] : part7_indice[1], :] = lib_matrix_1
        training_matrix[part8_indice[0] : part8_indice[1], :] = lib_matrix_2

        training_matrix = training_matrix.T
        training_matrix = MinMaxScaler().fit_transform(training_matrix)
        
        precursor_matrices.append(training_matrix)

    return middle_rt_list, np.array(precursor_matrices), lib_cos_scores

def set_rt_searching_range(irt_mode):
    if irt_mode == "irt":
        irt_k, irt_b = 40, 2000
    else:
        irt_k, irt_b = 1, 0
    return irt_k, irt_b

def score_irt(irt_score_res, precursor_list, ms1, ms2, win_range, irt_mode, 
              model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
              n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, 
              BM_model_file, feature_dimension, apex_indices, irt_searching_cycles):
    """
    Score iRT values for a list of precursors using a trained BM model.

    Parameters:
    - irt_score_res (list): List to append the scoring results.
    - precursor_list (list): List of precursor objects to score.
    - ms1, ms2: Mass spectrometry data.
    - win_range (tuple): Window range for the data.
    - irt_mode (str): Mode for iRT calculation.
    - model_cycles (int): Number of cycles for the model.
    - mz_unit (str): Unit for m/z values.
    - mz_min, mz_max (float): Minimum and maximum m/z values.
    - mz_tol_ms1, mz_tol_ms2 (float): Tolerance values for ms1 and ms2.
    - iso_range (float): Isotope range.
    - n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags (int): Number of fragments.
    - BM_model_file (str): File path for the BM model.
    - feature_dimension (int): Dimension of the feature space.
    - apex_indices (list): List of apex indices.
    - irt_searching_cycles (int): Number of retention time cycles for iRT searching.

    Returns:
    - None: Results are appended to irt_score_res.
    """

    irt_k, irt_b = set_rt_searching_range(irt_mode)
    
    set_gpu_memory()  
    BM_model = load_model(BM_model_file, compile = False)
    BM_model.call = tf.function(BM_model.call, experimental_relax_shapes = True)
    
    irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores = [], [], [], [], []
    
    for precursor in precursor_list:
        precursor_scoring_data = build_irt_RSMs(precursor, ms1, ms2, win_range, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, 
                                                model_cycles, iso_range, apex_indices, feature_dimension, irt_k, irt_b, irt_searching_cycles, 
                                                n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags)

        if precursor_scoring_data is None:
            continue
        else:
            middle_rt_list, precursor_matrices, lib_cos_scores = precursor_scoring_data

        if precursor_matrices.shape[0] > 0:
            scores = BM_model(precursor_matrices, training = False).numpy().T[0]
            max_index = np.argmax(scores)
            
            irt_precursor_ids.append(precursor.precursor_id)
            irt_values.append(precursor.iRT)
            rt_no1_values.append(middle_rt_list[max_index])
            irt_dream_scores.append(scores[max_index])
            irt_lib_cos_scores.append(lib_cos_scores[max_index])
    
    irt_score_res.append([irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores])

def collapse_irt_score_results(irt_score_res_list):
    irt_precursor_ids = flatten_list([i[0][0] for i in irt_score_res_list])
    irt_values = flatten_list([i[0][1] for i in irt_score_res_list])
    rt_no1_values = flatten_list([i[0][2] for i in irt_score_res_list])
    irt_dream_scores = flatten_list([i[0][3] for i in irt_score_res_list])
    irt_lib_cos_scores = flatten_list([i[0][4] for i in irt_score_res_list])

    return irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores

def pick_high_confidence_irt_precursors(irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores, 
                                        irt_mode, irt_score_cutoff, irt_score_libcos_cutoff, 
                                        n_picked_irt_thresh = 150, n_param_tuning_epochs = 5, irt_compensation = 0.02, score_decay_coef = 0.95):
    picked_irt_indices = []
    param_tuning_epoch_now = 0
    
    while (len(picked_irt_indices) < n_picked_irt_thresh) and (param_tuning_epoch_now < n_param_tuning_epochs):
        picked_irt_indices = []
        
        for idx in range(len(irt_precursor_ids)):
            if (irt_mode == "irt") and (irt_values[idx] < -25 or irt_values[idx] > 150):
                if irt_dream_scores[idx] >= irt_score_cutoff * (1 - irt_compensation) and irt_lib_cos_scores[idx] >= irt_score_libcos_cutoff * (1 - irt_compensation):
                    picked_irt_indices.append(idx)
            else:
                if irt_dream_scores[idx] >= irt_score_cutoff and irt_lib_cos_scores[idx] >= irt_score_libcos_cutoff:
                    picked_irt_indices.append(idx)
        
        param_tuning_epoch_now += 1
        irt_score_cutoff *= score_decay_coef
        irt_score_libcos_cutoff *= score_decay_coef
        
    irt_precursor_ids = [irt_precursor_ids[idx] for idx in picked_irt_indices]
    irt_values = [irt_values[idx] for idx in picked_irt_indices]
    rt_no1_values = [rt_no1_values[idx] for idx in picked_irt_indices]
    irt_dream_scores = [irt_dream_scores[idx] for idx in picked_irt_indices]
    irt_lib_cos_scores = [irt_lib_cos_scores[idx] for idx in picked_irt_indices]

    return irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores

def fit_irt_model(irt_score_res_list, rt_norm_dir, irt_score_cutoff, irt_score_libcos_cutoff, irt_mode, rt_norm_model, seed):
    if not os.path.exists(rt_norm_dir):
        os.mkdir(rt_norm_dir)

    irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores = collapse_irt_score_results(irt_score_res_list)

    irt_compensation = 0.02    
    n_picked_irt_thresh = 150
    n_param_tuning_epochs = 5
    score_decay_coef = 0.95

    irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores = pick_high_confidence_irt_precursors(irt_precursor_ids, irt_values, rt_no1_values, irt_dream_scores, irt_lib_cos_scores, 
                                                                                                                             irt_mode, irt_score_cutoff, irt_score_libcos_cutoff, 
                                                                                                                             n_picked_irt_thresh, n_param_tuning_epochs, irt_compensation, score_decay_coef)

    irt_data = pd.DataFrame({"irt_values" : irt_values, 
                             "rt_no1_values" : rt_no1_values, 
                             "irt_precursor_ids" : irt_precursor_ids, 
                             "irt_dream_scores" : irt_dream_scores, 
                             "irt_lib_cos_scores" : irt_lib_cos_scores})
    irt_data = irt_data.sort_values(by = "irt_values")
    irt_data.to_csv(os.path.join(rt_norm_dir, "time_points.tsv"), sep = "\t", index = False, header = False)

    irt_precursor_ids = np.array(irt_precursor_ids)
    
    if rt_norm_model == "linear":
        lr_RAN = RANSACRegressor(LinearRegression(), random_state = seed)
        lr_RAN.fit(np.array(irt_values).reshape(-1, 1), rt_no1_values)
        new_lr = LinearRegression()
        new_lr.fit(np.array(irt_values).reshape(-1, 1)[lr_RAN.inlier_mask_], np.array(rt_no1_values)[lr_RAN.inlier_mask_])
        r2 = new_lr.score(np.array(irt_values).reshape(-1, 1)[lr_RAN.inlier_mask_], np.array(rt_no1_values)[lr_RAN.inlier_mask_])
        slope, intercept = new_lr.coef_[0], new_lr.intercept_
        
        inliers = tukey_inliers(np.abs(np.array(irt_values) * slope + intercept - np.array(rt_no1_values)))        
        inlier_ids = irt_precursor_ids[inliers]
        
        with open(os.path.join(rt_norm_dir, "linear_irt_model.txt"), "w") as f:
            f.write("%s\n" % slope)
            f.write("%s\n" % intercept)
            f.write("%s\n" % r2)
        
        with open(os.path.join(rt_norm_dir, "preidentified_ids.txt"), "w") as f:
            f.writelines("%s\n" % pre_id for pre_id in inlier_ids)

        line_X = np.arange(min(irt_values) - 2, max(irt_values) + 2)
        line_y = new_lr.predict(line_X[:, np.newaxis])
        plt.figure(figsize = (6, 6))
        plt.scatter(irt_values, rt_no1_values)
        plt.plot(line_X, line_y, c = "orange")
        plt.xlabel("iRT")
        plt.ylabel("RT by DreamDIA")
        plt.title("DreamDIA RT normalization, $R^2 = $%.5f" % r2)
        plt.savefig(os.path.join(rt_norm_dir, "irt_model.pdf"))
        
        return [slope, intercept]
    
    elif rt_norm_model == "nonlinear":
        smoothed_points = sm.nonparametric.lowess(np.array(rt_no1_values), np.array(irt_values), frac = 0.15, it = 3, delta = 0)
        lowess_x = list(zip(*smoothed_points))[0]
        lowess_y = list(zip(*smoothed_points))[1]
        
        with open(os.path.join(rt_norm_dir, "nonlinear_irt_model.txt"), "w") as f:
            f.write("\n".join(["%s\t%s" % (x, y) for x, y in zip(lowess_x, lowess_y)]))
            f.write("\n")
        
        interpolate_function = interp1d(lowess_x, lowess_y, bounds_error = False, fill_value = "extrapolate")
        predicted_y = interpolate_function(np.array(irt_values))
        
        inliers = tukey_inliers(np.abs(predicted_y - np.array(rt_no1_values)))      
        inlier_ids = irt_precursor_ids[inliers]
        
        with open(os.path.join(rt_norm_dir, "preidentified_ids.txt"), "w") as f:
            f.writelines("%s\n" % pre_id for pre_id in inlier_ids)
            
        line_X = np.arange(min(irt_values) - 2, max(irt_values) + 2)
        line_y = interpolate_function(line_X)
        plt.figure(figsize = (6, 6))
        plt.scatter(irt_values, rt_no1_values)
        plt.plot(line_X, line_y, c = "orange")
        plt.xlabel("iRT")
        plt.ylabel("RT by DreamDIA")
        plt.title("DreamDIA RT normalization, Lowess")
        plt.savefig(os.path.join(rt_norm_dir, "irt_model.pdf"))
        
        return interpolate_function
    
    else:
        calib_rt_model = Calib_RT() 
        calib_rt_model.fit(np.array(irt_values), np.array(rt_no1_values))
        predicted_y = calib_rt_model.predict(np.array(irt_values))  

        with open(os.path.join(rt_norm_dir, "calib_rt_model.pkl"), "wb") as f:
            pickle.dump(calib_rt_model, f)
        
        inliers = tukey_inliers(np.abs(predicted_y - np.array(rt_no1_values)))      
        inlier_ids = irt_precursor_ids[inliers]
        
        with open(os.path.join(rt_norm_dir, "preidentified_ids.txt"), "w") as f:
            f.writelines("%s\n" % pre_id for pre_id in inlier_ids)
            
        line_X = np.arange(min(irt_values) - 2, max(irt_values) + 2)
        line_y = calib_rt_model.predict(line_X)
        plt.figure(figsize = (6, 6))
        plt.scatter(irt_values, rt_no1_values)
        plt.plot(line_X, line_y, c = "orange")
        plt.xlabel("iRT")
        plt.ylabel("RT by DreamDIA")
        plt.title("DreamDIA RT normalization, Calib-RT")
        plt.savefig(os.path.join(rt_norm_dir, "irt_model.pdf"))
        
        return calib_rt_model