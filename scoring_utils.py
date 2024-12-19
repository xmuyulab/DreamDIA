"""
╔═════════════════════════════════════════════════════╗
║                   scoring_utils.py                  ║
╠═════════════════════════════════════════════════════╣
║     Description: Utility functions for signal       ║
║                      scoring                        ║
╠═════════════════════════════════════════════════════╣
║         Author: Mingxuan Gao, Wenxian Yang          ║
║         Contact: mingxuan.gao@utoronto.ca           ║
╚═════════════════════════════════════════════════════╝
"""

import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import load_model

from mz_calculator import calc_all_fragment_mzs
from utils import calc_win_id, find_rt_pos, calc_XIC, filter_matrix, calc_pearson_sums, adjust_size, adjust_cycle, normalize_single_trace
import tools_cython as tools
from gpu_settings import set_gpu_memory
from file_io import Scoring_profile_cacher, compress_1d_array, init_sqdream, insert_chroms_batch, insert_ipf_scores_batch
from openswath_scoring import calculate_xcorr_scores, calculate_emg_scores

class Lib_frag:
    def __init__(self, mz, charge, fragtype, series, intensity):
        self.__mz = mz
        self.__charge = charge
        self.__fragtype = fragtype
        self.__series = series
        self.__intensity = intensity
    
    @property  
    def mz(self):
        return self.__mz
    
    @property  
    def charge(self):
        return self.__charge  
    
    @property  
    def intensity(self):
        return self.__intensity   
    
    @property
    def description(self):
        return "{0}_{1}_{2}_{3}_{4}".format(self.__fragtype, self.__series, self.__charge, self.__mz, self.__intensity)

class Precursor:
    def __init__(self, precursor_id, full_sequence, sequence, charge, precursor_mz, iRT, protein_name, decoy, 
                 mz_min, mz_max, iso_range, 
                 frag_mz_list, frag_charge_list, frag_type_list, frag_series_list, frag_intensity_list):
        """
        Three types of input needed to initiate a Precursor object:
        1. static attributes: 8
            (precursor_id, full_sequence, sequence, charge, precursor_mz, iRT, protein_name, decoy)
        2. XIC extraction parameters: 3
           (mz_min, mz_max, iso_range)
        3. library fragment ion properties: 5
           (mz, charge, type, series, intensity)
        """

        self.precursor_id = precursor_id
        self.full_sequence = full_sequence
        self.sequence = sequence
        self.charge = charge
        self.precursor_mz = precursor_mz
        self.iRT = iRT
        self.protein_name = protein_name
        self.decoy = decoy
        
        # Use the library fragment ion properties to build Lib_frag objects.
        self.lib_frags = [
            Lib_frag(mz, charge, fragtype, series, inten) 
            for mz, charge, fragtype, series, inten in zip(frag_mz_list, frag_charge_list, frag_type_list, frag_series_list, frag_intensity_list)
        ]
        self.lib_intensities = np.array(frag_intensity_list)
        self.lib_frag_mzs = np.array(frag_mz_list)
        self.lib_frag_series = [
            f"{fragtype}{series}_{charge}" 
            for fragtype, series, charge in zip(frag_type_list, frag_series_list, frag_charge_list)
        ]

        # Calculate self fragment ions
        self.self_frags, self.self_frag_charges, self.self_frag_series = calc_all_fragment_mzs(
            self.full_sequence, self.charge, (mz_min, mz_max), return_annotations = True
        )

        # Calculate qt3, iso, light fragment ions
        iso_shift_max = int(min(iso_range, (mz_max - self.precursor_mz) * self.charge)) + 1
        
        self.qt3_frags = [self.precursor_mz + iso_shift / self.charge for iso_shift in range(iso_shift_max)]        
        self.iso_frags = self.filter_frags([i.mz + 1 / i.charge for i in self.lib_frags], mz_min, mz_max, padding = True)
        self.light_frags = self.filter_frags([i.mz - 1 / i.charge for i in self.lib_frags], mz_min, mz_max, padding = True)

    def filter_frags(self, frag_list, mz_min, mz_max, padding = False, padding_value = -1):
        if padding:
            return list(map(lambda x : x if (mz_min <= x < mz_max) else padding_value, frag_list))
        return [i for i in frag_list if mz_min <= i < mz_max]

    def get_static_info(self):
        return {
            "precursor_id" : self.precursor_id, 
            "full_sequence" : self.full_sequence, 
            "sequence" : self.sequence, 
            "charge" : self.charge, 
            "precursor_mz" : self.precursor_mz, 
            "iRT" : self.iRT, 
            "protein_name" : self.protein_name, 
            "decoy" : self.decoy
        }
    
    def __eq__(self, obj):
        return (self.full_sequence == obj.full_sequence) and (self.charge == obj.charge)
    
    def __str__(self):
        return self.full_sequence + "_" + str(self.charge)    
    
    def __repr__(self):
        return self.full_sequence + "_" + str(self.charge)

def load_precursors(library, lib_cols, precursor_index, precursor_list, mz_min, mz_max, iso_range):
    """
    Load precursors from the library and append them to the precursor list.

    Parameters:
    - library (pd.DataFrame): The data frame containing peptide library records.
    - lib_cols (dict): A dictionary mapping column types to their respective column names in the library.
    - precursor_index (list): List of indices identifying precursors in the library to be processed.
    - precursor_list (list): List to append the created Precursor objects.
    - mz_min (float): Minimum m/z value for filtering.
    - mz_max (float): Maximum m/z value for filtering.
    - iso_range (float): Isotope range for calculating QT3 fragments.
    """

    for idx in precursor_index:
        library_part = library.iloc[idx, :]
        precursor_obj = Precursor(
            library_part[lib_cols["PRECURSOR_ID_COL"]].values[0], 
            library_part[lib_cols["FULL_SEQUENCE_COL"]].values[0], 
            library_part[lib_cols["PURE_SEQUENCE_COL"]].values[0], 
            library_part[lib_cols["PRECURSOR_CHARGE_COL"]].values[0], 
            library_part[lib_cols["PRECURSOR_MZ_COL"]].values[0], 
            library_part[lib_cols["IRT_COL"]].values[0], 
            library_part[lib_cols["PROTEIN_NAME_COL"]].values[0], 
            library_part[lib_cols["DECOY_OR_NOT_COL"]].values[0], 
            mz_min, mz_max, iso_range, 
            list(library_part[lib_cols["FRAGMENT_MZ_COL"]]), 
            list(library_part[lib_cols["FRAGMENT_CHARGE_COL"]]),
            list(library_part[lib_cols["FRAGMENT_TYPE_COL"]]), 
            list(library_part[lib_cols["FRAGMENT_SERIES_COL"]]), 
            list(library_part[lib_cols["LIB_INTENSITY_COL"]])
        )
        precursor_list.append(precursor_obj)

def set_RT(iRT, rt_norm_model, rt_model_params):
    if rt_norm_model == "linear":
        RT = iRT * rt_model_params[0] + rt_model_params[1]
    elif rt_norm_model == "nonlinear":
        RT = rt_model_params([iRT])[0]
    else:
        RT = rt_model_params.predict(np.array([iRT]))[0]
    return RT

def quantify(lib_pearsons, ms2_areas):
    """
    Quantify the MS2 signal based on Pearson correlation scores and MS2 areas.

    Parameters:
    - lib_pearsons (list of list of float): Pearson correlation scores for each time point.
    - ms2_areas (list of list of float): MS2 areas for each time point.

    Returns:
    - list of float: Quantified values for each time point.
    """

    quant = []
    for lib_pearsons_time_point, ms2_areas_time_point in zip(lib_pearsons, ms2_areas):
        n_frags_for_quant = len(lib_pearsons_time_point)
        
        # If no MS2 fragment XIC is available, set the quantity to 0.
        if n_frags_for_quant == 0:
            quant.append(0)
            continue

        # Denoise: only consider fragments with Pearson scores higher than `0.1 * n_frags_for_quant`.
        valid_index = np.where(np.array(lib_pearsons_time_point) > 0.1 * n_frags_for_quant)[0]

        if len(valid_index) <= 0:
            quant.append(0)
        else:
            valid_lib_pearsons_time_point = np.array(lib_pearsons_time_point)[valid_index]
            valid_ms2_areas_time_point = np.array(ms2_areas_time_point)[valid_index]
            quant_value = max(0, valid_lib_pearsons_time_point.dot(valid_ms2_areas_time_point) / len(valid_index))
            quant.append(quant_value)
    
    return quant 

def build_RSMs(precursor, ms1, ms2, win_range, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, model_cycles, iso_range, 
               n_cycles, rt_norm_model, rt_model_params, apex_indices, feature_dimension, 
               n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, ipf_scoring):
    RT = set_RT(precursor.iRT, rt_norm_model, rt_model_params)
    precursor_win_id = calc_win_id(precursor.precursor_mz, win_range)
    
    rt_pos_ms1 = find_rt_pos(RT, ms1.rt_list, n_cycles)
    rt_pos_ms2 = find_rt_pos(RT, ms2[precursor_win_id].rt_list, n_cycles)       
    precursor_rt_list = [ms1.rt_list[i] for i in rt_pos_ms1]
    precursor_ms1_spectra = [ms1.spectra[i] for i in rt_pos_ms1]
    precursor_ms2_spectra = [ms2[precursor_win_id].spectra[i] for i in rt_pos_ms2]
    
    all_lib_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.lib_frag_mzs])
    all_lib_xics_1 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.2 * mz_tol_ms2) for frag in precursor.lib_frag_mzs])
    all_lib_xics_2 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.45 * mz_tol_ms2) for frag in precursor.lib_frag_mzs])
    all_iso_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.iso_frags])
    all_light_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.light_frags])
    all_self_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.self_frags])
    all_qt3_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.qt3_frags])
    all_ms1_xics = [calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, mz_tol_ms1), 
                    calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, 0.2 * mz_tol_ms1), 
                    calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, 0.45 * mz_tol_ms1)]
    ms1_iso_frags = [precursor.precursor_mz - 1 / precursor.charge] + [precursor.precursor_mz + iso_shift / precursor.charge for iso_shift in range(1, iso_range + 1)]
    ms1_iso_frags = [i for i in ms1_iso_frags if mz_min <= i < mz_max]
    all_ms1_xics.extend([calc_XIC(precursor_ms1_spectra, frag, mz_unit, mz_tol_ms1) for frag in ms1_iso_frags])
    all_ms1_xics = np.array(all_ms1_xics)
    
    rsm_info = {"orig_matrices" : [], 
                "matrices" : [], 
                "rt_lists" : [], 
                "middle_rts" : [], 
                "ms1_area_list" : [], 
                "ms2_area_list" : [], 
                "lib_frags_real_intensities" : [], 
                "lib_pearsons" : [], 
                "delta_rts" : [], 
                "lib_cos_scores" : [], 
                "norm_lib_cos_scores" : [], 
                "quantities" : []}

    if ipf_scoring:
        ipf_info = {"xcorr_scores" : [], 
                    "xcorr_shape_scores" : [], 
                    "emg_scores" : []}

    for rt_start in range(n_cycles - model_cycles + 1):
        rt_end = rt_start + model_cycles
        precursor_rt_list_part = precursor_rt_list[rt_start : rt_end]
        rsm_info["middle_rts"].append(precursor_rt_list_part[model_cycles // 2])
        rsm_info["rt_lists"].append(precursor_rt_list_part)
        lib_xics = all_lib_xics[:, rt_start : rt_end]
        lib_xics_1 = all_lib_xics_1[:, rt_start : rt_end]
        lib_xics_2 = all_lib_xics_2[:, rt_start : rt_end]
        self_xics = all_self_xics[:, rt_start : rt_end] 
        qt3_xics = all_qt3_xics[:, rt_start : rt_end]
        ms1_xics = all_ms1_xics[:, rt_start : rt_end]
        iso_xics = all_iso_xics[:, rt_start : rt_end]
        light_xics = all_light_xics[:, rt_start : rt_end]          
        
        # `smooth_array` will generate new XIC arrays for downstream operations in case the original full-length XIC array being modified.
        lib_xics = tools.smooth_array(lib_xics.astype(float))
        lib_xics_1 = tools.smooth_array(lib_xics_1.astype(float))
        lib_xics_2 = tools.smooth_array(lib_xics_2.astype(float))
        self_xics = tools.smooth_array(self_xics.astype(float))
        qt3_xics = tools.smooth_array(qt3_xics.astype(float))
        ms1_xics = tools.smooth_array(ms1_xics.astype(float))
        iso_xics = tools.smooth_array(iso_xics.astype(float))
        light_xics = tools.smooth_array(light_xics.astype(float))
        
        precursor_rt_list_part_diff = np.array(precursor_rt_list_part[1:]) - np.array(precursor_rt_list_part[:-1])
        
        # ms2 XIC area for each lib fragment ion through the middle rt 
        ms2_areas = [tools.calc_area(lib_xics[i, :], precursor_rt_list_part_diff) for i in range(lib_xics.shape[0])]
        # ms1 XIC area through the middle rt 
        ms1_area = tools.calc_area(ms1_xics[0, :], precursor_rt_list_part_diff)
        
        rsm_info["ms2_area_list"].append(ms2_areas)
        rsm_info["ms1_area_list"].append(ms1_area)

        if ipf_scoring:
            mean_xcorr_scores, mean_xcorr_shape_scores = calculate_xcorr_scores(self_xics)
            emg_scores = calculate_emg_scores(self_xics)
            ipf_info["xcorr_scores"].append(mean_xcorr_scores)
            ipf_info["xcorr_shape_scores"].append(mean_xcorr_shape_scores)
            ipf_info["emg_scores"].append(emg_scores)

        # get apex values of the chromatograms
        apex_intensities = lib_xics[:, apex_indices].mean(axis = 1)
        rsm_info["lib_frags_real_intensities"].append(apex_intensities)

        std_indice, pearson_sums = calc_pearson_sums(lib_xics)
        rsm_info["lib_pearsons"].append(pearson_sums)

        self_xics = filter_matrix(self_xics)
        qt3_xics = filter_matrix(qt3_xics)

        if lib_xics.shape[0] > 0:
            sort_order = np.argsort(-np.array(pearson_sums))
            lib_xics = lib_xics[sort_order, :]
            lib_xics_1 = lib_xics_1[sort_order, :]
            lib_xics_2 = lib_xics_2[sort_order, :]
            iso_xics = iso_xics[sort_order, :]
            light_xics = light_xics[sort_order, :]
            if self_xics.shape[0] > 1 and len(std_indice) >= 1:
                self_pearson = np.array([tools.calc_pearson(self_xics[i, :], lib_xics[0, :]) for i in range(self_xics.shape[0])])
                self_xics = self_xics[np.argsort(-self_pearson), :]
            if qt3_xics.shape[0] > 1 and len(std_indice) >= 1:
                qt3_pearson = np.array([tools.calc_pearson(qt3_xics[i, :], lib_xics[0, :]) for i in range(qt3_xics.shape[0])])
                qt3_xics = qt3_xics[np.argsort(-qt3_pearson), :]

        lib_matrix = adjust_size(lib_xics, n_lib_frags)
        lib_matrix_1 = adjust_size(lib_xics_1, n_lib_frags)
        lib_matrix_2 = adjust_size(lib_xics_2, n_lib_frags)
        self_matrix = adjust_size(self_xics, n_self_frags)
        qt3_matrix = adjust_size(qt3_xics, n_qt3_frags)
        ms1_matrix = adjust_size(ms1_xics, n_ms1_frags)
        iso_matrix = adjust_size(iso_xics, n_iso_frags)
        light_matrix = adjust_size(light_xics, n_light_frags)   
        training_matrix = np.zeros((feature_dimension, model_cycles))

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
        training_matrix[part1_indice[0] : part1_indice[1], :] = lib_matrix
        training_matrix[part2_indice[0] : part2_indice[1], :] = self_matrix
        training_matrix[part3_indice[0] : part3_indice[1], :] = qt3_matrix
        training_matrix[part4_indice[0] : part4_indice[1], :] = ms1_matrix
        training_matrix[part5_indice[0] : part5_indice[1], :] = iso_matrix
        training_matrix[part6_indice[0] : part6_indice[1], :] = light_matrix
        training_matrix[part7_indice[0] : part7_indice[1], :] = lib_matrix_1
        training_matrix[part8_indice[0] : part8_indice[1], :] = lib_matrix_2
        training_matrix = training_matrix.T
        
        rsm_info["orig_matrices"].append(training_matrix)
        training_matrix = MinMaxScaler().fit_transform(training_matrix)
        rsm_info["matrices"].append(training_matrix)

    # Calculate scoring profiles        
    rsm_info["delta_rts"] = np.abs(RT - np.array(rsm_info["middle_rts"]))
    rsm_info["lib_cos_scores"] = cosine_similarity(np.array(rsm_info["lib_frags_real_intensities"]), precursor.lib_intensities.reshape(1, -1))[:, 0]
    rsm_info["norm_lib_cos_scores"] = normalize_single_trace(rsm_info["lib_cos_scores"])

    # quantification
    rsm_info["quantities"] = quantify(rsm_info["lib_pearsons"], rsm_info["ms2_area_list"])

    if ipf_scoring:
        ipf_info["xcorr_scores"] = np.array(ipf_info["xcorr_scores"]).T
        ipf_info["xcorr_shape_scores"] = np.array(ipf_info["xcorr_shape_scores"]).T
        ipf_info["emg_scores"] = np.array(ipf_info["emg_scores"]).T
        
        return rsm_info, precursor_rt_list, all_lib_xics, all_ms1_xics, ipf_info
    
    return rsm_info, precursor_rt_list, all_lib_xics, all_ms1_xics

def score_precursors(ms1, ms2, win_range, precursor_list, chrom_queue, sp_queue, progress_queue, 
                     n_cycles, model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
                     n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, ipf_scoring, 
                     rt_norm_model, rt_model_params, BM_model_file, RM_model_file, apex_indices, feature_dimension): 
    #drf_dim = 16

    set_gpu_memory()  
    BM_model = load_model(BM_model_file, compile = False)
    RM_model = load_model(RM_model_file, compile = False)
    BM_model.call = tf.function(BM_model.call, experimental_relax_shapes = True)
    RM_model.call = tf.function(RM_model.call, experimental_relax_shapes = True)
    
    scoring_profile_cacher = Scoring_profile_cacher()

    #if mode == "single-run":
    #    score_list = ScoreList(drf_dim)
    
    for precursor in precursor_list: 
        precursor_rsm_info = build_RSMs(precursor, ms1, ms2, win_range, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, model_cycles, iso_range, 
                                        n_cycles, rt_norm_model, rt_model_params, apex_indices, feature_dimension, 
                                        n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, ipf_scoring)
        if ipf_scoring:
            rsm_info, precursor_rt_list, all_lib_xics, all_ms1_xics, ipf_info = precursor_rsm_info
        else:
            rsm_info, precursor_rt_list, all_lib_xics, all_ms1_xics = precursor_rsm_info

        dream_scores = BM_model(np.array(rsm_info["matrices"]), training = False).numpy().T[0]
        drf_scores = RM_model(np.array(rsm_info["matrices"]), training = False).numpy()
        
        #if mode == "single-run":
        #    score_list.append_precursor(precursor, dream_scores, rsm_info, drf_scores, top_k)

        scoring_profile_cacher.append_precursor(precursor, rsm_info, dream_scores)

        if ipf_scoring:    
            chrom_queue.put([precursor.precursor_id, 
                            precursor.lib_frag_series, 
                            compress_1d_array(precursor_rt_list), 
                            [compress_1d_array(frag) for frag in all_lib_xics], 
                            compress_1d_array(all_ms1_xics[0]), 
                            precursor.self_frag_charges, 
                            precursor.self_frag_series, 
                            [compress_1d_array(frag) for frag in ipf_info["xcorr_scores"]], 
                            [compress_1d_array(frag) for frag in ipf_info["xcorr_shape_scores"]], 
                            [compress_1d_array(frag) for frag in ipf_info["emg_scores"]]])
        else:
            chrom_queue.put([precursor.precursor_id, 
                            precursor.lib_frag_series, 
                            compress_1d_array(precursor_rt_list), 
                            [compress_1d_array(frag) for frag in all_lib_xics], 
                            compress_1d_array(all_ms1_xics[0])])
        
        progress_queue.put(1)
            
    sp_queue.put(scoring_profile_cacher.cacher)
    chrom_queue.put(None)

    #if mode == "single-run":
    #    score_list.format(rawdata_file)
    #    result_queue.put([score_list.score_list])

def output_chromatograms(chrom_queue, n_threads, rt_norm_dir, ipf_scoring, n_chrom_writting_batch, sqdream_file_name, logger):
    """
    Output RAW chromatograms.
    """
    
    sqdream_file = os.path.join(rt_norm_dir, sqdream_file_name)
    init_sqdream(sqdream_file)
    logger.info("sqDream file initiated: %s" % sqdream_file)

    n_none = 0
    chrom_cacher = []
    
    while 1:
        chromatogram_info = chrom_queue.get()
        if chromatogram_info is None:
            n_none += 1
            chrom_queue.task_done()
            if n_none >= n_threads:
                break
            else:
                continue
        
        chrom_cacher.append(chromatogram_info)    
        
        if (len(chrom_cacher) != 0) and (len(chrom_cacher) % n_chrom_writting_batch == 0):
            insert_chroms_batch(chrom_cacher, sqdream_file)
            if ipf_scoring:
                insert_ipf_scores_batch(chrom_cacher, sqdream_file)
            chrom_cacher = []
            
        chrom_queue.task_done()
            
    if len(chrom_cacher) != 0:
        insert_chroms_batch(chrom_cacher, sqdream_file)
        if ipf_scoring:
            insert_ipf_scores_batch(chrom_cacher, sqdream_file)
        chrom_cacher = []
            
def output_scoring_profiles(sp_queue, n_threads, rt_norm_dir):
    sqdream_file = os.path.join(rt_norm_dir, "dreamdia_scoring_profile.sqDream")
    
    sp_cachers = []
    
    while 1:
        sp_cacher = sp_queue.get()
        sp_cachers.append(sp_cacher)
        sp_queue.task_done()
        if len(sp_cachers) >= n_threads:
            break
    
    total_sp_cacher = Scoring_profile_cacher()
    for sp_cacher in sp_cachers:
        for item in sp_cacher:
            total_sp_cacher.cacher[item].extend(sp_cacher[item])
    
    total_sp_cacher.output(sqdream_file)    
        
def output_progress(progress_queue, n_precursors, n_output_progress_precursors, logger):
    progress = 0
    while 1:
        _ = progress_queue.get()
        progress_queue.task_done()
        progress += 1
        
        if progress and progress % n_output_progress_precursors == 0:
            logger.info("(%d / %d) precursors processed..." % (progress, n_precursors))
            
        if progress >= n_precursors:
            break
        