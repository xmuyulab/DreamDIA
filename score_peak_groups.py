import sys
import os
import re
import os.path
import bisect
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model

from utils import *
from mz_calculator import calc_all_fragment_mzs

class Lib_frag:
    def __init__(self, mz, charge, fragtype, series, intensity):
        self.__mz = mz
        self.__charge = charge
        self.__fragtype = fragtype
        self.__series = series
        self.__intensity = intensity
    
    def get_mz(self):
        return self.__mz
    
    def get_intensity(self):
        return self.__intensity
    
    def format_output(self):
        return "{0}_{1}_{2}_{3}_{4}".format(self.__fragtype, self.__series, self.__charge, self.__mz, self.__intensity)

class Precursor:
    def __init__(self, precursor_id, full_sequence, sequence, charge, precursor_mz, iRT, protein_name, decoy, 
                 mz_min, mz_max, iso_range, 
                 frag_mz_list, frag_charge_list, frag_type_list, frag_series_list, frag_intensity_list):
        self.precursor_id = precursor_id
        self.full_sequence = full_sequence
        self.sequence = sequence
        self.charge = charge
        self.precursor_mz = precursor_mz
        self.iRT = iRT
        self.RT = None
        self.protein_name = protein_name
        self.decoy = decoy

        self.ms1_areas = []
        self.ms2_areas = []
        self.lib_frags_real_intensities = []

        self.self_frags, self.self_frag_charges = np.array(calc_all_fragment_mzs(self.full_sequence, 
                                                                                 self.charge, 
                                                                                 (mz_min, mz_max),  
                                                                                 return_charges = True))
        
        iso_shift_max = int(min(iso_range, (mz_max - self.precursor_mz) * self.charge)) + 1
        self.qt3_frags = [self.precursor_mz + iso_shift / self.charge for iso_shift in range(iso_shift_max)]

        self.lib_frags = [Lib_frag(mz, charge, fragtype, series, inten) for mz, charge, fragtype, series, inten in zip(frag_mz_list, frag_charge_list, frag_type_list, frag_series_list, frag_intensity_list)]

    def set_RT(self, k, b):
        self.RT = self.iRT * k + b

    def clear(self):
        self.ms1_areas = []
        self.ms2_areas = []
        self.lib_frags_real_intensities = []
    
    def __eq__(self, obj):
        return (self.full_sequence == obj.full_sequence) and (self.charge == obj.charge)

    def __str__(self):
        return self.full_sequence + "_" + str(self.charge)
    
    def __repr__(self):
        return self.full_sequence + "_" + str(self.charge)

def load_precursors(library, lib_cols, precursor_index, precursor_list, mz_min, mz_max, iso_range):
    for idx in precursor_index:
        library_part = library.iloc[idx, :]
        precursor_obj = Precursor(list(library_part.loc[:, lib_cols["PRECURSOR_ID_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["FULL_SEQUENCE_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["PURE_SEQUENCE_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["PRECURSOR_CHARGE_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["PRECURSOR_MZ_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["IRT_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["PROTEIN_NAME_COL"]])[0], 
                                  list(library_part.loc[:, lib_cols["DECOY_OR_NOT_COL"]])[0], 
                                  mz_min, mz_max, iso_range, 
                                  list(library_part[lib_cols["FRAGMENT_MZ_COL"]]), 
                                  list(library_part[lib_cols["FRAGMENT_CHARGE_COL"]]),
                                  list(library_part[lib_cols["FRAGMENT_TYPE_COL"]]), 
                                  list(library_part[lib_cols["FRAGMENT_SERIES_COL"]]), 
                                  list(library_part[lib_cols["LIB_INTENSITY_COL"]]))
        precursor_list.append(precursor_obj)

def extract_precursors(ms1, ms2, win_range, precursor_list, matrix_queue, 
                       n_cycles, model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
                       n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, peak_index_range, slope, intercept, p_id): 
    peak_indice = get_peak_indice(model_cycles, peak_index_range)
    feature_dimension = n_lib_frags + n_self_frags + n_qt3_frags + n_ms1_frags

    for idx, precursor in enumerate(precursor_list):
        precursor.set_RT(slope, intercept)

        precursor_win_id = calc_win_id(precursor.precursor_mz, win_range)
        rt_pos_ms1 = find_rt_pos(precursor.RT, ms1.rt_list, n_cycles)
        rt_pos_ms2 = find_rt_pos(precursor.RT, ms2[precursor_win_id].rt_list, n_cycles)
        
        precursor_rt_list = [ms1.rt_list[i] for i in rt_pos_ms1]
        precursor_ms1_spectra = [ms1.spectra[i] for i in rt_pos_ms1]
        precursor_ms2_spectra = [ms2[precursor_win_id].spectra[i] for i in rt_pos_ms2]

        lib_frags = [frag.get_mz() for frag in precursor.lib_frags]
        all_lib_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in lib_frags])
        all_self_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.self_frags])
        all_qt3_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in precursor.qt3_frags])
        all_ms1_xics = [calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, mz_unit, mz_tol_ms1), 
                        calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, "Da", 0.2), 
                        calc_XIC(precursor_ms1_spectra, precursor.precursor_mz, "Da", 0.45)]
        ms1_iso_frags = [precursor.precursor_mz + iso_shift / precursor.charge for iso_shift in range(1, iso_range + 1)]
        ms1_iso_frags = [i for i in ms1_iso_frags if mz_min <= i < mz_max]
        all_ms1_xics.extend([calc_XIC(precursor_ms1_spectra, frag, mz_unit, mz_tol_ms1) for frag in ms1_iso_frags])
        all_ms1_xics = np.array(all_ms1_xics)

        matrices, middle_rts, rt_lists, lib_pearsons = [], [], [], []

        for rt_start in range(n_cycles - model_cycles + 1):
            rt_end = rt_start + model_cycles
            precursor_rt_list_part = precursor_rt_list[rt_start : rt_end]

            lib_xics = all_lib_xics[:, rt_start : rt_end]
            self_xics = all_self_xics[:, rt_start : rt_end]
            qt3_xics = all_qt3_xics[:, rt_start : rt_end]
            ms1_xics = all_ms1_xics[:, rt_start : rt_end]
            
            middle_rts.append(precursor_rt_list_part[model_cycles // 2])
            rt_lists.append(precursor_rt_list_part)
            
            self_xics = filter_matrix(self_xics)
            qt3_xics = filter_matrix(qt3_xics)

            lib_xics = smooth_array(lib_xics)
            self_xics = smooth_array(self_xics)
            qt3_xics = smooth_array(qt3_xics)
            ms1_xics = smooth_array(ms1_xics)

            # calculate areas
            precursor_rt_list_part_diff = np.array(precursor_rt_list_part[1:]) - np.array(precursor_rt_list_part[:-1])
            ms2_areas = [calc_area(lib_xics[i, :], precursor_rt_list_part_diff) for i in range(lib_xics.shape[0])]
            ms1_area = calc_area(ms1_xics[0, :], precursor_rt_list_part_diff)
            precursor.ms2_areas.append("|".join([str(each) for each in ms2_areas]))
            precursor.ms1_areas.append(str(ms1_area))

            # calculate real intensities of all library fragments
            peak_intensities = lib_xics[:, peak_indice].mean(axis = 1)
            precursor.lib_frags_real_intensities.append(peak_intensities)

            pearson_matrix = np.corrcoef(lib_xics)
            pearson_matrix[np.isnan(pearson_matrix)] = 0
            pearson_sums = pearson_matrix.sum(axis = 1)
            lib_pearsons.append(list(pearson_sums))
            lib_xics = lib_xics[np.argsort(-pearson_sums), :]

            if self_xics.shape[0] > 0:
                self_pearson = np.corrcoef(self_xics, lib_xics[0, :])
                self_pearson[np.isnan(self_pearson)] = 0 
                self_pearson = self_pearson[0, :][:-1]
                self_xics = self_xics[np.argsort(-self_pearson), :]
            if qt3_xics.shape[0] > 0:
                qt3_pearson = np.corrcoef(qt3_xics, lib_xics[0, :])
                qt3_pearson[np.isnan(qt3_pearson)] = 0 
                qt3_pearson = qt3_pearson[0, :][:-1]
                qt3_xics = qt3_xics[np.argsort(-qt3_pearson), :]

            lib_matrix = adjust_size(lib_xics, n_lib_frags)
            self_matrix = adjust_size(self_xics, n_self_frags)
            qt3_matrix = adjust_size(qt3_xics, n_qt3_frags)
            ms1_matrix = adjust_size(ms1_xics, n_ms1_frags)

            training_matrix = np.zeros((feature_dimension, model_cycles))
            part1_indice = (0, 
                            lib_matrix.shape[0])
            part2_indice = (n_lib_frags, 
                            n_lib_frags + self_matrix.shape[0])
            part3_indice = (n_lib_frags + n_self_frags, 
                            n_lib_frags + n_self_frags + qt3_matrix.shape[0])
            part4_indice = (n_lib_frags + n_self_frags + n_qt3_frags, 
                            n_lib_frags + n_self_frags + n_qt3_frags + ms1_matrix.shape[0])
            training_matrix[part1_indice[0] : part1_indice[1], :] = lib_matrix
            training_matrix[part2_indice[0] : part2_indice[1], :] = self_matrix
            training_matrix[part3_indice[0] : part3_indice[1], :] = qt3_matrix
            training_matrix[part4_indice[0] : part4_indice[1], :] = ms1_matrix
            training_matrix = training_matrix.T

            training_matrix = MinMaxScaler().fit_transform(training_matrix)

            matrices.append(training_matrix)

        matrix_queue.put([precursor, matrices, middle_rts, rt_lists, precursor_win_id, lib_pearsons])
   
    matrix_queue.put(None)
    #print("%d extractor done!" % p_id)

def score_batch(matrix_queue, lib_cols, BM_model_file, RM_model_file, out_file, rawdata_file, top_k, n_threads, batch_size, n_total_precursors, logger):
    BM_model = load_model(BM_model_file, compile = False)
    RM_model = load_model(RM_model_file, compile = False)
    BM_model.call = tf.function(BM_model.call, experimental_relax_shapes = True)
    RM_model.call = tf.function(RM_model.call, experimental_relax_shapes = True)

    out_f = open(out_file, "w")
    out_head_1 = "%s\tfilename\tRT\t%s\t%s\t" % (lib_cols["PRECURSOR_ID_COL"], lib_cols["PURE_SEQUENCE_COL"], lib_cols["FULL_SEQUENCE_COL"])
    out_head_2 = "%s\t%s\t%s\t%s\tassay_rt\tdelta_rt\t" % (lib_cols["PRECURSOR_CHARGE_COL"], lib_cols["PRECURSOR_MZ_COL"], lib_cols["PROTEIN_NAME_COL"], lib_cols["DECOY_OR_NOT_COL"])
    out_head_3 = "%s\tnr_peaks\treal_intensities\tlib_cos_scores\t" % lib_cols["IRT_COL"]
    out_head_4 = "dream_scores\tms1_area\tms2_areas\taggr_Fragment_Annotation\tlib_pearsons\taux_scores\n"
    out_f.write(out_head_1 + out_head_2 + out_head_3 + out_head_4)
    out_f.close()

    n_none = 0
    precursor_count = 0

    # save data in batch
    batch_precursor = []
    batch_matrices = []
    batch_middle_rts = []
    batch_lib_pearsons = []   

    while True:
        matrix_data = matrix_queue.get()
        if matrix_data is None:
            n_none += 1
            matrix_queue.task_done()
            if n_none >= n_threads:
                break
            else:
                continue
            
        precursor_count += 1

        precursor, matrices, middle_rts, rt_lists, precursor_win_id, lib_pearsons = matrix_data

        batch_precursor.append(precursor)
        batch_matrices.append(matrices)        
        batch_middle_rts.append(middle_rts)
        batch_lib_pearsons.append(lib_pearsons)

        matrix_queue.task_done()

        if precursor_count % 10000 == 0:
            logger.info("(%d / %d) precursors processed ..." % (precursor_count, n_total_precursors))

        if precursor_count % batch_size == 0:
            batch_results = []
            batch_matrices = np.concatenate(batch_matrices)
            batch_scores = BM_model(batch_matrices, training = False).numpy().T[0]
            batch_aux_scores = RM_model(batch_matrices, training = False).numpy()

            batch_scores = np.split(batch_scores, batch_size)           
            batch_aux_scores = np.split(batch_aux_scores, batch_size)

            for pidx in range(batch_size):
                # retrieve data for current precursor
                precursor = batch_precursor[pidx]
                middle_rts = batch_middle_rts[pidx]
                lib_pearsons = batch_lib_pearsons[pidx]

                scores = batch_scores[pidx]
                aux_scores = batch_aux_scores[pidx]
                score_order = np.argsort(-np.array(scores))[:top_k]
                
                # preparing outputs
                assay_rt_kept = list(np.array(middle_rts)[score_order])
                delta_rt_kept = [precursor.RT - i for i in assay_rt_kept]        
                real_intensities_kept = [precursor.lib_frags_real_intensities[idx] for idx in score_order]        
                lib_intensities = [i.get_intensity() for i in precursor.lib_frags]
                lib_cos_score_kept = [cos_sim(i, lib_intensities) for i in real_intensities_kept]
                real_intensities_kept = ["|".join([str(k) for k in i]) for i in real_intensities_kept]
                dream_score_kept = list(scores[score_order])
                ms1_area_kept = list(np.array(precursor.ms1_areas)[score_order])
                ms2_area_kept = list(np.array(precursor.ms2_areas)[score_order])
                frag_kept = [i.format_output() for i in precursor.lib_frags]
                lib_pearsons_kept = ["|".join([str(k) for k in lib_pearsons[idx]]) for idx in score_order]
                aux_scores_kept = ["|".join([str(k) for k in i]) for i in list(aux_scores[score_order])]
        
                out_string_dict = {"transition_group_id" : precursor.precursor_id, 
                           "filename" : rawdata_file, 
                           "RT" : precursor.RT, 
                           "PeptideSequence" : precursor.sequence,
                           "FullPeptideName" : precursor.full_sequence, 
                           "Charge" : precursor.charge, 
                           "m/z" : precursor.precursor_mz, 
                           "ProteinName" : precursor.protein_name, 
                           "decoy" : precursor.decoy, 
                           "assay_rt" : ";".join([str(i) for i in assay_rt_kept]), 
                           "delta_rt" : ";".join([str(i) for i in delta_rt_kept]), 
                           "norm_rt" : precursor.iRT, 
                           "nr_peaks" : str(len(precursor.lib_frags)), 
                           "real_intensities" : ";".join(real_intensities_kept), 
                           "lib_cos_scores" : ";".join([str(i) for i in lib_cos_score_kept]), 
                           "dream_scores" : ";".join([str(i) for i in dream_score_kept]), 
                           "ms1_area" : ";".join(ms1_area_kept), 
                           "ms2_area" : ";".join(ms2_area_kept), 
                           "aggr_Fragment_Annotation" : ";".join(frag_kept), 
                           "lib_pearsons" : ";".join(lib_pearsons_kept), 
                           "aux_scores" : ";".join(aux_scores_kept)}
                out_string_1 = "%(transition_group_id)s\t%(filename)s\t%(RT)s\t"
                out_string_2 = "%(PeptideSequence)s\t%(FullPeptideName)s\t%(Charge)s\t"
                out_string_3 = "%(m/z)s\t%(ProteinName)s\t%(decoy)s\t"
                out_string_4 = "%(assay_rt)s\t%(delta_rt)s\t%(norm_rt)s\t"
                out_string_5 = "%(nr_peaks)s\t%(real_intensities)s\t%(lib_cos_scores)s\t"
                out_string_6 = "%(dream_scores)s\t%(ms1_area)s\t%(ms2_area)s\t%(aggr_Fragment_Annotation)s\t%(lib_pearsons)s\t%(aux_scores)s\n"
                out_string = out_string_1 + out_string_2 + out_string_3 + out_string_4 + out_string_5 + out_string_6   

                batch_results.append(out_string % out_string_dict)

            with open(out_file, "a") as v:
                v.write("".join(batch_results))
            
            batch_precursor = []
            batch_matrices = []
            batch_middle_rts = []
            batch_lib_pearsons = []

    # process last batch if not empty
    n_samples = len(batch_precursor)
    if n_samples:
        batch_results = []
        batch_matrices = np.concatenate(batch_matrices)
        
        batch_scores = BM_model(batch_matrices, training = False).numpy().T[0]
        batch_aux_scores = RM_model(batch_matrices, training = False).numpy()

        batch_scores = np.split(batch_scores, n_samples)
        batch_aux_scores = np.split(batch_aux_scores, n_samples)

        for pidx in range(n_samples):
            # retrieve data for current precursor
            precursor = batch_precursor[pidx]
            middle_rts = batch_middle_rts[pidx]
            lib_pearsons = batch_lib_pearsons[pidx]

            scores = batch_scores[pidx]
            aux_scores = batch_aux_scores[pidx]
            score_order = np.argsort(-np.array(scores))[:top_k]
            
            # preparing outputs
            assay_rt_kept = list(np.array(middle_rts)[score_order])
            delta_rt_kept = [precursor.RT - i for i in assay_rt_kept]        
            real_intensities_kept = [precursor.lib_frags_real_intensities[idx] for idx in score_order]        
            lib_intensities = [i.get_intensity() for i in precursor.lib_frags]
            lib_cos_score_kept = [cos_sim(i, lib_intensities) for i in real_intensities_kept]
            real_intensities_kept = ["|".join([str(k) for k in i]) for i in real_intensities_kept]
            dream_score_kept = list(scores[score_order])
            ms1_area_kept = list(np.array(precursor.ms1_areas)[score_order])
            ms2_area_kept = list(np.array(precursor.ms2_areas)[score_order])
            frag_kept = [i.format_output() for i in precursor.lib_frags]
            lib_pearsons_kept = ["|".join([str(k) for k in lib_pearsons[idx]]) for idx in score_order]
            aux_scores_kept = ["|".join([str(k) for k in i]) for i in list(aux_scores[score_order])]
    
            out_string_dict = {"transition_group_id" : precursor.precursor_id, 
                               "filename" : rawdata_file, 
                               "RT" : precursor.RT, 
                               "PeptideSequence" : precursor.sequence,
                               "FullPeptideName" : precursor.full_sequence, 
                               "Charge" : precursor.charge, 
                               "m/z" : precursor.precursor_mz, 
                               "ProteinName" : precursor.protein_name, 
                               "decoy" : precursor.decoy, 
                               "assay_rt" : ";".join([str(i) for i in assay_rt_kept]), 
                               "delta_rt" : ";".join([str(i) for i in delta_rt_kept]), 
                               "norm_rt" : precursor.iRT, 
                               "nr_peaks" : str(len(precursor.lib_frags)), 
                               "real_intensities" : ";".join(real_intensities_kept), 
                               "lib_cos_scores" : ";".join([str(i) for i in lib_cos_score_kept]), 
                               "dream_scores" : ";".join([str(i) for i in dream_score_kept]), 
                               "ms1_area" : ";".join(ms1_area_kept), 
                               "ms2_area" : ";".join(ms2_area_kept), 
                               "aggr_Fragment_Annotation" : ";".join(frag_kept), 
                               "lib_pearsons" : ";".join(lib_pearsons_kept), 
                               "aux_scores" : ";".join(aux_scores_kept)}
            out_string_1 = "%(transition_group_id)s\t%(filename)s\t%(RT)s\t"
            out_string_2 = "%(PeptideSequence)s\t%(FullPeptideName)s\t%(Charge)s\t"
            out_string_3 = "%(m/z)s\t%(ProteinName)s\t%(decoy)s\t"
            out_string_4 = "%(assay_rt)s\t%(delta_rt)s\t%(norm_rt)s\t"
            out_string_5 = "%(nr_peaks)s\t%(real_intensities)s\t%(lib_cos_scores)s\t"
            out_string_6 = "%(dream_scores)s\t%(ms1_area)s\t%(ms2_area)s\t%(aggr_Fragment_Annotation)s\t%(lib_pearsons)s\t%(aux_scores)s\n"
            out_string = out_string_1 + out_string_2 + out_string_3 + out_string_4 + out_string_5 + out_string_6   

            batch_results.append(out_string % out_string_dict)

        with open(out_file, "a") as v:
            v.write("".join(batch_results))