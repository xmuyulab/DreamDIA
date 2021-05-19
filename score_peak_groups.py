import os
import os.path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
from scipy.stats import pearsonr
import tools_cython as tools
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
    def get_charge(self):
        return self.__charge  
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
        self.precursor_win_id = None
        self.ms1_areas = []
        self.ms2_areas = []
        self.lib_frags_real_intensities = []
        self.lib_pearsons = [] 
        self.self_frags, self.self_frag_charges = np.array(calc_all_fragment_mzs(self.full_sequence, 
                                                                                 self.charge, 
                                                                                 (mz_min, mz_max),  
                                                                                 return_charges = True))       
        iso_shift_max = int(min(iso_range, (mz_max - self.precursor_mz) * self.charge)) + 1
        self.qt3_frags = [self.precursor_mz + iso_shift / self.charge for iso_shift in range(iso_shift_max)]
        self.lib_frags = [Lib_frag(mz, charge, fragtype, series, inten) for mz, charge, fragtype, series, inten in zip(frag_mz_list, frag_charge_list, frag_type_list, frag_series_list, frag_intensity_list)]
        self.iso_frags = self.filter_frags([i.get_mz() + 1 / i.get_charge() for i in self.lib_frags], mz_min, mz_max, padding = True)
        self.light_frags = self.filter_frags([i.get_mz() - 1 / i.get_charge() for i in self.lib_frags], mz_min, mz_max, padding = True)
    def filter_frags(self, frag_list, mz_min, mz_max, padding = False, padding_value = -1):
        if padding:
            return list(map(lambda x : x if (mz_min <= x < mz_max) else padding_value, frag_list))
        return [i for i in frag_list if mz_min <= i < mz_max]
    def set_RT(self, rt_norm_model, rt_model_params):
        if rt_norm_model == "linear":
            self.RT = self.iRT * rt_model_params[0] + rt_model_params[1]
        else:
            self.RT = np.poly1d(rt_model_params)(self.iRT)
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
                       n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, 
                       peak_index_range, rt_norm_model, rt_model_params, p_id): 
    peak_indice = get_peak_indice(model_cycles, peak_index_range)
    feature_dimension = n_lib_frags * 3 + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags
    for idx, precursor in enumerate(precursor_list): 
        precursor.set_RT(rt_norm_model, rt_model_params)
        precursor.precursor_win_id = calc_win_id(precursor.precursor_mz, win_range)
        rt_pos_ms1 = find_rt_pos(precursor.RT, ms1.rt_list, n_cycles)
        rt_pos_ms2 = find_rt_pos(precursor.RT, ms2[precursor.precursor_win_id].rt_list, n_cycles)       
        precursor_rt_list = [ms1.rt_list[i] for i in rt_pos_ms1]
        precursor_ms1_spectra = [ms1.spectra[i] for i in rt_pos_ms1]
        precursor_ms2_spectra = [ms2[precursor.precursor_win_id].spectra[i] for i in rt_pos_ms2]
        lib_frags = [frag.get_mz() for frag in precursor.lib_frags]
        all_lib_xics = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, mz_tol_ms2) for frag in lib_frags])
        all_lib_xics_1 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.2 * mz_tol_ms2) for frag in lib_frags])
        all_lib_xics_2 = np.array([calc_XIC(precursor_ms2_spectra, frag, mz_unit, 0.45 * mz_tol_ms2) for frag in lib_frags])
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
        orig_matrices, matrices, middle_rts, rt_lists = [], [], [], []
        for rt_start in range(n_cycles - model_cycles + 1):
            rt_end = rt_start + model_cycles
            precursor_rt_list_part = precursor_rt_list[rt_start : rt_end]
            middle_rts.append(precursor_rt_list_part[model_cycles // 2])
            rt_lists.append(precursor_rt_list_part)
            lib_xics = all_lib_xics[:, rt_start : rt_end]
            lib_xics_1 = all_lib_xics_1[:, rt_start : rt_end]
            lib_xics_2 = all_lib_xics_2[:, rt_start : rt_end]
            self_xics = all_self_xics[:, rt_start : rt_end] 
            qt3_xics = all_qt3_xics[:, rt_start : rt_end]
            ms1_xics = all_ms1_xics[:, rt_start : rt_end]
            iso_xics = all_iso_xics[:, rt_start : rt_end]
            light_xics = all_light_xics[:, rt_start : rt_end]          
            self_xics = filter_matrix(self_xics)
            qt3_xics = filter_matrix(qt3_xics)
            lib_xics = tools.smooth_array(lib_xics.astype(float))
            lib_xics_1 = tools.smooth_array(lib_xics_1.astype(float))
            lib_xics_2 = tools.smooth_array(lib_xics_2.astype(float))
            self_xics = tools.smooth_array(self_xics.astype(float))
            qt3_xics = tools.smooth_array(qt3_xics.astype(float))
            ms1_xics = tools.smooth_array(ms1_xics.astype(float))
            iso_xics = tools.smooth_array(iso_xics.astype(float))
            light_xics = tools.smooth_array(light_xics.astype(float))
            precursor_rt_list_part_diff = np.array(precursor_rt_list_part[1:]) - np.array(precursor_rt_list_part[:-1])
            ms2_areas = [tools.calc_area(lib_xics[i, :], precursor_rt_list_part_diff) for i in range(lib_xics.shape[0])]
            ms1_area = tools.calc_area(ms1_xics[0, :], precursor_rt_list_part_diff)
            precursor.ms2_areas.append("|".join([str(each) for each in ms2_areas]))
            precursor.ms1_areas.append(str(ms1_area))
            peak_intensities = lib_xics[:, peak_indice].mean(axis = 1)
            precursor.lib_frags_real_intensities.append(peak_intensities)
            std_indice, pearson_sums = calc_pearson_sums(lib_xics)
            precursor.lib_pearsons.append(pearson_sums)
            if lib_xics.shape[0] > 0:
                std_indice, pearson_sums = calc_pearson_sums(lib_xics)
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
            orig_matrices.append(training_matrix)
            training_matrix = MinMaxScaler().fit_transform(training_matrix)
            matrices.append(training_matrix)
        matrix_queue.put([precursor, orig_matrices, matrices, middle_rts, rt_lists])   
    matrix_queue.put(None)
def score_batch(matrix_queue, lib_cols, BM_model_file, RM_model_file, out_file, rawdata_file, top_k, n_threads, batch_size, n_total_precursors, logger, out_chrom, rt_norm_dir):
    BM_model = load_model(BM_model_file, compile = False)
    RM_model = load_model(RM_model_file, compile = False)
    BM_model.call = tf.function(BM_model.call, experimental_relax_shapes = True)
    RM_model.call = tf.function(RM_model.call, experimental_relax_shapes = True)
    if out_chrom:
        chrom_dir = os.path.join(rt_norm_dir, "chrom")
        if not os.path.exists(chrom_dir):
            os.mkdir(chrom_dir)
    out_head_1 = "%s\tfilename\tRT\t%s\t%s\t" % (lib_cols["PRECURSOR_ID_COL"], lib_cols["PURE_SEQUENCE_COL"], lib_cols["FULL_SEQUENCE_COL"])
    out_head_2 = "%s\t%s\t%s\t%s\tassay_rt\tdelta_rt\t" % (lib_cols["PRECURSOR_CHARGE_COL"], lib_cols["PRECURSOR_MZ_COL"], lib_cols["PROTEIN_NAME_COL"], lib_cols["DECOY_OR_NOT_COL"])
    out_head_3 = "%s\tnr_peaks\treal_intensities\tlib_cos_scores\t" % lib_cols["IRT_COL"]
    out_head_4 = "dream_scores\tms1_area\tms2_areas\taggr_Fragment_Annotation\tlib_pearsons\tdrf_scores\n"
    n_none = 0
    precursor_count = 0
    results = []
    batch_precursor = []
    batch_matrices = []
    batch_middle_rts = []  
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
        precursor, orig_matrices, matrices, middle_rts, rt_lists = matrix_data       
        if out_chrom:
            with open(os.path.join(chrom_dir, "%s.pkl" % precursor.precursor_id), "wb") as kk:
                pickle.dump([orig_matrices, middle_rts, rt_lists], kk)
        batch_precursor.append(precursor)
        batch_matrices.append(matrices)        
        batch_middle_rts.append(middle_rts)
        matrix_queue.task_done()
        if precursor_count % 10000 == 0:
            logger.info("(%d / %d) precursors scored ..." % (precursor_count, n_total_precursors))
        if precursor_count % batch_size == 0:
            batch_matrices = np.concatenate(batch_matrices)
            batch_scores = BM_model(batch_matrices, training = False).numpy().T[0]
            batch_drf_scores = RM_model(batch_matrices, training = False).numpy()
            batch_scores = np.split(batch_scores, batch_size)           
            batch_drf_scores = np.split(batch_drf_scores, batch_size)
            for pidx in range(batch_size):
                precursor = batch_precursor[pidx]
                middle_rts = batch_middle_rts[pidx]
                scores = batch_scores[pidx]
                drf_scores = batch_drf_scores[pidx]
                score_order = np.argsort(-np.array(scores))[:top_k]
                assay_rt_kept = list(np.array(middle_rts)[score_order])
                delta_rt_kept = [precursor.RT - i for i in assay_rt_kept]        
                real_intensities_kept = [precursor.lib_frags_real_intensities[idx] for idx in score_order]        
                lib_intensities = [i.get_intensity() for i in precursor.lib_frags]
                lib_cos_score_kept = [cos_sim(i, lib_intensities) for i in real_intensities_kept]
                real_intensities_kept = [pad_list_with_zeros(i, 6) for i in real_intensities_kept]
                real_intensities_kept = ["|".join([str(k) for k in i]) for i in real_intensities_kept]
                dream_score_kept = list(scores[score_order])
                ms1_area_kept = list(np.array(precursor.ms1_areas)[score_order])
                ms2_area_kept = list(np.array(precursor.ms2_areas)[score_order])
                frag_kept = [i.format_output() for i in precursor.lib_frags]
                lib_pearsons_kept = ["|".join([str(k) for k in precursor.lib_pearsons[idx]]) for idx in score_order]
                drf_scores_kept = ["|".join([str(k) for k in i]) for i in list(drf_scores[score_order])]
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
                           "drf_scores" : ";".join(drf_scores_kept)}
                out_string_1 = "%(transition_group_id)s\t%(filename)s\t%(RT)s\t"
                out_string_2 = "%(PeptideSequence)s\t%(FullPeptideName)s\t%(Charge)s\t"
                out_string_3 = "%(m/z)s\t%(ProteinName)s\t%(decoy)s\t"
                out_string_4 = "%(assay_rt)s\t%(delta_rt)s\t%(norm_rt)s\t"
                out_string_5 = "%(nr_peaks)s\t%(real_intensities)s\t%(lib_cos_scores)s\t"
                out_string_6 = "%(dream_scores)s\t%(ms1_area)s\t%(ms2_area)s\t%(aggr_Fragment_Annotation)s\t%(lib_pearsons)s\t%(drf_scores)s"
                out_string = out_string_1 + out_string_2 + out_string_3 + out_string_4 + out_string_5 + out_string_6 + "\n"
                results.append(out_string % out_string_dict)          
            batch_precursor = []
            batch_matrices = []
            batch_middle_rts = []
    n_samples = len(batch_precursor)
    if n_samples:
        batch_matrices = np.concatenate(batch_matrices)
        batch_scores = BM_model(batch_matrices, training = False).numpy().T[0]
        batch_drf_scores = RM_model(batch_matrices, training = False).numpy()
        batch_scores = np.split(batch_scores, n_samples)
        batch_drf_scores = np.split(batch_drf_scores, n_samples)
        for pidx in range(n_samples):
            precursor = batch_precursor[pidx]
            middle_rts = batch_middle_rts[pidx]
            scores = batch_scores[pidx]
            drf_scores = batch_drf_scores[pidx]
            score_order = np.argsort(-np.array(scores))[:top_k]
            assay_rt_kept = list(np.array(middle_rts)[score_order])
            delta_rt_kept = [precursor.RT - i for i in assay_rt_kept]        
            real_intensities_kept = [precursor.lib_frags_real_intensities[idx] for idx in score_order]        
            lib_intensities = [i.get_intensity() for i in precursor.lib_frags]
            lib_cos_score_kept = [cos_sim(i, lib_intensities) for i in real_intensities_kept]
            real_intensities_kept = [pad_list_with_zeros(i, 6) for i in real_intensities_kept]
            real_intensities_kept = ["|".join([str(k) for k in i]) for i in real_intensities_kept]
            dream_score_kept = list(scores[score_order])
            ms1_area_kept = list(np.array(precursor.ms1_areas)[score_order])
            ms2_area_kept = list(np.array(precursor.ms2_areas)[score_order])
            frag_kept = [i.format_output() for i in precursor.lib_frags]
            lib_pearsons_kept = ["|".join([str(k) for k in precursor.lib_pearsons[idx]]) for idx in score_order]
            drf_scores_kept = ["|".join([str(k) for k in i]) for i in list(drf_scores[score_order])]   
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
                        "drf_scores" : ";".join(drf_scores_kept)}
            out_string_1 = "%(transition_group_id)s\t%(filename)s\t%(RT)s\t"
            out_string_2 = "%(PeptideSequence)s\t%(FullPeptideName)s\t%(Charge)s\t"
            out_string_3 = "%(m/z)s\t%(ProteinName)s\t%(decoy)s\t"
            out_string_4 = "%(assay_rt)s\t%(delta_rt)s\t%(norm_rt)s\t"
            out_string_5 = "%(nr_peaks)s\t%(real_intensities)s\t%(lib_cos_scores)s\t"
            out_string_6 = "%(dream_scores)s\t%(ms1_area)s\t%(ms2_area)s\t%(aggr_Fragment_Annotation)s\t%(lib_pearsons)s\t%(drf_scores)s"
            out_string = out_string_1 + out_string_2 + out_string_3 + out_string_4 + out_string_5 + out_string_6 + "\n"
            results.append(out_string % out_string_dict)
    with open(out_file, "w") as out_f:
        out_f.write(out_head_1 + out_head_2 + out_head_3 + out_head_4)
        out_f.write("".join(sorted(results)))