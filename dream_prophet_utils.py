"""
╔═════════════════════════════════════════════════════╗
║               dream_prophet_utils.py                ║
╠═════════════════════════════════════════════════════╣
║   Description: Utility functions of `dreamprophet`  ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os

import numpy as np
import pandas as pd

from file_io import load_all_scoring_profiles, decompress_1d_array, load_all_precursor_ids
from utils import tear_list_given_n_each_batch

def load_scoring_profiles_and_tear_into_chunks(sqdream_file, n_threads):
    all_scoring_profiles = load_all_scoring_profiles(sqdream_file)
    all_precursors = list(all_scoring_profiles["PRECURSOR_ID"])
    all_precursor_row_ids = list(range(all_scoring_profiles.shape[0]))
    
    n_precursors_each_batch = int(len(all_precursors) / n_threads)
    batch_precursors = tear_list_given_n_each_batch(all_precursors, n_precursors_each_batch)
    batch_precursor_row_ids = tear_list_given_n_each_batch(all_precursor_row_ids, n_precursors_each_batch)

    return all_scoring_profiles, batch_precursors, batch_precursor_row_ids

def load_precursor_ids_and_tear_into_chunks(sqdream_file, n_total_precursors_batch, runs_under_analysis):
    all_precursors = load_all_precursor_ids(sqdream_file)    
    n_precursors_each_batch = max(1, int(n_total_precursors_batch / len(runs_under_analysis)))
    batch_precursors = tear_list_given_n_each_batch(all_precursors, n_precursors_each_batch)

    return batch_precursors

class Scoring_profile:
    def __init__(self, precursor_sp_record, run_name):
        self.run_name = run_name
        self.precursor_id = precursor_sp_record["PRECURSOR_ID"]
        self.peptide_sequence = precursor_sp_record["FULL_SEQUENCE"]
        self.pure_sequence = precursor_sp_record["SEQUENCE"]
        self.precursor_charge = precursor_sp_record["CHARGE"]
        self.precursor_mz = precursor_sp_record["PRECURSOR_MZ"]
        self.irt = precursor_sp_record["IRT"]
        self.protein_name = precursor_sp_record["PROTEIN_NAME"]
        self.decoy = precursor_sp_record["DECOY"]

        self.scores = {"middle_rts" : decompress_1d_array(precursor_sp_record["MIDDLE_RTS"]), 
                       "dream_scores" : decompress_1d_array(precursor_sp_record["DREAM_SCORE"]), 
                       "lib_cos_scores" : decompress_1d_array(precursor_sp_record["LIB_COS_SCORE"]), 
                       "ms1_area" : decompress_1d_array(precursor_sp_record["MS1_AREA"]), 
                       "ms2_area" : decompress_1d_array(precursor_sp_record["MS2_AREA"]), 
                       "delta_rt" : decompress_1d_array(precursor_sp_record["DELTA_RT"]), 
                       "quantification" : decompress_1d_array(precursor_sp_record["QUANTIFICATION"])}

    def get_static_info(self):
        static_info = {"precursor_id" : self.precursor_id, 
                       "peptide_sequence" : self.peptide_sequence, 
                       "pure_sequence" : self.pure_sequence, 
                       "precursor_charge" : self.precursor_charge, 
                       "precursor_mz" : self.precursor_mz, 
                       "irt" : self.irt, 
                       "protein_name" : self.protein_name, 
                       "decoy" : self.decoy}
        return static_info
        
    def pick_peaks_and_score_single_run(self, top_k):
        picked_indices = np.argsort(-self.scores["dream_scores"])[:top_k]
        self.picked_scores = {"alignment_boosted" : 0, 
                               "dream_scores" : pd.DataFrame(self.scores["dream_scores"][picked_indices], columns = [self.run_name]), 
                               "delta_rt" : pd.DataFrame(self.scores["delta_rt"][picked_indices], columns = [self.run_name]), 
                               "lib_cos_scores" : pd.DataFrame(self.scores["lib_cos_scores"][picked_indices], columns = [self.run_name]), 
                               "ms1_area" : pd.DataFrame(self.scores["ms1_area"][picked_indices], columns = [self.run_name]), 
                               "ms2_area" : pd.DataFrame(self.scores["ms2_area"][picked_indices], columns = [self.run_name]), 
                               "middle_rts" : pd.DataFrame(self.scores["middle_rts"][picked_indices], columns = [self.run_name]), 
                               "quant" : pd.DataFrame(self.scores["quantification"][picked_indices], columns = [self.run_name]), 
                               "aligned_dream_score" : pd.DataFrame(self.scores["dream_scores"][picked_indices], columns = [self.run_name]), 
                               "aligned_lib_cos_score" : pd.DataFrame(self.scores["lib_cos_scores"][picked_indices], columns = [self.run_name]), 
                               "aligned_ms1_area" : pd.DataFrame(self.scores["ms1_area"][picked_indices], columns = [self.run_name]), 
                               "aligned_ms2_area" : pd.DataFrame(self.scores["ms2_area"][picked_indices], columns = [self.run_name]), 
                               "rt_mean" : pd.Series([self.scores["middle_rts"][picked_indices].mean()], index = [self.run_name]), 
                               "rt_std" : pd.Series([self.scores["middle_rts"][picked_indices].std()], index = [self.run_name]), 
                               "delta_rt_mean" : pd.Series([self.scores["delta_rt"][picked_indices].mean()], index = [self.run_name]), 
                               "delta_rt_std" : pd.Series([self.scores["delta_rt"][picked_indices].std()], index = [self.run_name]), 
                               "dream_score_mean" : pd.Series([self.scores["dream_scores"][picked_indices].mean()], index = [self.run_name]), 
                               "dream_score_std" : pd.Series([self.scores["dream_scores"][picked_indices].std()], index = [self.run_name]), 
                               "lib_cos_score_mean" : pd.Series([self.scores["lib_cos_scores"][picked_indices].mean()], index = [self.run_name]), 
                               "lib_cos_score_std" : pd.Series([self.scores["lib_cos_scores"][picked_indices].std()], index = [self.run_name])}
        self.top_k = top_k
        
    def format_scoring_table_single_run(self):
        scoring_table = {}
        
        scoring_table["transition_group_id"] = [self.precursor_id] * self.top_k
        scoring_table["filename"] = [self.run_name] * self.top_k
        scoring_table["PeptideSequence"] = [self.pure_sequence] * self.top_k
        scoring_table["FullPeptideName"] = [self.peptide_sequence] * self.top_k
        scoring_table["SCORE_iRT"] = [self.irt] * self.top_k
        scoring_table["ProteinName"] = [self.protein_name] * self.top_k

        scoring_table["SCORE_RT"] = self.picked_scores["middle_rts"][self.run_name].to_list()
        scoring_table["SCORE_RT_mean"] = [self.picked_scores["rt_mean"][self.run_name]] * self.top_k
        scoring_table["SCORE_RT_std"] = [self.picked_scores["rt_std"][self.run_name]] * self.top_k

        scoring_table["SCORE_DREAM"] = self.picked_scores["dream_scores"][self.run_name].to_list()
        scoring_table["SCORE_DREAM_mean"] = [self.picked_scores["dream_score_mean"][self.run_name]] * self.top_k
        scoring_table["SCORE_DREAM_std"] = [self.picked_scores["dream_score_std"][self.run_name]] * self.top_k

        scoring_table["SCORE_lib_cosine"] = self.picked_scores["lib_cos_scores"][self.run_name].to_list()
        scoring_table["SCORE_lib_cosine_mean"] = [self.picked_scores["lib_cos_score_mean"][self.run_name]] * self.top_k
        scoring_table["SCORE_lib_cosine_std"] = [self.picked_scores["lib_cos_score_std"][self.run_name]] * self.top_k

        scoring_table["SCORE_deltaRT"] = self.picked_scores["delta_rt"][self.run_name].to_list()
        scoring_table["SCORE_deltaRT_mean"] = [self.picked_scores["delta_rt_mean"][self.run_name]] * self.top_k
        scoring_table["SCORE_deltaRT_std"] = [self.picked_scores["delta_rt_std"][self.run_name]] * self.top_k

        scoring_table["SCORE_MS1_area"] = self.picked_scores["ms1_area"][self.run_name].to_list()
        scoring_table["SCORE_MS2_area"] = self.picked_scores["ms2_area"][self.run_name].to_list()
        
        scoring_table["SCORE_charge"] = [self.precursor_charge] * self.top_k
        scoring_table["SCORE_pep_len"] = [len(self.pure_sequence)] * self.top_k
        scoring_table["SCORE_mz"] = [self.precursor_mz] * self.top_k
        scoring_table["decoy"] = [self.decoy] * self.top_k
        scoring_table["Intensity"] = self.picked_scores["quant"][self.run_name].to_list()
        
        return scoring_table

def get_peak_picking_single_run_results(scoring_profiles, feature_queue, run_name, top_k):
    for i in range(scoring_profiles.shape[0]):
        precursor_sp_record = scoring_profiles.iloc[i, :]
        scoring_profile_precursor = Scoring_profile(precursor_sp_record, run_name)
        scoring_profile_precursor.pick_peaks_and_score_single_run(top_k)
        
        feature_queue.put(scoring_profile_precursor)
    
    feature_queue.put(None)

def collect_scoring_table(feature_queue, output_dir, single_run_scoring_table_name, n_threads):
    scoring_table_list = []

    n_none = 0

    while 1:
        scoring_profile_precursor = feature_queue.get()

        if scoring_profile_precursor is None:
            n_none += 1
            feature_queue.task_done()
            if n_none >= n_threads:
                break
            else:
                continue           
        
        scoring_table_precursor = scoring_profile_precursor.format_scoring_table_single_run()
        scoring_table_list.append(pd.DataFrame(scoring_table_precursor))

        feature_queue.task_done()

    total_scoring_table = pd.concat(scoring_table_list)
    total_scoring_table.to_csv(os.path.join(output_dir, single_run_scoring_table_name), sep = "\t", index = False)

def load_chromatograms_of_one_precursor_from_memory(chromatograms, row_ids):
    precursor_chrom_record = chromatograms.iloc[row_ids, :]
    ms2_inten_lists = []
    for anno, data in zip(precursor_chrom_record["ANNOTATION"], precursor_chrom_record["DATA"]):
        if anno == "RT":
            rt_list = decompress_1d_array(data)
        elif anno == "MS1":
            ms1_inten = decompress_1d_array(data)
        else:
            ms2_inten_lists.append(decompress_1d_array(data))

    return rt_list, ms2_inten_lists, ms1_inten

def merge_dataframes(data_frame_list):
    return pd.concat([df.reset_index(drop = True) for df in data_frame_list], axis = 1)

def merge_pd_series(pd_series_list):
    return pd.concat(pd_series_list)

def merge_score_packages(all_score_packages):
    if len(all_score_packages) == 1:
        return all_score_packages[0]

    merged_score_package = {}
    merged_score_package["alignment_boosted"] = max([i["alignment_boosted"] for i in all_score_packages])
    merged_score_package["dream_scores"] = merge_dataframes([i["dream_scores"] for i in all_score_packages])
    merged_score_package["delta_rt"] = merge_dataframes([i["delta_rt"] for i in all_score_packages])
    merged_score_package["lib_cos_scores"] = merge_dataframes([i["lib_cos_scores"] for i in all_score_packages])
    merged_score_package["ms1_area"] = merge_dataframes([i["ms1_area"] for i in all_score_packages])
    merged_score_package["ms2_area"] = merge_dataframes([i["ms2_area"] for i in all_score_packages])
    merged_score_package["middle_rts"] = merge_dataframes([i["middle_rts"] for i in all_score_packages])
    merged_score_package["quant"] = merge_dataframes([i["quant"] for i in all_score_packages])
    merged_score_package["aligned_dream_score"] = merge_dataframes([i["aligned_dream_score"] for i in all_score_packages])
    merged_score_package["aligned_lib_cos_score"] = merge_dataframes([i["aligned_lib_cos_score"] for i in all_score_packages])
    merged_score_package["aligned_ms1_area"] = merge_dataframes([i["aligned_ms1_area"] for i in all_score_packages])
    merged_score_package["aligned_ms2_area"] = merge_dataframes([i["aligned_ms2_area"] for i in all_score_packages])
    merged_score_package["rt_mean"] = merge_pd_series([i["rt_mean"] for i in all_score_packages])
    merged_score_package["rt_std"] = merge_pd_series([i["rt_std"] for i in all_score_packages])
    merged_score_package["delta_rt_mean"] = merge_pd_series([i["delta_rt_mean"] for i in all_score_packages])
    merged_score_package["delta_rt_std"] = merge_pd_series([i["delta_rt_std"] for i in all_score_packages])
    merged_score_package["dream_score_mean"] = merge_pd_series([i["dream_score_mean"] for i in all_score_packages])
    merged_score_package["dream_score_std"] = merge_pd_series([i["dream_score_std"] for i in all_score_packages])
    merged_score_package["lib_cos_score_mean"] = merge_pd_series([i["lib_cos_score_mean"] for i in all_score_packages])
    merged_score_package["lib_cos_score_std"] = merge_pd_series([i["lib_cos_score_std"] for i in all_score_packages])
    
    return merged_score_package