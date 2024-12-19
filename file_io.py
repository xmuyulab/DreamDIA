"""
╔═════════════════════════════════════════════════════╗
║                     file_io.py                      ║
╠═════════════════════════════════════════════════════╣
║    Description: Utility functions for file I/O      ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os
import zlib
import array
import sqlite3

import numpy as np
import pandas as pd

def compress_1d_array(a_array):
    return zlib.compress(array.array("d", a_array).tobytes())
def decompress_1d_array(array_bytes):
    return np.array(array.array("d", zlib.decompress(array_bytes)).tolist())
def compress_nd_array(a_array, dtype = "float64"):
    return zlib.compress(np.array(a_array).astype(dtype))
def decompress_nd_array(array_bytes, dtype = "float64"):
    return np.frombuffer(zlib.decompress(array_bytes), dtype = dtype)

def init_sqdream(sqdream_name):
    if os.path.exists(sqdream_name):
        os.remove(sqdream_name)
    
    output_db = sqlite3.connect(sqdream_name)
    chromatogram_df = pd.DataFrame({"PRECURSOR_ID" : [], 
                                    "ANNOTATION" : [], 
                                    "DATA" : []})
    ipf_df = pd.DataFrame({"PRECURSOR_ID" : [], 
                           "ANNOTATION" : [], 
                           "XCORR_SCORE" : [], 
                           "XCORR_SHAPE_SCORE" : [], 
                           "EMG_SCORE" : []})
    
    chromatogram_df.to_sql("CHROMATOGRAM", 
                           con = output_db, 
                           dtype = {"PRECURSOR_ID" : "TEXT", 
                                    "ANNOTATION" : "TEXT", 
                                    "DATA" : "BLOB"}, 
                           index = False)
    ipf_df.to_sql("IPF_SCORE", 
                  con = output_db, 
                  dtype = {"PRECURSOR_ID" : "TEXT", 
                           "FRAGMENT_ION" : "TEXT", 
                           "XCORR_SCORE" : "BLOB", 
                           "XCORR_SHAPE_SCORE" : "BLOB", 
                           "EMG_SCORE" : "BLOB"}, 
                  index = False)

    output_db.close()

class Scoring_profile_cacher:
    def __init__(self):
        self.cacher = {"PRECURSOR_ID" : [], 
                       "FULL_SEQUENCE" : [], 
                       "SEQUENCE" : [], 
                       "CHARGE" : [], 
                       "PRECURSOR_MZ" : [], 
                       "IRT" : [], 
                       "PROTEIN_NAME" : [], 
                       "DECOY" : [], 
                       "MIDDLE_RTS" : [], 
                       "DREAM_SCORE" : [], 
                       "LIB_COS_SCORE" : [], 
                       "MS1_AREA" : [], 
                       "MS2_AREA" : [], 
                       "DELTA_RT" : [], 
                       "QUANTIFICATION" : []}    
    
    def append_precursor(self, precursor, rsm_info, dream_scores): 
        self.cacher["PRECURSOR_ID"].append(precursor.precursor_id)
        self.cacher["FULL_SEQUENCE"].append(precursor.full_sequence)
        self.cacher["SEQUENCE"].append(precursor.sequence)
        self.cacher["CHARGE"].append(precursor.charge)
        self.cacher["PRECURSOR_MZ"].append(precursor.precursor_mz)
        self.cacher["IRT"].append(precursor.iRT)
        self.cacher["PROTEIN_NAME"].append(precursor.protein_name)
        self.cacher["DECOY"].append(precursor.decoy)
        self.cacher["MIDDLE_RTS"].append(zlib.compress(array.array("d", rsm_info["middle_rts"]).tobytes()))
        self.cacher["DREAM_SCORE"].append(zlib.compress(array.array("d", dream_scores).tobytes()))
        self.cacher["LIB_COS_SCORE"].append(zlib.compress(array.array("d", rsm_info["lib_cos_scores"]).tobytes()))
        self.cacher["MS1_AREA"].append(zlib.compress(array.array("d", rsm_info["ms1_area_list"]).tobytes()))
        self.cacher["MS2_AREA"].append(zlib.compress(array.array("d", np.array(rsm_info["ms2_area_list"]).sum(axis = 1)).tobytes()))
        self.cacher["DELTA_RT"].append(zlib.compress(array.array("d", rsm_info["delta_rts"]).tobytes()))
        self.cacher["QUANTIFICATION"].append(zlib.compress(array.array("d", np.array(rsm_info["quantities"])).tobytes()))
        
    def output(self, sqdream_file):
        output_cacher = pd.DataFrame(self.cacher)
        output_db = sqlite3.connect(sqdream_file, timeout = 10000)
        output_cacher.to_sql("SCORING_PROFILE", 
                            con = output_db, 
                            dtype = {"PRECURSOR_ID" : "TEXT", 
                                    "FULL_SEQUENCE" : "TEXT", 
                                    "SEQUENCE" : "TEXT", 
                                    "CHARGE" : "INT", 
                                    "PRECURSOR_MZ" : "REAL", 
                                    "IRT" : "REAL", 
                                    "PROTEIN_NAME" : "TEXT", 
                                    "DECOY" : "INT", 
                                    "MIDDLE_RTS" : "BLOB", 
                                    "DREAM_SCORE" : "BLOB", 
                                    "LIB_COS_SCORE" : "BLOB", 
                                    "MS1_AREA" : "BLOB", 
                                    "MS2_AREA" : "BLOB", 
                                    "DELTA_RT" : "BLOB", 
                                    "QUANTIFICATION" : "BLOB"}, 
                            index = False)
        output_db.close()

def insert_chroms_batch(chrom_info_list, sqdream_name):
    output_db = sqlite3.connect(sqdream_name, timeout = 10000)
    output_cursor = output_db.cursor()
    insert_to_chromatogram = []
    for chrom_info in chrom_info_list:
        insert_to_chromatogram.append([chrom_info[0], "RT", sqlite3.Binary(chrom_info[2])])
        insert_to_chromatogram.append([chrom_info[0], "MS1", sqlite3.Binary(chrom_info[4])])
        for ms2_anno, ms2_chrom in zip(chrom_info[1], chrom_info[3]):
            insert_to_chromatogram.append([chrom_info[0], "MS2_%s" % ms2_anno, sqlite3.Binary(ms2_chrom)])
    
    output_cursor.executemany("INSERT INTO CHROMATOGRAM VALUES (?, ?, ?);", insert_to_chromatogram)
    output_db.commit()
    output_db.close()     

def insert_ipf_scores_batch(chrom_info_list, sqdream_name):
    output_db = sqlite3.connect(sqdream_name, timeout = 10000)
    output_cursor = output_db.cursor()
    insert_to_ipf = []
    for chrom_info in chrom_info_list:
        for frag_idx in range(len(chrom_info[5])):
            insert_to_ipf.append([chrom_info[0], 
                                  "%s_%s" % (chrom_info[5][frag_idx], chrom_info[6][frag_idx]), 
                                  sqlite3.Binary(chrom_info[7][frag_idx]), 
                                  sqlite3.Binary(chrom_info[8][frag_idx]), 
                                  sqlite3.Binary(chrom_info[9][frag_idx])])
    
    output_cursor.executemany("INSERT INTO IPF_SCORE VALUES (?, ?, ?, ?, ?);", insert_to_ipf)
    output_db.commit()
    output_db.close()     

def load_all_scoring_profiles(sqdream_file):
    """
    Load all scoring profiles from the specified SQLite database file and sort them by PRECURSOR_ID.

    Parameters:
    sqdream_file (str): The path to the SQLite database file containing the scoring profiles.

    Returns:
    pd.DataFrame: A DataFrame containing all scoring profiles, sorted by PRECURSOR_ID.
    """

    db = sqlite3.connect(sqdream_file)
    scoring_profile = pd.read_sql("SELECT * FROM SCORING_PROFILE;", con = db)
    scoring_profile = scoring_profile.sort_values(by = "PRECURSOR_ID")
    db.close()
    
    return scoring_profile

def load_all_precursor_ids(sqdream_file):
    db = sqlite3.connect(sqdream_file)
    precursor_ids = pd.read_sql("SELECT PRECURSOR_ID FROM SCORING_PROFILE;", con = db)
    precursor_ids = precursor_ids.sort_values(by = "PRECURSOR_ID")
    db.close()
    
    return precursor_ids["PRECURSOR_ID"].values

def load_batch_chromatograms(sqdream_file, precursor_id_list):
    wanted_precursors = ["'" + i + "'" for i in precursor_id_list]
    wanted_precursors = ",".join(wanted_precursors)

    db = sqlite3.connect(sqdream_file)
    chromatograms = pd.read_sql("SELECT * FROM CHROMATOGRAM WHERE PRECURSOR_ID in ( %s );" % wanted_precursors, con = db)
    chromatograms = chromatograms.sort_values(by = "PRECURSOR_ID")
    db.close()
    
    return chromatograms

def load_batch_scoring_profiles(sqdream_file, precursor_id_list):
    wanted_precursors = ["'" + i + "'" for i in precursor_id_list]
    wanted_precursors = ",".join(wanted_precursors)

    db = sqlite3.connect(sqdream_file)
    scoring_profiles = pd.read_sql("SELECT * FROM SCORING_PROFILE WHERE PRECURSOR_ID in ( %s );" % wanted_precursors, con = db)
    scoring_profiles = scoring_profiles.sort_values(by = "PRECURSOR_ID")
    db.close()
    
    return scoring_profiles