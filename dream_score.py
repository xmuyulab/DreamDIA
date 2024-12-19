"""
╔═════════════════════════════════════════════════════╗
║                   dream_score.py                    ║
╠═════════════════════════════════════════════════════╣
║     Description: Main function of `dreamscore`      ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os
import sys
import multiprocessing

from utils import get_dreamdia_logger, welcome, check_raw_data_files, get_apex_indices
from library_processing import load_library_and_generate_decoys, tear_library
from ms_file_processing import convert_and_load_raw_data
from rt_normalization import generate_endoIRT, load_irt_precursors, score_irt, fit_irt_model
from scoring_utils import load_precursors, score_precursors, output_chromatograms, output_scoring_profiles, output_progress

def dream_score(
    file_dir, 
    lib, 
    out, 
    n_threads, 
    seed, 
    mz_unit, 
    mz_min, 
    mz_max, 
    mz_tol_ms1, 
    mz_tol_ms2, 
    n_irts, 
    irt_mode,
    irt_score_cutoff, 
    irt_score_libcos_cutoff, 
    rt_norm_model, 
    n_cycles, 
    n_frags_each_precursor, 
    decoy_method, 
    ipf_scoring
):    
    logger = get_dreamdia_logger()
    welcome(logger, "dream_score")
    
    logger.info("Output files will be at: %s" % out)
    if not os.path.exists(out):
        os.mkdir(out)
    
    # static parameters for DreamDIA
    BM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/M2_170_12_11epochs_BM.h5")
    RM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/M2_170_12_11epochs_RM.h5")  
    n_lib_frags = 20
    n_self_frags = 50
    n_qt3_frags = 10
    n_ms1_frags = 10
    n_iso_frags = 20
    n_light_frags = 20
    model_cycles = 12
    feature_dimension = n_lib_frags * 3 + n_self_frags + n_qt3_frags + n_ms1_frags + n_iso_frags + n_light_frags
    TRFP_file = os.path.join(os.path.dirname(sys.argv[0]), "third_party/ThermoRawFileParser.exe")
    queue_size = 512 
    iso_range = 4
    apex_index_range = 1
    apex_indices = get_apex_indices(model_cycles, apex_index_range)
    irt_searching_cycles = 600
    n_output_progress_precursors = 4000
    n_chrom_writting_batch = 512
    rt_normalization_dir_suffix = "_dreamdia"
    sqdream_file_name = "dreamdia_scoring_profile.sqDream"
    mgr = multiprocessing.Manager()
    
    logger.info("Load spectral library and generate decoys: %s" % lib)  
    lib_cols, library = load_library_and_generate_decoys(lib, n_threads, seed, mz_min, mz_max, n_frags_each_precursor, decoy_method, logger)

    logger.info("Subsample peptides for RT normalization...") 
    irt_library = generate_endoIRT(lib_cols, library, n_irts, seed)
    irt_precursors, irt_chunk_indices = load_irt_precursors(irt_library, lib_cols, mz_min, mz_max, iso_range, n_threads)
          
    logger.info("Calculate theoretical fragment ions...")
    precursor_indices, chunk_indices = tear_library(library, lib_cols, n_threads)
    n_total_precursors = len(precursor_indices)
    
    archivers = []    
    precursor_lists = [mgr.list() for _ in range(n_threads)]
    
    for i, chunk_index in enumerate(chunk_indices):
        precursor_index = [precursor_indices[idx] for idx in chunk_index]
        p = multiprocessing.Process(target = load_precursors, 
                                    args = (library, lib_cols, precursor_index, precursor_lists[i], mz_min, mz_max, iso_range, ))
        archivers.append(p)
        p.daemon = True
        p.start()
    
    for p in archivers:
        p.join()

    logger.info("Load raw data files from %s" % os.path.abspath(file_dir))
    rawdata_files = check_raw_data_files(file_dir, logger)

    for file_count, rawdata_file in enumerate(rawdata_files):
        logger.info("Load (%d / %d) file: %s ..." % (file_count + 1, len(rawdata_files), rawdata_file))

        ms1, ms2, win_range, rawdata_prefix = convert_and_load_raw_data(rawdata_file, file_dir, TRFP_file, logger, mz_min, mz_max)
        if ms1 is None:
            continue
 
        logger.info("Perform RT normalization...")      
        
        rt_norm_dir = os.path.join(out, rawdata_prefix + rt_normalization_dir_suffix)
        
        irt_score_res_list = [mgr.list() for _ in range(n_threads)]
        extractors = []
        for p_id, coord in enumerate(irt_chunk_indices):
            p = multiprocessing.Process(target = score_irt, 
                                        args = (irt_score_res_list[p_id], [irt_precursors[i] for i in coord], ms1, ms2, win_range, irt_mode, 
                                                model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
                                                n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, 
                                                BM_model_file, feature_dimension, apex_indices, irt_searching_cycles, ))       
            p.daemon = True
            extractors.append(p)
            p.start()
        for p in extractors:
            p.join()
                        
        rt_model_params = fit_irt_model(irt_score_res_list, rt_norm_dir, irt_score_cutoff, irt_score_libcos_cutoff, irt_mode, rt_norm_model, seed)

        logger.info("Score peak groups...")
        sp_queue = mgr.JoinableQueue(queue_size)
        chrom_queue = mgr.JoinableQueue(queue_size)
        progress_queue = mgr.JoinableQueue(queue_size)
        
        extractors = []
        for i in range(n_threads):
            p = multiprocessing.Process(target = score_precursors, 
                                        args = (ms1, ms2, win_range, precursor_lists[i], chrom_queue, sp_queue, progress_queue, 
                                                n_cycles, model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
                                                n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, n_iso_frags, n_light_frags, ipf_scoring, 
                                                rt_norm_model, rt_model_params, BM_model_file, RM_model_file, apex_indices, feature_dimension, ))    

            p.daemon = True
            extractors.append(p)
            p.start()

        chrom_archiver = multiprocessing.Process(target = output_chromatograms, 
                                                 args = (chrom_queue, n_threads, rt_norm_dir, ipf_scoring, n_chrom_writting_batch, sqdream_file_name, logger, ))
        chrom_archiver.daemon = True
        chrom_archiver.start()
        
        sp_archiver = multiprocessing.Process(target = output_scoring_profiles, 
                                                 args = (sp_queue, n_threads, rt_norm_dir, ))
        sp_archiver.daemon = True
        sp_archiver.start()
        
        progressor = multiprocessing.Process(target = output_progress, 
                                             args = (progress_queue, n_total_precursors, n_output_progress_precursors, logger, ))
        progressor.daemon = True
        progressor.start()


        for ext in extractors:
            ext.join()       
        chrom_archiver.join()
        sp_archiver.join()
        progressor.join()

        logger.info("DreamDIA scoring done!")  

    logger.info("Done!")
