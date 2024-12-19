"""
╔═════════════════════════════════════════════════════╗
║                  dream_prophet.py                   ║
╠═════════════════════════════════════════════════════╣
║     Description: Main function of `dreamprophet`    ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os
import sys
import multiprocessing

import pandas as pd

from utils import get_dreamdia_logger, welcome, check_dream_score_files
from dream_prophet_utils import load_scoring_profiles_and_tear_into_chunks, load_precursor_ids_and_tear_into_chunks, get_peak_picking_single_run_results, collect_scoring_table
from multi_run_alignment import build_global_constraint, align_a_batch_of_precursors, get_peak_picking_cross_run_results, collect_scoring_table_multi_run, output_scoring_table
from statistical_analysis import dream_stats

def dream_prophet(
    dream_dir, 
    out, 
    n_threads,
    seed, 
    top_k, 
    disc_model, 
    disc_sample_rate, 
    fdr_precursor, 
    fdr_peptide, 
    fdr_protein, 
    dreamdialignr,
    r_home, 
    mra_algorithm, 
    global_constraint_type, 
    span_value, 
    distance_metric, 
    rt_tol, 
    exp_decay
):
    logger = get_dreamdia_logger()
    welcome(logger, "dream_prophet")

    if not os.path.exists(out):
        os.mkdir(out)

    # static parameters for DreamDIA
    sqdream_file_name = "dreamdia_scoring_profile.sqDream"
    single_run_scoring_table_name = "DreamDIA_single-run_scoring_table.tsv"
    multi_run_scoring_table_name = "DreamDIA_multi-run_scoring_table.tsv"
    rt_normalization_dir_suffix = "_dreamdia"
    build_mst_r_script = os.path.join(os.path.dirname(sys.argv[0]), "build_mst.R")
    dream_align_script = os.path.join(os.path.dirname(sys.argv[0]), "dream_align.R")
    n_total_precursors_batch = 20000
    delta_rt_weight = 0
    queue_size = 512 
    n_writting_batch = 20000
    peak_picking_mode = "average"
    all_result_name = "DreamDIA_all_results.tsv"
    precursor_result_name = "DreamDIA_precursor_fdr_results.tsv"
    peptide_result_name = "DreamDIA_peptide_fdr_results.tsv"
    protein_result_name = "DreamDIA_protein_fdr_results.tsv"
    mgr = multiprocessing.Manager()

    dream_dirs_run, sqdream_files = check_dream_score_files(dream_dir, sqdream_file_name, rt_normalization_dir_suffix, logger)
    runs_under_analysis = [run[:-len(rt_normalization_dir_suffix)] for run in dream_dirs_run]
    
    if not dreamdialignr:
        logger.info("DreamDIA: single-run analysis mode.")

        logger.info("Build scoring tables...")

        for idx, (run_dir, sqdream_file) in enumerate(zip(dream_dirs_run, sqdream_files)):     
            all_scoring_profiles, batch_precursors, batch_precursor_row_ids = load_scoring_profiles_and_tear_into_chunks(os.path.join(dream_dir, sqdream_file), n_threads)
            run_name = runs_under_analysis[idx]

            feature_queue = mgr.JoinableQueue(512)

            peak_pickers = []
            for batch_index, (batch_precursor, batch_precursor_row_id) in enumerate(zip(batch_precursors, batch_precursor_row_ids)):
                batch_scoring_profiles = all_scoring_profiles.iloc[batch_precursor_row_id, :]
                p = multiprocessing.Process(target = get_peak_picking_single_run_results, 
                                            args = (batch_scoring_profiles, feature_queue, run_name, top_k, ))
                peak_pickers.append(p)
                p.daemon = True
                p.start()

            
            scoring_table_collector = multiprocessing.Process(target = collect_scoring_table, 
                                                              args = (feature_queue, os.path.join(dream_dir, run_dir), single_run_scoring_table_name, n_threads, ))   

            scoring_table_collector.start()      

            for p in peak_pickers:
                p.join()
            scoring_table_collector.join()

            logger.info("(%d / %d) run: %s scoring table buiding done..." % (idx + 1, len(sqdream_files), run_name))

        scoring_table_list = [pd.read_csv(os.path.join(dream_dir, run_dir, single_run_scoring_table_name), sep = "\t") for run_dir in dream_dirs_run]
        logger.info("Start building discriminative model...")
        dream_stats(scoring_table_list, out, -1, logger, all_result_name, precursor_result_name, peptide_result_name, protein_result_name, fdr_precursor, fdr_peptide, fdr_protein, disc_model, disc_sample_rate, seed, "dreamdia")

    else:
        logger.info("DreamDIA: cross-run analysis mode.")

        if not os.path.exists(r_home):
            logger.error("Error: R home does not exist!")
            return
   
        logger.info("Build global constraint...")

        run_weights, merge_order, key_runs, global_fit = build_global_constraint(
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
        )
            
        logger.info("Start multi-run alignment...")
        logger.info(f"Multi-run alignment algorithm: {mra_algorithm}")

        alignment_queue = mgr.JoinableQueue(queue_size)
        feature_queue = mgr.JoinableQueue(queue_size)
        
        if mra_algorithm == "dialignr":
            os.environ['R_HOME'] = r_home
            import rpy2.robjects as ro            
            r = ro.r
            r['source'](dream_align_script)
            if global_constraint_type == "lowess":
                getAlignedTimesFast_function = ro.globalenv["dream_align_lowess"]
            else:
                getAlignedTimesFast_function = ro.globalenv["dream_align_linear"]
        else:
            getAlignedTimesFast_function = None

        precursors_in_batches = load_precursor_ids_and_tear_into_chunks(os.path.join(dream_dir, sqdream_files[0]), n_total_precursors_batch, runs_under_analysis)
        
        score_calculator = multiprocessing.Process(target = get_peak_picking_cross_run_results, 
                                                   args = (alignment_queue, feature_queue, runs_under_analysis, logger, top_k, rt_tol,
                                                           delta_rt_weight, run_weights, peak_picking_mode, ))
        score_calculator.daemon = True
        score_calculator.start()
        
        feature_archiver = multiprocessing.Process(target = collect_scoring_table_multi_run, 
                                                args = (feature_queue, runs_under_analysis, out, top_k, n_writting_batch, logger, ))
        feature_archiver.daemon = True
        feature_archiver.start()

        n_alignment_processes = max(1, n_threads - 2)
        for batch_index, batch_precursor in enumerate(precursors_in_batches):
            logger.info(f"Processing ({batch_index + 1} / {len(precursors_in_batches)}) batch...")

            aligners = []

            chunk_size = len(batch_precursor) // n_alignment_processes
            precursor_chunks = [batch_precursor[i:i + chunk_size] for i in range(0, len(batch_precursor), chunk_size)]

            for i, chunk in enumerate(precursor_chunks):
                p = multiprocessing.Process(target = align_a_batch_of_precursors, 
                                            args = (alignment_queue, dream_dir, runs_under_analysis, sqdream_files, chunk, global_fit, 
                                                    mra_algorithm, global_constraint_type, rt_tol, merge_order, key_runs, getAlignedTimesFast_function, ))
                
                aligners.append(p)
                p.start()

            for p in aligners:
                p.join()

            logger.info(f"({batch_index + 1} / {len(precursors_in_batches)}) batch alignment done!")
        
        alignment_queue.put(None)
        logger.info("All the alignment done!")

        score_calculator.join()
        feature_archiver.join()
        
        logger.info("Start building output scoring table...")
        scoring_table = output_scoring_table(out, multi_run_scoring_table_name)
        
        logger.info("Start building discriminative model...")
        dream_stats([scoring_table], out, -1, logger, all_result_name, precursor_result_name, peptide_result_name, protein_result_name, fdr_precursor, fdr_peptide, fdr_protein, disc_model, disc_sample_rate, seed, "dreamdialignr")
   
    logger.info("Done!")