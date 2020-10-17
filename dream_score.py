import sys
import os
import os.path
import logging
import numpy as np
import pandas as pd
import multiprocessing

import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from decoy_generator import generate_decoys
from rt_normalization import load_irt_precursors, extract_irt_xics, score_irt
from score_peak_groups import load_precursors, extract_precursors, score_batch
from dream_prophet import combine_res, dream_prophet
from utils import load_rawdata, endoIRT_generator, tear_library

def dream_score(file_dir, lib, win, out, n_threads, seed, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, fdr_precursor, fdr_protein, n_irt, top_k, n_cycles, n_frags_each_precursor, do_not_output_library, swath, model_cycles, n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, prophet_mode, disc_model, dream_indicators):
    logging.basicConfig(level = logging.INFO, format = "Dream-DIA: %(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.info("Welcome to Dream-DIA!")

    if not os.path.exists(out):
        os.mkdir(out)

    if swath:
        mz_unit = "Da"
        mz_tol_ms1 = 0.05
        mz_tol_ms2 = 0.05
        BM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/TTOF5600_11epochs.h5")
        RM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/TTOF5600_11epochs.aux.h5")
    else:
        BM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/Thermo_13epochs.h5")
        RM_model_file = os.path.join(os.path.dirname(sys.argv[0]), "models/Thermo_13epochs.aux.h5")
    
    iso_range = 4
    peak_index_range = 1
    score_cutoff = 0.9
    batch_size = 400
    
    logger.info("Load spectral library and generate decoys: %s" % lib)
    
    lib_cols, library = generate_decoys(lib, do_not_output_library, n_threads, seed, mz_min, mz_max, n_frags_each_precursor, logger)
    irt_library = endoIRT_generator(lib_cols, library, n_irt)
    irt_precursors, irt_chunk_indice = load_irt_precursors(irt_library, lib_cols, mz_min, mz_max, iso_range, n_threads)

    logger.info("Calculate m/z values of the ions...")

    precursor_indice, chunk_indice = tear_library(library, lib_cols, n_threads)
    n_total_precursors = len(precursor_indice)
    archivers = []
    mgr = multiprocessing.Manager()
    precursor_lists = [mgr.list() for _ in range(n_threads)]

    for i, chunk_index in enumerate(chunk_indice):
        precursor_index = [precursor_indice[idx] for idx in chunk_index]
        p = multiprocessing.Process(target = load_precursors, 
                                    args = (library, lib_cols, precursor_index, precursor_lists[i], mz_min, mz_max, iso_range, ))
        archivers.append(p)
        p.daemon = True
        p.start()

    for arc in archivers:
        arc.join()

    logger.info("Load raw data files from %s" % os.path.abspath(file_dir))
    rawdata_files = [i for i in sorted(os.listdir(file_dir)) if i.endswith(".mzML") or i.endswith(".mzXML") or i.endswith(".raw")]
    if not rawdata_files:
        raise Exception("Cannot find any valid MS raw data files.")
    tmp = rawdata_files[:]
    for rawdata_file in tmp:
        if rawdata_file.endswith(".raw"):
            if (rawdata_file[:-4] + ".mzML" in rawdata_files) or (rawdata_file[:-4] + ".mzXML" in rawdata_files):
                rawdata_files.remove(rawdata_file)
    n_rawdata_files = len(rawdata_files)

    logger.info("%d raw data file(s) in total: \n%s" % (n_rawdata_files, "\n".join(rawdata_files)))

    dream_score_res_files = []
    for file_count, rawdata_file in enumerate(rawdata_files):
        logger.info("Load (%d / %d) file: %s ..." % (file_count + 1, n_rawdata_files, rawdata_file))

        if rawdata_file.endswith(".raw"):
            logger.info("Perform data format conversion...")
            rawdata_prefix = rawdata_file[:-4]
            if sys.platform == "linux":
                convert_status = os.system("mono %s -i=%s -o=%s -f=1" % (os.path.join(os.path.dirname(sys.argv[0]), "third_party/ThermoRawFileParser.exe"), os.path.join(file_dir, rawdata_file), file_dir))
            else:
                convert_status = os.system("%s -i=%s -o=%s -f=1" % (os.path.join(os.path.dirname(sys.argv[0]), "third_party/ThermoRawFileParser.exe"), os.path.join(file_dir, rawdata_file), file_dir))

            if convert_status != 0:
                logger.info("Error!!!: File format conversion failed for %s" % rawdata_file)
                continue
            
            ms1, ms2, win_range = load_rawdata(os.path.join(file_dir, rawdata_prefix + ".mzML"), win, mz_min, mz_max)
        
        else:
            if rawdata_file.endswith(".mzML"):
                rawdata_prefix = rawdata_file[:-5]
            elif rawdata_file.endswith(".mzXML"):
                rawdata_prefix = rawdata_file[:-6]
            
            ms1, ms2, win_range = load_rawdata(os.path.join(file_dir, rawdata_file), win, mz_min, mz_max)

        logger.info("Perform RT normalization...")
        
        rt_norm_dir = os.path.join(out, rawdata_prefix + "_" + os.path.basename(lib)[:-4] + "_rt_norm")

        extract_queue = multiprocessing.JoinableQueue(n_threads)
        extractors = []
        for coord in irt_chunk_indice:
            p = multiprocessing.Process(target = extract_irt_xics, 
                                        args = (ms1, ms2, win_range, extract_queue, [irt_precursors[i] for i in coord], 
                                                model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, iso_range, 
                                                n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, ))                        
            p.daemon = True
            extractors.append(p)
            p.start()

        scorer = multiprocessing.Process(target = score_irt, args = (extract_queue, BM_model_file, rt_norm_dir, 
                                                                     n_threads, score_cutoff, seed, ))
        scorer.start()

        for p in extractors:
            p.join()
        scorer.join()

        irt_model_file = os.path.join(rt_norm_dir, "irt_model.txt")
        file_handle = open(irt_model_file, "r")
        irt_model = [float(v) for v in file_handle.read().splitlines()]
        slope, intercept = irt_model[0], irt_model[1]

        logger.info("Score peak groups...")

        dream_score_res_file = os.path.join(out, rawdata_prefix + "_" + os.path.basename(lib)[:-4] + ".dream_score.tsv")
        matrix_queue = multiprocessing.JoinableQueue(256)
        extractors = []

        for i in range(n_threads):
            p = multiprocessing.Process(target = extract_precursors, 
                                        args = (ms1, ms2, win_range, precursor_lists[i], matrix_queue, 
                                                n_cycles, model_cycles, mz_unit, mz_min, mz_max, mz_tol_ms1, 
                                                mz_tol_ms2, iso_range, n_lib_frags, n_self_frags, 
                                                n_qt3_frags, n_ms1_frags, peak_index_range, slope, intercept, i, ))    
            p.daemon = True
            extractors.append(p)
            p.start()

        scorer = multiprocessing.Process(target = score_batch, 
                                         args = (matrix_queue, lib_cols, BM_model_file, RM_model_file, dream_score_res_file, rawdata_file, top_k, 
                                                 n_threads, batch_size, n_total_precursors, logger, ))
        scorer.daemon = True
        scorer.start()

        for ext in extractors:
            ext.join()
        scorer.join()

        dream_score_res_files.append(dream_score_res_file)

        logger.info("Dream-DIA scoring done!")
        
        if prophet_mode == "local":
            logger.info("Build discriminant model...")
            disc_dir = os.path.join(out, rawdata_prefix + "_" + os.path.basename(lib)[:-4] + "_dream_prophet")
            dream_score_res = pd.read_csv(dream_score_res_file, sep = "\t")
            dream_prophet(dream_score_res, lib_cols, disc_model, top_k, n_threads, seed, dream_indicators, disc_dir, logger, fdr_precursor, fdr_protein)

        for precursor_list in precursor_lists:
            for precursor in precursor_list:
                precursor.clear()
    
    if prophet_mode == "global":
        logger.info("Build discriminant model...")
        disc_dir = os.path.join(out, "dream_prophet")
        dream_score_res = combine_res(dream_score_res_files, lib_cols)
        dream_prophet(dream_score_res, lib_cols, disc_model, top_k, n_threads, seed, dream_indicators, disc_dir, logger, fdr_precursor, fdr_protein)
        
    logger.info("Done!")