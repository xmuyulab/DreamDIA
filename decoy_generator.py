import os.path
import random
import multiprocessing
import pandas as pd

from utils import load_library, correct_full_sequence, get_precursor_indice, tear_library, flatten_list
from mz_calculator import calc_fragment_mz

def shuffle_seq(seq = None, seed = None):
    """Fisher-Yates algorithm. Modified from PECAN's decoyGenerator.py"""
    if seq is None:
        return None
    else:
        l = list(seq)
        random.seed(seed)
        for i in range(len(l) - 1, 0, -1) :
            j = int(random.random() * (i + 1))
            if i == j:
                continue
            else:
                (l[i], l[j]) = (l[j], l[i])

        return tuple(l)

def get_mod_indice(sort_base):
    cursor, lock = -1, 0
    poses, mods = [], []
    for i, lett in enumerate(sort_base):
        if lett == "(":
            lock = 1
            poses.append(cursor)
            mod = ""
        elif lett == ")":
            lock = 0
            cursor -= 1
            mods.append(mod + ")")
        if not lock:
            cursor += 1
        else:
            mod += sort_base[i]
    return poses, mods

def decoy_generator(library, lib_cols, precursor_indice, original_colnames, result_collector, fixed_colnames, seed):
    product_mz, peptide_sequence, full_uniMod_peptide_name = [], [], []
    transition_group_id, decoy, protein_name = [], [], []
    transition_name, peptide_group_label = [], []

    valid_indice = []

    for idx, pep in enumerate(precursor_indice):    
        target_record = library.iloc[pep, :]
        
        if ("decoy" in list(library.columns)) and (list(target_record["decoy"])[0] == 1):
            continue
        
        valid_indice.extend(pep)
        
        target_fullseq = list(target_record[lib_cols["FULL_SEQUENCE_COL"]])[0]
        target_pureseq = list(target_record[lib_cols["PURE_SEQUENCE_COL"]])[0]
        unimod5, KR_end, KR_mod_end = False, False, False
        
        sort_base = target_fullseq[:]
        if sort_base.startswith("(UniMod:5)"):
            unimod5 = True
            sort_base = sort_base[10:]
        if sort_base[-1] in ["K", "R"]:
            KR_end = sort_base[-1]
            sort_base = sort_base[:-1]
        elif (sort_base.endswith("(UniMod:259)") or sort_base.endswith("(UniMod:267)")):
            KR_mod_end = sort_base[-13:]
            sort_base = sort_base[:-13]
            
        mod_indice, mod_list = get_mod_indice(sort_base)
            
        if KR_end or KR_mod_end:
            pure_seq_list = [i for i in target_pureseq[:-1]]
        else:
            pure_seq_list = [i for i in target_pureseq]
        
        seq_list = pure_seq_list[:]
        for mod_id, mod in zip(mod_indice, mod_list):
            seq_list[mod_id] += mod
        
        shuffled_indice = shuffle_seq([i for i in range(len(seq_list))], seed = seed)
        decoy_fullseq = "".join([seq_list[i] for i in shuffled_indice])
        decoy_pureseq = "".join([pure_seq_list[i] for i in shuffled_indice])
        
        if unimod5:
            decoy_fullseq = "(UniMod:5)" + decoy_fullseq
        if KR_end:
            decoy_fullseq += KR_end
            decoy_pureseq += KR_end
        elif KR_mod_end:
            decoy_fullseq += KR_mod_end
            decoy_pureseq += KR_mod_end[0]
        
        for charge, tp, series in zip(target_record[lib_cols["FRAGMENT_CHARGE_COL"]], target_record[lib_cols["FRAGMENT_TYPE_COL"]], target_record[lib_cols["FRAGMENT_SERIES_COL"]]):
            product_mz.append(calc_fragment_mz(decoy_fullseq, decoy_pureseq, charge, "%s%d" % (tp, series)))
            peptide_sequence.append(decoy_pureseq)
            full_uniMod_peptide_name.append(decoy_fullseq)
        
        if "transition_name" in original_colnames: 
            transition_name.extend(["DECOY_" + list(target_record["transition_name"])[0]] * target_record.shape[0])
        if "PeptideGroupLabel" in original_colnames:
            peptide_group_label.extend(["DECOY_" + list(target_record["PeptideGroupLabel"])[0]] * target_record.shape[0])
            
        transition_group_id.extend(["DECOY_" + list(target_record[lib_cols["PRECURSOR_ID_COL"]])[0]] * target_record.shape[0])
        decoy.extend([1] * target_record.shape[0])
        protein_name.extend(["DECOY_" + list(target_record[lib_cols["PROTEIN_NAME_COL"]])[0]] * target_record.shape[0])

    result_collector.append([product_mz, peptide_sequence, full_uniMod_peptide_name, 
                             transition_group_id, decoy, protein_name, transition_name, 
                             peptide_group_label, library.iloc[valid_indice, :].loc[:, fixed_colnames]])

def generate_decoys(lib, do_not_output_library, n_threads, seed, mz_min, mz_max, n_frags_each_precursor, logger):
    output_filename = os.path.join(os.path.dirname(lib), os.path.basename(lib)[:-4] + ".DreamDIA.with_decoys.tsv")

    lib_cols, library = load_library(lib)
    
    library = correct_full_sequence(library, lib_cols["PRECURSOR_ID_COL"], lib_cols["FULL_SEQUENCE_COL"])
    
    library = library[(library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
    library = library[(library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]
    library.index = [i for i in range(library.shape[0])]

    precursor_indice = get_precursor_indice(library[lib_cols["PRECURSOR_ID_COL"]])
    too_few_indice = flatten_list([i for i in precursor_indice if len(i) < n_frags_each_precursor])
    library.drop(too_few_indice, inplace = True)
    library.index = [i for i in range(library.shape[0])]
    precursor_indice, chunk_indice = tear_library(library, lib_cols, n_threads)

    original_colnames = list(library.columns)
    modifiable_colnames = [lib_cols["FRAGMENT_MZ_COL"], 
                           lib_cols["PURE_SEQUENCE_COL"], 
                           lib_cols["FULL_SEQUENCE_COL"], 
                           lib_cols["PRECURSOR_ID_COL"], 
                           lib_cols["PROTEIN_NAME_COL"], 
                           "transition_name", "decoy", "PeptideGroupLabel"]
    fixed_colnames = [i for i in original_colnames if i not in modifiable_colnames]

    if "decoy" in original_colnames:
        decoy_types = library["decoy"].value_counts()
        if 0 in decoy_types and 1 in decoy_types:
            if decoy_types[1] > 0.5 * decoy_types[0]:
                logger.info("The spectral library has enough decoys, so DreamDIA-XMBD will not generate more.")
                if not do_not_output_library:
                    library.to_csv(output_filename, sep = "\t", index = False)
                return lib_cols, library

    generators = []
    mgr = multiprocessing.Manager()
    result_collectors = [mgr.list() for _ in range(n_threads)]

    for i, chunk_index in enumerate(chunk_indice):
        precursor_index = [precursor_indice[idx] for idx in chunk_index]
        p = multiprocessing.Process(target = decoy_generator, 
                                    args = (library, lib_cols, precursor_index, original_colnames, result_collectors[i], fixed_colnames, seed, ))
        generators.append(p)
        p.daemon = True
        p.start()

    for p in generators:
        p.join()

    product_mz = flatten_list([collector[0][0] for collector in result_collectors])
    peptide_sequence = flatten_list([collector[0][1] for collector in result_collectors])
    full_uniMod_peptide_name = flatten_list([collector[0][2] for collector in result_collectors])
    transition_group_id = flatten_list([collector[0][3] for collector in result_collectors])
    decoy = flatten_list([collector[0][4] for collector in result_collectors])
    protein_name = flatten_list([collector[0][5] for collector in result_collectors])
    transition_name = flatten_list([collector[0][6] for collector in result_collectors])
    peptide_group_label = flatten_list([collector[0][7] for collector in result_collectors])
    fixed_part = pd.concat([collector[0][8] for collector in result_collectors])

    modified_part = pd.DataFrame({lib_cols["FRAGMENT_MZ_COL"] : product_mz, 
                                  lib_cols["PURE_SEQUENCE_COL"] : peptide_sequence, 
                                  lib_cols["FULL_SEQUENCE_COL"] : full_uniMod_peptide_name, 
                                  lib_cols["PRECURSOR_ID_COL"] : transition_group_id, 
                                  lib_cols["DECOY_OR_NOT_COL"] : decoy, 
                                  lib_cols["PROTEIN_NAME_COL"] : protein_name})
    if "transition_name" in original_colnames:
        modified_part["transition_name"] = transition_name
    if "PeptideGroupLabel" in original_colnames:
        modified_part["PeptideGroupLabel"] = peptide_group_label

    modified_part.index = [nn for nn in range(modified_part.shape[0])]
    fixed_part.index = [nn for nn in range(fixed_part.shape[0])]
    
    if "decoy" in original_colnames:
        decoy_data = pd.concat([modified_part, fixed_part], axis = 1).loc[:, original_colnames]
    else:
        decoy_data = pd.concat([modified_part, fixed_part], axis = 1).loc[:, original_colnames + ["decoy"]]
        library["decoy"] = [0 for _ in range(library.shape[0])]

    library_with_decoys = pd.concat([library, decoy_data])
    library_with_decoys = library_with_decoys.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], lib_cols["LIB_INTENSITY_COL"]], ascending = [True, False])
    library_with_decoys.index = [i for i in range(library_with_decoys.shape[0])]

    if (not do_not_output_library) and (not os.path.exists(output_filename)):
        library_with_decoys.to_csv(output_filename, index = False, sep = "\t")
    return lib_cols, library_with_decoys