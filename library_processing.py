"""
╔═════════════════════════════════════════════════════╗
║               library_processing.py                 ║
╠═════════════════════════════════════════════════════╣
║       Description: Utility functions for library    ║
║                   file processing                   ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║             Contact: mingxuan.gao@utoronto.ca       ║
╚═════════════════════════════════════════════════════╝
"""

import os
import sys
import random
import multiprocessing

import numpy as np
import pandas as pd

from utils import flatten_list, get_precursor_indices
from mz_calculator import calc_fragment_mz

class Sequence_processor:
    """Generate decoys given an amino acid sequence without modifications."""
    def __init__(self, seq):
        self.seq = list(seq)

    def shuffle_seq(self, seed = None):
        """
        Fisher-Yates algorithm. Modified from PECAN's decoyGenerator.py
        """

        l = self.seq.copy()
        random.seed(seed)
        for i in range(len(l) - 1, 0, -1):
            j = int(random.random() * (i + 1))
            if i == j:
                continue
            else:
                (l[i], l[j]) = (l[j], l[i])

        return l

    def reverse_seq(self):
        return self.seq[::-1]

    def shift_seq(self):
        i = len(self.seq) // 2
        return self.seq[i::] + self.seq[:i:]

    def mutate_seq(self):
        """Decoy generator in DIA-NN"""
        mutations = {"G" : "L", 
                    "A" : "L", 
                    "V" : "L", 
                    "L" : "V", 
                    "I" : "V", 
                    "F" : "L", 
                    "M" : "L", 
                    "P" : "L", 
                    "W" : "L", 
                    "S" : "T", 
                    "C" : "S", 
                    "T" : "S", 
                    "Y" : "S", 
                    "H" : "S", 
                    "K" : "L", 
                    "R" : "L", 
                    "Q" : "N", 
                    "E" : "D", 
                    "N" : "Q", 
                    "D" : "E"}
        return [self.seq[0], mutations[self.seq[1]]] + self.seq[2:-2] + [mutations[self.seq[-2]], self.seq[-1]]
    
def get_modification_indices(modified_sequence):
    """
    Parses the input modified peptide sequence and extracts the positions and contents of all the modifications.

    Args:
        modified_sequence (str): The input string representing the modified peptide sequence.

    Returns:
        tuple: A tuple containing two lists:
            - positions (list of int): The positions of modifications.
            - modifications (list of str): The modifications extracted from the input string.
    """

    cursor, lock = -1, 0
    poses, mods = [], []
    for i, lett in enumerate(modified_sequence):
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
            mod += modified_sequence[i]
    return poses, mods

def load_library(library_file):
    """
    Load the spectral library from a file.
    Necessary file: lib_col_settings.txt in the root directory of DreamDIA software package.
    
    Args:
        library_file (str): Path to the spectral library file (.tsv or .csv).
        
    Returns:
        tuple: A dictionary of library columns and the library DataFrame.
        
    Raises:
        ValueError: If the file format is not supported or necessary columns are missing.
    """
    # Load the library based on file extension
    if library_file.endswith(".tsv"):
        library = pd.read_csv(library_file, sep="\t")
    elif library_file.endswith(".csv"):
        library = pd.read_csv(library_file)
    else:
        raise ValueError(f"Invalid spectral library format: {library_file}. Only .tsv and .csv formats are supported.")
    
    # Load the library column settings
    lib_cols = {}
    lib_col_settings_path = os.path.join(os.path.dirname(sys.argv[0]), "lib_col_settings.txt")
    
    with open(lib_col_settings_path) as f:
        for line in f:
            record = line.strip()
            if record and not record.startswith("#"):
                key, value = map(str.strip, record.split("="))
                lib_cols[key] = value
    
    # Check necessary columns
    necessary_columns = list(lib_cols.values())
    real_columns = list(library.columns)
    no_columns = [col for col in necessary_columns if col not in real_columns]
    
    if no_columns:
        raise ValueError(f"Cannot find column(s) '{'; '.join(no_columns)}' in the spectral library.")
    
    lib_cols["DECOY_OR_NOT_COL"] = "decoy"
    
    return lib_cols, library

def check_full_sequence(library, id_column, full_seq_column):
    """
    Check if the full sequence column matches the ID column in the library.
    
    Args:
        library (pd.DataFrame): The spectral library DataFrame.
        id_column (str): Column name for precursor IDs.
        full_seq_column (str): Column name for full sequences.
        
    Returns:
        list: List of abnormal precursor IDs.
    """
    return [
        pep_id for pep_id, full_seq in zip(library[id_column], library[full_seq_column])
        if not pep_id.startswith("DECOY") and pep_id.strip().split("_")[1] != full_seq
    ]

def correct_full_sequence(library, id_column, full_seq_column):
    """
    Correct the full sequence column in the library.
    
    Args:
        library (pd.DataFrame): The spectral library DataFrame.
        id_column (str): Column name for IDs.
        full_seq_column (str): Column name for full sequences.
        
    Returns:
        pd.DataFrame: The corrected library DataFrame.
    """
    # Identify abnormal records
    abnormal_records = check_full_sequence(library, id_column, full_seq_column)
    
    # Correct the full sequence column for abnormal records
    library.loc[library[id_column].isin(abnormal_records), full_seq_column] = \
        library[library[id_column].isin(abnormal_records)][id_column].apply(lambda x: x.strip().split("_")[1])
    
    return library

def tear_library(library, lib_cols, n_threads):
    """
    Split the library into chunks based on the number of threads.

    Parameters:
    - library: pd.DataFrame containing the spectral library.
    - lib_cols: Dictionary with column names.
    - n_threads: Number of threads to split the data into.

    Returns:
    - tuple: (precursor_indices, chunk_indices)
        - precursor_indices: List of lists containing the indices of each unique precursor.
        - chunk_indices: List of lists containing the indices for each chunk.
    
    Example usage:
    >>> library = pd.DataFrame({'precursor_id': ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'E', 'E', 'F', 'F', 'G', 'H', 'H']})
    >>> lib_cols = {"PRECURSOR_ID_COL": "precursor_id"}
    >>> n_threads = 3
    >>> tear_library(library, lib_cols, n_threads)
    ([[0, 1], [2, 3, 4], [5, 6], [7, 8], [9, 10, 11, 12], [13, 14], [15], [16, 17]], 
     [[0, 1, 2], [3, 4, 5], [6, 7]])
    """
    # Get the indices of each unique precursor
    precursor_indices = get_precursor_indices(library[lib_cols["PRECURSOR_ID_COL"]])
    n_precursors = len(precursor_indices)  # Number of precursors in total

    # Calculate the chunk indices
    chunk_indices = np.array_split(np.arange(n_precursors), n_threads)
    chunk_indices = [list(chunk) for chunk in chunk_indices]

    return precursor_indices, chunk_indices

def decoy_generator(
    library, 
    lib_cols, 
    decoy_method, 
    precursor_indices, 
    original_colnames, 
    result_collector, 
    fixed_colnames, 
    seed):
    """
    Generate decoy peptides and their associated data from a given library and a subset of target peptides.

    Parameters:
    - library (pd.DataFrame): The data frame containing peptide library records.
    - lib_cols (dict): A dictionary mapping column types to their respective column names in the library.
    - decoy_method (str): The method used to generate decoy peptides. Options include "shuffle", "pseudo_reverse", "shift", "reverse", and "mutate".
    - precursor_indices (list): List of indices identifying precursors in the library to be processed.
    - original_colnames (list): List of original column names to be preserved in the output.
    - result_collector (list): A list that collects results for further processing.
    - fixed_colnames (list): List of column names that are fixed and should be included in the final result.
    - seed (int): Seed for random number generator to ensure reproducibility.

    Generates:
    - Decoy peptides with modified sequences based on the specified method.
    - Associated metadata such as product m/z, transition names, peptide sequences, and protein names.
    - Appends the results to the result_collector list.
    """

    product_mz, peptide_sequence, full_uniMod_peptide_name = [], [], []
    transition_group_id, decoy, protein_name = [], [], []
    transition_name, peptide_group_label = [], []

    valid_indices = []

    for pep in precursor_indices:    
        target_record = library.iloc[pep, :]
        
        if ("decoy" in list(library.columns)) and (list(target_record["decoy"])[0] == 1):
            continue
        
        valid_indices.extend(pep)
        
        target_fullseq = list(target_record[lib_cols["FULL_SEQUENCE_COL"]])[0]
        target_pureseq = list(target_record[lib_cols["PURE_SEQUENCE_COL"]])[0]

        if decoy_method in ["shuffle", "pseudo_reverse", "shift"]:
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
                
            mod_indice, mod_list = get_modification_indices(sort_base)
                
            if KR_end or KR_mod_end:
                pure_seq_list = [i for i in target_pureseq[:-1]]
            else:
                pure_seq_list = [i for i in target_pureseq]
            
            seq_list = pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                seq_list[mod_id] += mod
            
            seq_processor = Sequence_processor(list(range(len(seq_list))))
            if decoy_method == "shuffle":
                shuffled_indice = seq_processor.shuffle_seq(seed)
            elif decoy_method == "pseudo_reverse":
                shuffled_indice = seq_processor.reverse_seq()
            elif decoy_method == "shift":
                shuffled_indice = seq_processor.shift_seq()
            
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

        elif decoy_method == "reverse":
            unimod5 = False
            
            sort_base = target_fullseq[:]
            if sort_base.startswith("(UniMod:5)"):
                unimod5 = True
                sort_base = sort_base[10:]
                
            mod_indice, mod_list = get_modification_indices(sort_base)
                
            pure_seq_list = [i for i in target_pureseq]
            
            seq_list = pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                seq_list[mod_id] += mod
            
            seq_processor = Sequence_processor(list(range(len(seq_list))))
            shuffled_indice = seq_processor.reverse_seq()

            decoy_fullseq = "".join([seq_list[i] for i in shuffled_indice])
            decoy_pureseq = "".join([pure_seq_list[i] for i in shuffled_indice])
               
            if unimod5:
                decoy_fullseq = "(UniMod:5)" + decoy_fullseq

        elif decoy_method == "mutate":
            unimod5 = False
            
            sort_base = target_fullseq[:]
            if sort_base.startswith("(UniMod:5)"):
                unimod5 = True
                sort_base = sort_base[10:]
                
            mod_indice, mod_list = get_modification_indices(sort_base)
                
            pure_seq_list = [i for i in target_pureseq]
            seq_processor = Sequence_processor(pure_seq_list)
            mutated_pure_seq_list = seq_processor.mutate_seq()
            
            mutated_seq_list = mutated_pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                mutated_seq_list[mod_id] += mod

            decoy_fullseq = "".join(mutated_seq_list)
            decoy_pureseq = "".join(mutated_pure_seq_list)

            if unimod5:
                decoy_fullseq = "(UniMod:5)" + decoy_fullseq
      
        for charge, tp, series in zip(target_record[lib_cols["FRAGMENT_CHARGE_COL"]], target_record[lib_cols["FRAGMENT_TYPE_COL"]], target_record[lib_cols["FRAGMENT_SERIES_COL"]]):
            product_mz.append(calc_fragment_mz(decoy_fullseq, decoy_pureseq, charge, "%s%d" % (tp, series)))
            peptide_sequence.append(decoy_pureseq)
            full_uniMod_peptide_name.append(decoy_fullseq)
        
        # Process the two columns that usually exist in OpenSWATH-style libraries
        if "transition_name" in original_colnames: 
            transition_name.extend(["DECOY_" + list(target_record["transition_name"])[0]] * target_record.shape[0])
        if "PeptideGroupLabel" in original_colnames:
            peptide_group_label.extend(["DECOY_" + list(target_record["PeptideGroupLabel"])[0]] * target_record.shape[0])
            
        transition_group_id.extend(["DECOY_" + list(target_record[lib_cols["PRECURSOR_ID_COL"]])[0]] * target_record.shape[0])
        decoy.extend([1] * target_record.shape[0])
        protein_name.extend(["DECOY_" + list(target_record[lib_cols["PROTEIN_NAME_COL"]])[0]] * target_record.shape[0])

    result_collector.append([product_mz, 
                             peptide_sequence, 
                             full_uniMod_peptide_name, 
                             transition_group_id, 
                             decoy, 
                             protein_name, 
                             transition_name, 
                             peptide_group_label, 
                             library.iloc[valid_indices, :].loc[:, fixed_colnames]])

def collapse_decoy_generation_results(result_collectors, library, lib_cols, original_colnames):
    """
    Collapse decoy generation results into a single library DataFrame.

    Parameters:
    - result_collectors (list): List of result collector lists from the decoy generation process.
    - library (pd.DataFrame): The original peptide library.
    - lib_cols (dict): A dictionary mapping column types to their respective column names in the library.
    - original_colnames (list): List of original column names to be preserved in the final DataFrame.

    Returns:
    - library_with_decoys (pd.DataFrame): The original library combined with the generated decoys.
    """

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

    modified_part.reset_index(drop = True, inplace = True)
    fixed_part.reset_index(drop = True, inplace = True)
    
    if "decoy" in original_colnames:
        decoy_library = pd.concat([modified_part, fixed_part], axis = 1).loc[:, original_colnames]
    else:
        decoy_library = pd.concat([modified_part, fixed_part], axis = 1).loc[:, original_colnames + ["decoy"]]
        library["decoy"] = np.zeros(library.shape[0], dtype = int)

    library_with_decoys = pd.concat([library, decoy_library])

    return library_with_decoys

def filter_library(library, lib_cols, mz_min, mz_max, n_frags_each_precursor): 
    """
    Filter the peptide library based on precursor and fragment m/z ranges and the number of fragments for each precursor.

    Parameters:
    - library (pd.DataFrame): The original peptide library.
    - lib_cols (dict): A dictionary mapping column types to their respective column names in the library.
    - mz_min (float): The minimum m/z value for filtering.
    - mz_max (float): The maximum m/z value for filtering.
    - n_frags_each_precursor (int): The minimum number of fragments each precursor must have.

    Returns:
    - filtered_library (pd.DataFrame): The filtered peptide library.
    """

    filtered_library = library[(library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
    filtered_library = filtered_library[(filtered_library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (filtered_library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]
    filtered_library.reset_index(inplace = True, drop = True)

    precursor_indices = get_precursor_indices(filtered_library[lib_cols["PRECURSOR_ID_COL"]].values)
    too_few_indice = flatten_list([i for i in precursor_indices if len(i) < n_frags_each_precursor])
    filtered_library.drop(too_few_indice, inplace = True)
    
    filtered_library = filtered_library.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], lib_cols["LIB_INTENSITY_COL"]], ascending = [True, False])
    filtered_library.reset_index(inplace = True, drop = True)

    return filtered_library

def load_library_and_generate_decoys(library_file, n_threads, seed, mz_min, mz_max, n_frags_each_precursor, decoy_method, logger):
    """
    Generate a library with decoys and save it in the same directory as the original library.

    Parameters:
    - library_file (str): Path to the original library file.
    - n_threads (int): Number of threads to use for parallel processing.
    - seed (int): Seed for random number generation to ensure reproducibility.
    - mz_min (float): Minimum m/z value for filtering.
    - mz_max (float): Maximum m/z value for filtering.
    - n_frags_each_precursor (int): Minimum number of fragments each precursor must have.
    - decoy_method (str): Method to generate decoy peptides.
    - logger (logging.Logger): Logger for logging information.

    Returns:
    - lib_cols (dict): Dictionary mapping column types to their respective column names in the library.
    - library_with_decoys (pd.DataFrame): The original library combined with the generated decoys.
    """

    output_file = os.path.join(os.path.dirname(library_file), os.path.basename(library_file)[:-4] + ".DreamDIA.with_decoys.tsv")
    
    # If there has already been a library, load it and return.
    if os.path.exists(output_file):
        lib_cols, library = load_library(output_file)
        return lib_cols, library

    # Load library and check the columns needed
    lib_cols, library = load_library(library_file)
    
    # Correct those records in the library where the sequence in XXX_PEPTIDESEQ_3 is not the same as the corresponding one in sequence column
    library = correct_full_sequence(library, lib_cols["PRECURSOR_ID_COL"], lib_cols["FULL_SEQUENCE_COL"])
    
    # Filter m/z values of precursors and fragment ions 
    library = library[(library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
    library = library[(library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]
    library.reset_index(drop = True, inplace = True)

    # Discard precursors with too few fragment ions
    precursor_indices = get_precursor_indices(library[lib_cols["PRECURSOR_ID_COL"]])
    too_few_indices = flatten_list([i for i in precursor_indices if len(i) < n_frags_each_precursor])
    library.drop(too_few_indices, inplace = True)
    library.reset_index(drop = True, inplace = True)
    
    # Tear the library into pieces for multiprocessing run
    precursor_indices, chunk_indices = tear_library(library, lib_cols, n_threads)

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
            if decoy_types[1] > 0.9 * decoy_types[0]:
                logger.info("The spectral library has already got enough decoys, so DreamDIA will not generate more.")

                library.to_csv(output_file, sep = "\t", index = False)
                return lib_cols, library

    # Generate decoys
    generators = []
    mgr = multiprocessing.Manager()
    result_collectors = [mgr.list() for _ in range(n_threads)]

    for i, chunk_index in enumerate(chunk_indices):
        precursor_indices_of_one_chunk = [precursor_indices[idx] for idx in chunk_index]
        p = multiprocessing.Process(target = decoy_generator, 
                                    args = (library, lib_cols, decoy_method, precursor_indices_of_one_chunk, 
                                            original_colnames, result_collectors[i], fixed_colnames, seed, ))
        generators.append(p)
        p.daemon = True
        p.start()

    for p in generators:
        p.join()

    library_with_decoys = collapse_decoy_generation_results(result_collectors, library, lib_cols, original_colnames)
    library_with_decoys = filter_library(library_with_decoys, lib_cols, mz_min, mz_max, n_frags_each_precursor)

    if not os.path.exists(output_file):
        library_with_decoys.to_csv(output_file, index = False, sep = "\t")

    return lib_cols, library_with_decoys