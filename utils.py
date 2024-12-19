"""
╔═════════════════════════════════════════════════════╗
║                     utils.py                        ║
╠═════════════════════════════════════════════════════╣
║           Description: Utility functions            ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║             Contact: mingxuan.gao@utoronto.ca       ║
╚═════════════════════════════════════════════════════╝
"""

import os
import bisect
import logging
from typing import Tuple, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from art import logo

def get_dreamdia_logger():
    logging.basicConfig(level = logging.INFO, format = "DreamDIA: %(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    return logger

def welcome(logger, function_name):
    logger.info(logo)
    if function_name == "dream_score":
        logger.info("Welcome to DreamDIA: peak scoring module!")
    else:
        logger.info("Welcome to DreamDIA: peak identification and quantification module!")

def check_raw_data_files(file_dir, logger):
    rawdata_files = [i for i in sorted(os.listdir(file_dir)) if i.endswith(".mzML") or i.endswith(".mzXML") or i.endswith(".raw")]
    if not rawdata_files:
        raise Exception("Cannot find any valid MS raw data files.")
    tmp = rawdata_files[:]
    for rawdata_file in tmp:
        if rawdata_file.endswith(".raw"):
            if (rawdata_file[:-4] + ".mzML" in rawdata_files) or (rawdata_file[:-4] + ".mzXML" in rawdata_files):
                rawdata_files.remove(rawdata_file)
    
    n_rawdata_files = len(rawdata_files)
    
    if n_rawdata_files <= 10:
        logger.info("%d raw data file(s) in total: \n\t%s" % (n_rawdata_files, "\n\t".join(rawdata_files)))
    else:
        logger.info("%d raw data file(s) in total: \n\t%s" % (n_rawdata_files, "\n\t".join(rawdata_files[:10])))
        logger.info("\t...")   

    return rawdata_files

def check_dream_score_files(dream_dir, sqdream_file_name, rt_normalization_dir_suffix, logger):
    dream_dirs_run = [i for i in sorted(os.listdir(dream_dir)) if i.endswith(rt_normalization_dir_suffix)]
    if not dream_dirs_run:
        raise Exception("Cannot find any valid DreamDIA files.")
    
    sqdream_files = [os.path.join(i, sqdream_file_name) for i in dream_dirs_run]
    invalid_runs = []
    for dir_run, file in zip(dream_dirs_run, sqdream_files):
        if not os.path.exists(os.path.join(dream_dir, file)):
            invalid_runs.append(dir_run)
    if invalid_runs:
        for run in invalid_runs:
            logger.error("Cannot find DreamDIA scoring profile data for run: %s" % run)
        raise Exception("Cannot find DreamDIA scoring profile data!")
    
    if len(dream_dirs_run) <= 10:
        logger.info("%d run(s) in total: \n\t%s" % (len(dream_dirs_run), "\n\t".join(dream_dirs_run)))
    else:
        logger.info("%d run(s) in total: \n\t%s" % (len(dream_dirs_run), "\n\t".join(dream_dirs_run[:10])))
        logger.info("\t...")   

    return dream_dirs_run, sqdream_files

def flatten_list(alist):
    """
    Flatten a list of lists.
    
    Args:
        alist (list of lists): The list to be flattened.
        
    Returns:
        list: The flattened list.
    """
    return [item for sublist in alist for item in sublist]

def get_precursor_indices(precursor_ids):
    """
    Get the indices of each unique precursor in the list of precursor IDs.

    Parameters:
    - precursor_ids: List of precursor IDs.

    Returns:
    - List of lists containing the indices of each unique precursor.

    Example usage:
    >>> precursor_ids = ["A", "A", "B", "B", "B", "C", "C", "A"]
    >>> get_precursor_indices(precursor_ids)
    [[0, 1], [2, 3, 4], [5, 6], [7]]
    """
    precursor_indices = []
    last_precursor = None
    current_indices = []

    for index, precursor in enumerate(precursor_ids):
        if precursor != last_precursor:
            if current_indices:
                precursor_indices.append(current_indices)
            current_indices = [index]
        else:
            current_indices.append(index)
        last_precursor = precursor

    if current_indices:
        precursor_indices.append(current_indices)

    return precursor_indices

def get_apex_indices(n_cycles, peak_apex_range):
    """
    Generate a list of indices centered around the middle of a list with length n_cycles.
    
    Parameters:
    - n_cycles (int): The total number of cycles.
    - peak_apex_range (int): The number of indices around the apex to include.
    
    Returns:
    - list: List of indices centered around the middle.
    """
    middle_index = n_cycles // 2
    half_range = peak_apex_range // 2

    # If peak_apex_range is odd, the range is symmetric around the middle_index
    if peak_apex_range % 2 == 0:
        start_index = middle_index - half_range
        end_index = middle_index + half_range
    else:
        start_index = middle_index - half_range
        end_index = middle_index + half_range + 1

    # Ensure indices are within bounds
    start_index = max(start_index, 0)
    end_index = min(end_index, n_cycles)

    return list(range(start_index, end_index))

def calc_win_id(precursor_mz, win_range):
    """
    Calculate the window ID for a given precursor m/z value.

    Parameters:
    - precursor_mz (float): The m/z value of the precursor.
    - win_range (np.ndarray): A 2D array where each row represents a window range with 
                              the first column as the start of the range.

    Returns:
    - int: The window ID for the given precursor m/z value.

    Example usage:
    >>> win_range = np.array([[100, 200], [200, 300], [300, 400]])
    >>> precursor_mz = 250
    >>> calc_win_id(precursor_mz, win_range)
    1
    """
    return bisect.bisect(win_range[:, 0], precursor_mz) - 1

def find_rt_pos(RT, rt_list, n_cycles):
    """
    Find the positions in rt_list centered around the closest value to RT with the length of n_cycles.

    Parameters:
    - RT (float): The retention time to find the closest match for.
    - rt_list (list of float): The list of retention times.
    - n_cycles (int): The number of positions to include around the closest match.

    Returns:
    - list: List of indices centered around the closest match to RT.
    """

    middle_pos = np.argmin(np.abs(np.array(rt_list) - RT))
    expand_range = n_cycles // 2
    
    start_pos = middle_pos - expand_range   
    if n_cycles % 2 == 0:        
        end_pos = middle_pos + expand_range
    else:
        end_pos = middle_pos + expand_range + 1       
    
    if start_pos < 0:
        rt_pos = list(range(n_cycles))
    elif end_pos > len(rt_list):
        rt_pos = list(range(len(rt_list) - n_cycles, len(rt_list)))
    else:
        rt_pos = list(range(start_pos, end_pos))
    
    if len(rt_pos) > len(rt_list):
        return list(range(len(rt_list)))
    
    return rt_pos

def calc_XIC(spectra: List[Tuple[np.ndarray, np.ndarray]], mz_to_extract: float, mz_unit: str, mz_tol: float) -> List[float]:
    """
    Calculate the Extracted Ion Chromatogram (XIC) for a given m/z value.

    Parameters:
    - spectra (List[Tuple[np.ndarray, np.ndarray]]): List of tuples containing m/z arrays and intensity arrays.
    - mz_to_extract (float): The m/z value to extract.
    - mz_unit (str): The unit for m/z tolerance ('Da' or 'ppm').
    - mz_tol (float): The tolerance for m/z extraction.

    Returns:
    - List[float]: The XIC intensity values for each spectrum.

    Example usage:
    >>> spectra = [(np.array([100, 150, 200, 250, 300]), np.array([10, 20, 30, 40, 50]))]
    >>> calc_XIC(spectra, 200, "Da", 10)
    [30]
    """
    if mz_to_extract == -1:
        return [0.0 for _ in spectra]

    if mz_unit == "Da":
        extract_width = [mz_to_extract - mz_tol / 2, mz_to_extract + mz_tol / 2]
    elif mz_unit == "ppm":
        mz_tol_da = mz_to_extract * mz_tol * 0.000001
        extract_width = [mz_to_extract - mz_tol_da / 2, mz_to_extract + mz_tol_da / 2]
    else:
        raise ValueError("Invalid mz_unit. Use 'Da' or 'ppm'.")

    xic = []
    for mz_array, intensity_array in spectra:
        start_idx = bisect.bisect_left(mz_array, extract_width[0])
        end_idx = bisect.bisect_right(mz_array, extract_width[1])
        xic_value = sum(intensity_array[start_idx:end_idx])
        xic.append(xic_value)    

    return xic

def filter_matrix(matrix):
    """
    Filter the input matrix based on specific criteria:
    1. Keep only rows where the sum is greater than or equal to 200.
    2. For the remaining rows, keep only those where the ratio of the maximum to 
       the minimum non-zero value in the row is greater than or equal to 1.5.

    Parameters:
    - matrix (Any): Input matrix, convertible to a NumPy array.

    Returns:
    - np.ndarray: Filtered matrix.
    
    Example usage:
    >>> matrix = np.array([[100, 0, 100], [0, 0, 300], [150, 50, 0]])
    >>> filter_matrix(matrix)
    array([[150.,  50.,   0.]])
    """
    matrix = np.array(matrix, dtype=float)
    
    # Keep only rows where the sum is greater than or equal to 200
    matrix = matrix[np.sum(matrix, axis=1) >= 200]
    
    # Compute the max and min (non-zero) values for each row
    ms2_max_list = np.max(matrix, axis=1)
    
    # Replace zeros with inf for min calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix_non_zero = np.where(matrix == 0, np.inf, matrix)
        ms2_min_list = np.min(matrix_non_zero, axis=1)
    
    # Restore zeros
    matrix[matrix_non_zero == np.inf] = 0
    
    # Filter rows where the ratio of max to min is greater than or equal to 1.5
    ratio_condition = ms2_max_list / ms2_min_list >= 1.5
    matrix = matrix[ratio_condition]
    
    return matrix

def calc_pearson_sums(lib_xics):
    """
    @param lib_xics: raw XICs of `lib` fragment ions.
                     np.arary([[XIC_frag1_t1, XIC_frag1_t2, ... XIC_frag1_tn], 
                               [XIC_frag2_t1, XIC_frag2_t2, ... XIC_frag2_tn], 
                               ...
                               [XIC_fragM_t1, XIC_fragM_t2, ... XIC_fragM_tn]])

    @ output std_indice: The standard deviation (std) of each XIC must not be 0 for Pearson correlation calculation. 
                         Thus, if the std of an XIC is 0, its Pearson score will be set to 0.
                         This output indicates the indice of valid XICs with Pearson scores not equalling to 0.
                         For example, the std_indice will be np.array([0, 1, 3]) if the Pearson scores of XICs of indice 0, 1 and 3 are not 0.
    
    @output pearson_sums: Pearson score of each XIC. 
    """

    lib_xics_std = lib_xics.std(axis = 1)
    std_indice = np.where(lib_xics_std != 0)[0]
    pearson_sums = [0] * len(lib_xics_std)
    if len(std_indice) == 1:
        pearson_sums[std_indice[0]] = 1               
    elif len(std_indice) > 1:
        lib_xics_for_pearson = lib_xics[std_indice, :] 
        pearson_matrix = np.corrcoef(lib_xics_for_pearson)
        pearson_sums_part = pearson_matrix.sum(axis = 1)
        for std_index, pearson_sum in zip(std_indice, pearson_sums_part):
            pearson_sums[std_index] = pearson_sum

    return std_indice, pearson_sums

def adjust_size(frag_matrix, n_frags):
    """
    Adjust the size of the fragment matrix to have at most n_frags rows.
    If the matrix has more than n_frags rows, keep the top n_frags rows 
    with the highest sum across columns.

    Parameters:
    - frag_matrix (Any): Input fragment matrix, convertible to a NumPy array.
    - n_frags (int): Maximum number of rows to keep.

    Returns:
    - np.ndarray: The adjusted fragment matrix.
    
    Example usage:
    >>> frag_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]])
    >>> adjust_size(frag_matrix, 2)
    array([[7, 8, 9],
           [4, 5, 6]])
    """
    frag_matrix = np.array(frag_matrix, dtype=float)
    
    if frag_matrix.shape[0] > n_frags:
        frag_sum = frag_matrix.sum(axis=1)
        frag_selected = frag_sum.argsort()[::-1][:n_frags]
        return frag_matrix[frag_selected]
    
    return frag_matrix

def adjust_cycle(frag_matrix, n_cycles):
    """
    Adjust the number of cycles in the fragment matrix to match n_cycles.

    Parameters:
    - frag_matrix (numpy.ndarray): The original fragment matrix.
    - n_cycles (int): The desired number of cycles.

    Returns:
    - numpy.ndarray: The adjusted fragment matrix.
    """

    if frag_matrix.shape[1] < n_cycles:
        new_matrix = np.zeros((frag_matrix.shape[0], n_cycles))
        new_matrix[:, :frag_matrix.shape[1]] = frag_matrix
        return new_matrix
    else:
        return frag_matrix[:, :n_cycles]
    
def tukey_inliers(a_array):
    """
    Identify inliers in an array using Tukey's fences.

    Parameters:
    - a_array (numpy.ndarray): Input array to identify inliers.

    Returns:
    - numpy.ndarray: Indices of inliers in the input array.
    """

    upper_quantile = np.quantile(a_array, 0.75)
    lower_quantile = np.quantile(a_array, 0.25)
    iqr = upper_quantile - lower_quantile
    upper_bound = upper_quantile + 2.5 * iqr
    lower_bound = lower_quantile - 2.5 * iqr

    return np.array([i for i, num in enumerate(a_array) if (lower_bound <= num <= upper_bound)])

def normalize_single_trace(x: List[float]) -> np.ndarray:
    """
    Normalize a single trace using Min-Max scaling.

    Parameters:
    - x (List[float]): A list of values to be normalized.

    Returns:
    - np.ndarray: The normalized values as a NumPy array.

    Example usage:
    >>> normalize_single_trace([1, 2, 3, 4, 5])
    array([0. , 0.25, 0.5 , 0.75, 1. ])
    """
    scaler = MinMaxScaler()
    x_reshaped = np.array(x).reshape(-1, 1)
    x_normalized = scaler.fit_transform(x_reshaped)
    
    return x_normalized[:, 0]

def tear_list_given_n_each_batch(alist, n_each_batch):    
    """
    Splits the input list into smaller lists of a specified size.

    Parameters:
    alist (list): The list to be split into batches.
    n_each_batch (int): The size of each batch. If n_each_batch is greater than or 
                        equal to the length of the list or less than 1, the whole 
                        list is returned as a single batch.

    Returns:
    list: A list of smaller lists, each of size n_each_batch, except possibly the last one.
    
    Example:
    >>> alist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> n_each_batch = 3
    >>> tear_list_given_n_each_batch(alist, n_each_batch)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """

    if n_each_batch >= len(alist) or n_each_batch < 1:
        return [alist]
    batch_list = []
    cursor = 0
    while cursor < len(alist):
        batch_list.append(alist[cursor : cursor + n_each_batch])
        cursor += n_each_batch
    return batch_list