"""
╔═════════════════════════════════════════════════════╗
║               ms_file_processing.py                 ║
╠═════════════════════════════════════════════════════╣
║       Description: Utility functions for mass       ║
║            spectrometry file processing             ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║             Contact: mingxuan.gao@utoronto.ca       ║
╚═════════════════════════════════════════════════════╝
"""

import os
import sys
from typing import List

import numpy as np
from pyteomics import mzxml, mzml

from utils import calc_win_id

class MS1_Chrom:
    """
    A class to represent MS1 Chromatograms.

    Attributes:
    - rt_list (list): List of retention times.
    - spectra (list): List of spectra.
    """
    def __init__(self):
        self.rt_list = []
        self.spectra = []

class MS2_Chrom:
    """
    A class to represent MS2 Chromatograms.

    Attributes:
    - win_id (int): Window ID.
    - win_min (float): Minimum window value.
    - win_max (float): Maximum window value.
    - rt_list (list): List of retention times.
    - spectra (list): List of spectra.
    """
    def __init__(self, win_id, win_min, win_max):
        self.win_id = win_id
        self.win_min = win_min
        self.win_max = win_max
        self.rt_list = []
        self.spectra = []

def filter_spectrum(spectrum, mz_min, mz_max):
    """
    Filter the spectrum by m/z range and remove zero intensity values.

    Parameters:
    - spectrum (dict): Dictionary containing 'm/z array' and 'intensity array'.
    - mz_min (float): Minimum m/z value for filtering.
    - mz_max (float): Maximum m/z value for filtering.

    Returns:
    - tuple: Filtered m/z array and intensity array.

    Example usage:
    >>> spectrum = {
    ...     'm/z array': np.array([100, 150, 200, 250, 300]),
    ...     'intensity array': np.array([10, 0, 20, 30, 0])
    ... }
    >>> mz_min = 150
    >>> mz_max = 300
    >>> filtered_mz, filtered_intensity = filter_spectrum(spectrum, mz_min, mz_max)
    >>> print("Filtered m/z array:", filtered_mz)
    >>> print("Filtered intensity array:", filtered_intensity)
    Filtered m/z array: [200 250]
    Filtered intensity array: [20 30]
    """
    intensity_array = spectrum['intensity array']
    mz_array = spectrum['m/z array'][intensity_array > 0]
    intensity_array = intensity_array[intensity_array > 0]

    ms_range = (mz_array >= mz_min) & (mz_array < mz_max)
    mz_array = mz_array[ms_range]
    intensity_array = intensity_array[ms_range]

    return mz_array, intensity_array

def update_chrom(ms1: MS1_Chrom, ms2: List[MS2_Chrom]) -> List[MS2_Chrom]:
    """
    Update MS2 chromatograms to match the retention time list of the MS1 chromatogram.

    Parameters:
    - ms1: MS1_Chrom object containing the retention time list and spectra.
    - ms2: List of MS2_Chrom objects to be updated.

    Returns:
    - List of updated MS2_Chrom objects with spectra aligned to the MS1 retention times.

    Example usage:
    >>> ms1 = MS1_Chrom()
    >>> ms1.rt_list = [0, 1, 2, 3]
    >>> ms1.spectra = [(np.array([100, 200]), np.array([10, 20])), (np.array([110, 210]), np.array([15, 25])), (np.array([120, 220]), np.array([18, 28])), (np.array([130, 230]), np.array([20, 30]))]
    >>> ms2_1 = MS2_Chrom(1, 400, 500)
    >>> ms2_1.rt_list = [1, 3]
    >>> ms2_1.spectra = [(np.array([140, 240]), np.array([14, 24])), (np.array([160, 260]), np.array([16, 26]))]
    >>> ms2_2 = MS2_Chrom(2, 500, 600)
    >>> ms2_2.rt_list = [0, 2]
    >>> ms2_2.spectra = [(np.array([150, 250]), np.array([15, 25])), (np.array([170, 270]), np.array([17, 27]))]
    >>> ms2 = [ms2_1, ms2_2]
    >>> new_ms2 = update_chrom(ms1, ms2)
    >>> for ms2_spectra in new_ms2:
    ...     print(ms2_spectra.rt_list)
    ...     print([spec[0] for spec in ms2_spectra.spectra])
    ...     print([spec[1] for spec in ms2_spectra.spectra])
    [0, 1, 2, 3]
    [array([500, 800]), array([140, 240]), array([500, 800]), array([160, 260])]
    [array([0, 0]), array([14, 24]), array([0, 0]), array([16, 26])]
    [0, 1, 2, 3]
    [array([150, 250]), array([500, 800]), array([170, 270]), array([500, 800])]
    [array([15, 25]), array([0, 0]), array([17, 27]), array([0, 0])]
    """
    new_ms2 = []
    for ms2_spectra in ms2:
        new_spectra = MS2_Chrom(ms2_spectra.win_id, ms2_spectra.win_min, ms2_spectra.win_max)
        new_spectra.rt_list = ms1.rt_list
        for time_point in ms1.rt_list:
            if time_point in ms2_spectra.rt_list:
                new_spectra.spectra.append(ms2_spectra.spectra[ms2_spectra.rt_list.index(time_point)])
            else:
                new_spectra.spectra.append((np.array([500, 800]), np.array([0, 0])))
        new_ms2.append(new_spectra)
    return new_ms2

def load_rawdata(rawdata_file, mz_min, mz_max):
    """
    Load raw data from an mzXML or mzML file and extract MS1 and MS2 chromatograms.

    Parameters:
    - rawdata_file (str): Path to the raw data file (.mzXML or .mzML).
    - mz_min (float): Minimum m/z value for filtering the spectra.
    - mz_max (float): Maximum m/z value for filtering the spectra.

    Returns:
    - Tuple[MS1_Chrom, List[MS2_Chrom], np.ndarray]: 
        - MS1_Chrom object containing MS1 chromatogram.
        - List of MS2_Chrom objects containing MS2 chromatograms.
        - Numpy array representing the window range.

    Raises:
    - Exception: If the file format is not supported.
    """

    # Determine the reader and functions based on file type
    if rawdata_file.endswith(".mzXML"):
        rawdata_reader = mzxml.MzXML(rawdata_file)
        mslevel_string = "msLevel"
        def get_RT_from_rawdata_spectrum(spectrum):
            return spectrum["retentionTime"]
        def get_precursor_mz_from_rawdata_spectrum(spectrum):
            return spectrum['precursorMz'][0]['precursorMz']
        def get_winWidth_from_rawdata_spectrum(spectrum):
            return spectrum['precursorMz'][0]['windowWideness']
    elif rawdata_file.endswith(".mzML"):
        rawdata_reader = mzml.MzML(rawdata_file)
        mslevel_string = "ms level"
        def get_RT_from_rawdata_spectrum(spectrum):
            return spectrum["scanList"]["scan"][0]["scan start time"]
        def get_precursor_mz_from_rawdata_spectrum(spectrum):
            return spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]['selected ion m/z']
        def get_precursor_leftBound_from_rawdata_spectrum(spectrum):
            return spectrum["precursorList"]["precursor"][0]["isolationWindow"]["isolation window lower offset"]
    else:
        raise Exception(f"Invalid rawdata file: {rawdata_file}!\nOnly mzXML and mzML files are supported!")   

    def win_calculator(rawdata_reader, mslevel_string):
        raw_win = []
        flag = 0
        for spectrum in rawdata_reader:
            if spectrum[mslevel_string] == 1:
                flag += 1
            else:
                if flag == 0:
                    continue
                elif flag == 1:
                    p_mz = get_precursor_mz_from_rawdata_spectrum(spectrum)
                    if mslevel_string == "msLevel":
                        p_width = get_winWidth_from_rawdata_spectrum(spectrum)
                    else:
                        p_width = get_precursor_leftBound_from_rawdata_spectrum(spectrum) * 2
                    raw_win.append([p_mz, p_width])
                else:
                    rawdata_reader.reset()
                    break
        raw_win = list(map(lambda x : [x[0] - x[1] / 2, x[0] + x[1] / 2], raw_win))
        win_range = [raw_win[0][0]]
        for i in range(len(raw_win) - 1):
            if raw_win[i][1] > raw_win[i + 1][0]:
                overlap = raw_win[i][1] - raw_win[i + 1][0]
                win_range.append(raw_win[i][1] - overlap / 2)
                win_range.append(raw_win[i + 1][0] + overlap / 2)
            else:
                win_range.append(raw_win[i][1])
                win_range.append(raw_win[i + 1][0])
        win_range.append(raw_win[-1][-1])   
        win_range = np.array([[win_range]]).reshape(-1, 2)   
        return win_range
    
    # Calculate the window range
    win_range = win_calculator(rawdata_reader, mslevel_string)

    # Initialize MS1 and MS2 chromatograms
    ms1 = MS1_Chrom()
    ms2 = [MS2_Chrom(i, each_win[0], each_win[1]) for i, each_win in enumerate(win_range)]

    # Parse the raw data and populate chromatograms
    for idx, spectrum in enumerate(rawdata_reader):
        if spectrum[mslevel_string] == 1:
            RT = 60 * get_RT_from_rawdata_spectrum(spectrum)
            mz_array, intensity_array = filter_spectrum(spectrum, mz_min, mz_max)
            ms1.rt_list.append(RT)
            ms1.spectra.append((mz_array, intensity_array))
        elif spectrum[mslevel_string] == 2: 
            if idx == 0: 
                RT = 0
            precursor_mz = get_precursor_mz_from_rawdata_spectrum(spectrum)
            mz_array, intensity_array = filter_spectrum(spectrum, mz_min, mz_max)
            win_id = calc_win_id(precursor_mz, win_range)
            ms2[win_id].rt_list.append(RT)
            ms2[win_id].spectra.append((mz_array, intensity_array))
    
    # Update MS2 chromatograms to match MS1 retention times
    ms2 = update_chrom(ms1, ms2)
    
    return ms1, ms2, win_range

def convert_and_load_raw_data(rawdata_file, file_dir, TRFP_file, logger, mz_min, mz_max):
    """
    Convert and load raw mass spectrometry data.

    Parameters:
    - rawdata_file (str): The path to the raw data file.
    - file_dir (str): The directory where the converted file will be stored.
    - TRFP_file (str): The path to the file conversion tool ThermoRawFileParser.exe.
    - logger (logging.Logger): Logger for logging information.
    - mz_min (float): Minimum m/z value for filtering.
    - mz_max (float): Maximum m/z value for filtering.

    Returns:
    - ms1, ms2, win_range, rawdata_prefix: The loaded mass spectrometry data and window range.
    """

    if rawdata_file.endswith(".raw"):
        logger.info("Perform data format conversion...")
        rawdata_prefix = rawdata_file[:-4]
        if sys.platform == "linux":
            convert_status = os.system("mono %s -i=%s -o=%s -f=1" % (TRFP_file, os.path.join(file_dir, rawdata_file), file_dir))
        else:
            convert_status = os.system("%s -i=%s -o=%s -f=1" % (TRFP_file, os.path.join(file_dir, rawdata_file), file_dir))
        if convert_status != 0:
            logger.info("Error: File format conversion failed for %s" % rawdata_file)   
            return None, None, None, None      
        ms1, ms2, win_range = load_rawdata(os.path.join(file_dir, rawdata_prefix + ".mzML"), mz_min, mz_max)       
    else:
        if rawdata_file.endswith(".mzML"):
            rawdata_prefix = rawdata_file[:-5]
        elif rawdata_file.endswith(".mzXML"):
            rawdata_prefix = rawdata_file[:-6]            
        ms1, ms2, win_range = load_rawdata(os.path.join(file_dir, rawdata_file), mz_min, mz_max)

    return ms1, ms2, win_range, rawdata_prefix