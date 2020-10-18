import random
import os
import os.path
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

DELTA_RT_COL = "delta_rt"
DREAM_SCORE_COL = "dream_scores"
LIB_COS_SCORE_COL = "lib_cos_scores"
DRF_SCORES_COL = "drf_scores"

n_base_scores = 4
heuristic_depth_xgboost = 6
heuristic_depth_rf = 12

def get_label_from_protein(x):
    if x.startswith("DECOY"):
        return "DECOY"
    return "TARGET"

def feature_construction(dream_score_res, lib_cols, top_k):  
    # 1. top_scores
    get_top_scores = lambda x : [abs(float(i)) for i in x.strip().split(";")[:top_k]]
    get_top_square_scores = lambda x : [float(i) ** 2 for i in x.strip().split(";")[:top_k]]
    
    delta_rt_scores = np.array(list(dream_score_res[DELTA_RT_COL].apply(get_top_scores)))
    delta_rt_square_scores = np.array(list(dream_score_res[DELTA_RT_COL].apply(get_top_square_scores)))
    dream_scores = np.array(list(dream_score_res[DREAM_SCORE_COL].apply(get_top_scores)))
    lib_cos_scores = np.array(list(dream_score_res[LIB_COS_SCORE_COL].apply(get_top_scores)))
    
    # 2. mean_scores
    delta_rt_scores_mean = delta_rt_scores.mean(axis = 1)
    delta_rt_square_scores_mean = delta_rt_square_scores.mean(axis = 1)
    dream_scores_mean = dream_scores.mean(axis = 1)
    lib_cos_scores_mean = lib_cos_scores.mean(axis = 1)
    
    mean_scores_list = [delta_rt_scores_mean, delta_rt_square_scores_mean, dream_scores_mean, lib_cos_scores_mean]
    all_mean_scores = np.vstack(mean_scores_list).T
    
    # 3. std_scores
    delta_rt_scores_std = delta_rt_scores.std(axis = 1)
    delta_rt_square_scores_std = delta_rt_square_scores.std(axis = 1)
    dream_scores_std = dream_scores.std(axis = 1)
    lib_cos_scores_std = lib_cos_scores.std(axis = 1)
   
    std_scores_list = [delta_rt_scores_std, delta_rt_square_scores_std, dream_scores_std, lib_cos_scores_std] 
    all_std_scores = np.vstack(std_scores_list).T
    
    # 4. combine scores
    all_scores_list = [delta_rt_scores, delta_rt_square_scores, dream_scores, lib_cos_scores, all_mean_scores, all_std_scores]
    all_scores = np.hstack(all_scores_list)
    n_scores = all_scores.shape[1]
    
    all_scores_df = pd.DataFrame(all_scores, columns = ["score%d" % i for i in range(n_scores)])
    
    # 5. other_scores
    all_scores_df["score%d" % n_scores] = dream_score_res[lib_cols["PURE_SEQUENCE_COL"]].apply(lambda x : len(x))
    all_scores_df["score%d" % (n_scores+1)] = dream_score_res[lib_cols["PRECURSOR_MZ_COL"]]
    all_scores_df["score%d" % (n_scores+2)] = dream_score_res[lib_cols["PRECURSOR_CHARGE_COL"]]
    
    original_columns = list(dream_score_res.columns)
    score_columns = list(all_scores_df.columns)
    all_data = pd.concat([dream_score_res, all_scores_df], axis = 1, ignore_index = True)
    all_data.columns = original_columns + score_columns
    
    return all_data

def data_augmentation(all_data, top_k):
    other_columns = [i for i in list(all_data.columns) if not i.startswith("score")]

    for i in range(top_k):
        top_score_columns = ["score%d" % k for k in range(i, top_k * n_base_scores, top_k)]
        other_score_columns = ["score%d" % k for k in range(top_k * n_base_scores, top_k * n_base_scores + (2 * n_base_scores + 3))]
        wanted_columns = top_score_columns + other_score_columns
        if not i:
            augmented_data = all_data.loc[:, other_columns + wanted_columns]
            augmented_data.columns = other_columns + ["score%d" % k for k in range(len(wanted_columns))]
            augmented_data["peak_group_rank"] = [i+1 for _ in range(augmented_data.shape[0])]
        else:
            all_data_part = all_data.loc[:, other_columns + wanted_columns]
            all_data_part.columns = other_columns + ["score%d" % k for k in range(len(wanted_columns))]
            all_data_part["peak_group_rank"] = [i+1 for _ in range(all_data_part.shape[0])]
            augmented_data = pd.concat([augmented_data, all_data_part])
    return augmented_data

def calc_cut_at_same_decoys(final_data, n_decoys, label_column, score_column):
    decoy_scores = sorted(list(final_data[final_data[label_column] == "DECOY"][score_column]), reverse = True)
    if n_decoys >= len(decoy_scores):
        return -np.inf, ">"
    if decoy_scores[n_decoys] == decoy_scores[n_decoys - 1]:
        return decoy_scores[n_decoys], ">="
    return decoy_scores[n_decoys], ">"

def calc_score_cut(altsv, score_column, label_column, cut_off, logger, smooth_factor = 0.01, plot = True, plot_name = None):  
    target = altsv[altsv[label_column] == "TARGET"]
    target = target.sort_values(by = score_column, ascending = False)
    decoy = altsv[altsv[label_column] != "TARGET"]
    decoy = decoy.sort_values(by = score_column, ascending = False)
    
    def larger_count(alist, value):
        return list(map(lambda x : 1 if x >= value else 0, alist)).count(1)
    
    target_scores = list(target[score_column])
    decoy_scores = list(decoy[score_column])

    fdrs = []
    n_fp = 0
    for i, cut in enumerate(target_scores):
        n_tp = i + 1
        while n_fp < len(decoy_scores) and decoy_scores[n_fp] >= cut:
            n_fp += 1
        fdr = n_fp / (n_fp + n_tp)
        fdrs.append(fdr)
    
    cut_modified, times = 0, 0 
    final_cut = 0.7
    while cut_modified == 0 and times < 20 and int(smooth_factor * len(fdrs)) > 0:
        slider = collections.deque()
        for i in range(int(smooth_factor * len(fdrs))):
            slider.append(fdrs[i])
            
        for i in range(int(smooth_factor * len(fdrs)), len(fdrs)):
            if slider[0] >= cut_off:
                if larger_count(slider, cut_off) >= len(slider) * 0.8:
                    final_cut = target_scores[i - 1]
                    cut_modified = 1
                    break
            slider.popleft()
            slider.append(fdrs[i])
        smooth_factor /= 1.2
        times += 1
    
    if cut_modified == 0:
        logger.info("Warning: failed to calculate FDR. Used a heuristic cut-off value: %s" % final_cut)
    
    if plot and plot_name:
        plt.figure(figsize = (9.7, 7.4))
        sns.distplot(target[score_column], label = "target")
        sns.distplot(decoy[score_column], label = "decoy")
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.xlabel(score_column, fontsize = 22)
        plt.ylabel("Frequency", fontsize = 22)
        plt.legend(fontsize = 22)
        plt.axvline(x = final_cut, color = "red")
        plt.title("%.6f" % final_cut)
        plt.savefig(plot_name)
        plt.close()
    return fdrs, final_cut

def feature(dream_score_res, lib_cols, top_k, score0_cutoff, score2_cutoff):
    dream_score_res["label"] = dream_score_res[lib_cols["PROTEIN_NAME_COL"]].apply(get_label_from_protein)
    all_data = feature_construction(dream_score_res, lib_cols, top_k)
    augmented_data = data_augmentation(all_data, top_k)
    augmented_data = augmented_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "score2"], ascending = [True, False])

    target_filter1 = augmented_data["label"] != "DECOY"
    target_filter2 = augmented_data["score2"] >= score2_cutoff
    target_filter3 = augmented_data["score0"] < score0_cutoff

    augmented_data = augmented_data[(augmented_data["label"] == "DECOY") | (target_filter1 & target_filter2 & target_filter3)]
    augmented_data.index = [i for i in range(augmented_data.shape[0])]

    def get_drf_string(x, y):
        return x.strip().split(";")[y - 1]

    augmented_data["drf_string"] = augmented_data.apply(lambda df: get_drf_string(df[DRF_SCORES_COL], df["peak_group_rank"]), axis = 1)

    for i in range(10):
        augmented_data["score%d" % (15 + i)] = augmented_data["drf_string"].apply(lambda x : float(x.strip().split("|")[i]))
    return augmented_data

def prophet(augmented_data, lib_cols, max_depth, disc_model, n_threads, seed):
    score_ids = [i for i in range(3 * n_base_scores + 3 + 10) if i != 2]
    score_columns = ["score%d" % i for i in score_ids]

    if disc_model == "xgboost":
        cl = XGBClassifier(n_jobs = n_threads, max_depth = max_depth, random_state = seed)
    else:
        cl = RandomForestClassifier(n_estimators = 200, max_depth = max_depth, n_jobs = n_threads, random_state = seed)

    cl.fit(augmented_data.loc[:, score_columns].values, augmented_data["label"])
    prediction = cl.predict_proba(augmented_data.loc[:, score_columns].values)[:, 1]
    augmented_data["dr_score"] = prediction

    augmented_data = augmented_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "dr_score"], ascending = [True, False])
    augmented_data_grouped = augmented_data.drop_duplicates(subset = lib_cols["PRECURSOR_ID_COL"], keep = "first")

    return augmented_data, augmented_data_grouped

def quant(gp_rank, ms2_areas, lib_pearsons):
    selected_area_string = ms2_areas.strip().split(";")[gp_rank - 1]
    selected_area = np.array([float(i) for i in selected_area_string.strip().split("|")])
    weights = np.array([float(i) for i in lib_pearsons.strip().split(";")[gp_rank - 1].split("|")])
    
    return selected_area.dot(weights) / len(weights)

def combine_res(dream_score_res_files, lib_cols):
    dream_score_res = pd.read_csv(dream_score_res_files[0], sep = "\t")
    dream_score_res[lib_cols["PRECURSOR_ID_COL"] + "_original"] = list(dream_score_res[lib_cols["PRECURSOR_ID_COL"]])
    dream_score_res[lib_cols["PRECURSOR_ID_COL"]] = dream_score_res[lib_cols["PRECURSOR_ID_COL"]].apply(lambda x : "run0_%s" % x)
    for i, res_file in enumerate(dream_score_res_files[1:]):
        dream_score_part = pd.read_csv(res_file, sep = "\t")
        dream_score_part[lib_cols["PRECURSOR_ID_COL"] + "_original"] = list(dream_score_part[lib_cols["PRECURSOR_ID_COL"]])
        dream_score_part[lib_cols["PRECURSOR_ID_COL"]] = dream_score_part[lib_cols["PRECURSOR_ID_COL"]].apply(lambda x : "run%d_%s" % (i + 1, x))
        dream_score_res = pd.concat([dream_score_res, dream_score_part])
    dream_score_res.index = [i for i in range(dream_score_res.shape[0])]
    return dream_score_res

def dream_indicators_xgboost(augmented_data, lib_cols, n_threads, seed, output_path, logger, fdr_precursor):
    search_range = np.arange(1, 21, 1)

    downsample_cutoff = 10000000
    if augmented_data.shape[0] > downsample_cutoff:
        indi_data = augmented_data.sample(downsample_cutoff)
    else:
        indi_data = augmented_data.copy()

    spike_rate = 0.2
    decoy_indice = list(np.where(indi_data["label"] == "DECOY")[0])
    random.seed(seed)
    modifiable_indice = sorted(random.sample(decoy_indice, int(len(decoy_indice) * spike_rate)))
    indicate_labels, pointer = [], 0
    for i, lab in enumerate(list(indi_data["label"])):
        if pointer < len(modifiable_indice) and i == modifiable_indice[pointer]:
            indicate_labels.append("TARGET")
            pointer += 1
        else:
            indicate_labels.append(lab)
    indi_data["indicate_label"] = indicate_labels

    score_ids = [i for i in range(3 * n_base_scores + 3 + 10) if i != 2]
    score_columns = ["score%d" % i for i in score_ids]

    cl = XGBClassifier(n_jobs = n_threads, max_depth = heuristic_depth_xgboost, random_state = seed)
    cl.fit(indi_data.loc[:, score_columns].values, indi_data["indicate_label"])
    prediction = cl.predict_proba(indi_data.loc[:, score_columns].values)[:, 1]
    indi_data["dr_score"] = prediction

    indi_data = indi_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "dr_score"], ascending = [True, False])
    indi_data_grouped = indi_data.drop_duplicates(subset = lib_cols["PRECURSOR_ID_COL"], keep = "first")

    fdrs, final_cut = calc_score_cut(indi_data_grouped, score_column = "dr_score", label_column = "indicate_label", cut_off = fdr_precursor, 
                                     logger = logger, plot = True, plot_name = os.path.join(output_path, "default_score_distribution.pdf"))
    default_res = indi_data_grouped[indi_data_grouped["dr_score"] >= final_cut].label.value_counts()
    n_default_target = default_res["TARGET"]
    n_default_decoy = default_res["DECOY"]

    n_target_col, n_indi_target_col = [], []
    for i, test_max_depth in enumerate(search_range):
        cl = XGBClassifier(n_jobs = n_threads, max_depth = test_max_depth, random_state = seed)
        cl.fit(indi_data.loc[:, score_columns].values, indi_data["indicate_label"])
        prediction = cl.predict_proba(indi_data.loc[:, score_columns].values)[:, 1]
        indi_data["dr_score"] = prediction
        indi_data = indi_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "dr_score"], ascending = [True, False])
        indi_data_grouped = indi_data.drop_duplicates(subset = lib_cols["PRECURSOR_ID_COL"], keep = "first")
        
        cutoff, sign = calc_cut_at_same_decoys(indi_data_grouped, n_default_decoy, lib_cols["PRECURSOR_ID_COL"], "dr_score")
        if sign == ">=":
            data_part = indi_data_grouped[indi_data_grouped["dr_score"] >= cutoff]
        else:
            data_part = indi_data_grouped[indi_data_grouped["dr_score"] > cutoff]
        n_target_col.append(data_part["indicate_label"].value_counts()["TARGET"])
        target_part = data_part[data_part["indicate_label"] == "TARGET"]
        n_indi_target_col.append(target_part[target_part["label"] == "DECOY"].shape[0])
        logger.info("(%d / %d) depth: %d test finished..." % (i + 1, len(search_range), test_max_depth))
    
    gs_results = pd.DataFrame({"MAX_DEPTH" : search_range, 
                               "N_TARGETS" : n_target_col, 
                               "N_INDI_TARGETS" : n_indi_target_col})

    gs_results["INDI_RATE"] = gs_results["N_INDI_TARGETS"] / (gs_results["N_TARGETS"] + gs_results["N_INDI_TARGETS"])
    gs_results["diff"] = gs_results["INDI_RATE"].diff()

    plt.plot(gs_results["MAX_DEPTH"], gs_results["INDI_RATE"])
    plt.scatter(gs_results["MAX_DEPTH"], gs_results["INDI_RATE"])
    plt.savefig(os.path.join(output_path, "max_depth_chosen.pdf"))
    plt.close()

    optimal_depth = gs_results["MAX_DEPTH"][np.argmin(gs_results["INDI_RATE"])]
    logger.info("Optimal max depth: %d" % optimal_depth)

    return optimal_depth

def dream_indicators_rf(augmented_data, lib_cols, n_threads, seed, output_path, logger, fdr_precursor):
    search_range = np.arange(1, 21, 1)

    downsample_cutoff = 10000000
    if augmented_data.shape[0] > downsample_cutoff:
        indi_data = augmented_data.sample(downsample_cutoff)
    else:
        indi_data = augmented_data.copy()

    spike_rate = 0.2
    decoy_indice = list(np.where(indi_data["label"] == "DECOY")[0])
    random.seed(seed)
    modifiable_indice = sorted(random.sample(decoy_indice, int(len(decoy_indice) * spike_rate)))
    indicate_labels, pointer = [], 0
    for i, lab in enumerate(list(indi_data["label"])):
        if pointer < len(modifiable_indice) and i == modifiable_indice[pointer]:
            indicate_labels.append("TARGET")
            pointer += 1
        else:
            indicate_labels.append(lab)
    indi_data["indicate_label"] = indicate_labels

    score_ids = [i for i in range(3 * n_base_scores + 3 + 10) if i != 2]
    score_columns = ["score%d" % i for i in score_ids]
       
    cl = RandomForestClassifier(n_estimators = 200, max_depth = heuristic_depth_rf, n_jobs = n_threads, random_state = seed)
    cl.fit(indi_data.loc[:, score_columns].values, indi_data["indicate_label"])
    prediction = cl.predict_proba(indi_data.loc[:, score_columns].values)[:, 1]
    indi_data["dr_score"] = prediction

    indi_data = indi_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "dr_score"], ascending = [True, False])
    indi_data_grouped = indi_data.drop_duplicates(subset = lib_cols["PRECURSOR_ID_COL"], keep = "first")

    fdrs, final_cut = calc_score_cut(indi_data_grouped, score_column = "dr_score", label_column = "indicate_label", cut_off = fdr_precursor, 
                                     logger = logger, plot = True, plot_name = os.path.join(output_path, "default_score_distribution.pdf"))
    default_res = indi_data_grouped[indi_data_grouped["dr_score"] >= final_cut].label.value_counts()
    n_default_target = default_res["TARGET"]
    n_default_decoy = default_res["DECOY"]

    n_target_col, n_indi_target_col = [], []
    for i, test_max_depth in enumerate(search_range):
        cl = RandomForestClassifier(n_estimators = 200, max_depth = test_max_depth, n_jobs = n_threads, random_state = seed)
        cl.fit(indi_data.loc[:, score_columns].values, indi_data["indicate_label"])
        prediction = cl.predict_proba(indi_data.loc[:, score_columns].values)[:, 1]
        indi_data["dr_score"] = prediction
        indi_data = indi_data.sort_values(by = [lib_cols["PRECURSOR_ID_COL"], "dr_score"], ascending = [True, False])
        indi_data_grouped = indi_data.drop_duplicates(subset = lib_cols["PRECURSOR_ID_COL"], keep = "first")
        
        cutoff, sign = calc_cut_at_same_decoys(indi_data_grouped, n_default_decoy, lib_cols["PRECURSOR_ID_COL"], "dr_score")
        if sign == ">=":
            data_part = indi_data_grouped[indi_data_grouped["dr_score"] >= cutoff]
        else:
            data_part = indi_data_grouped[indi_data_grouped["dr_score"] > cutoff]
        n_target_col.append(data_part["indicate_label"].value_counts()["TARGET"])
        target_part = data_part[data_part["indicate_label"] == "TARGET"]
        n_indi_target_col.append(target_part[target_part["label"] == "DECOY"].shape[0])
        logger.info("(%d / %d) depth: %d test finished..." % (i + 1, len(search_range), test_max_depth))
    
    gs_results = pd.DataFrame({"MAX_DEPTH" : search_range, 
                               "N_TARGETS" : n_target_col, 
                               "N_INDI_TARGETS" : n_indi_target_col})

    gs_results["INDI_RATE"] = gs_results["N_INDI_TARGETS"] / (gs_results["N_TARGETS"] + gs_results["N_INDI_TARGETS"])
    gs_results["diff"] = gs_results["INDI_RATE"].diff()

    plt.plot(gs_results["MAX_DEPTH"], gs_results["INDI_RATE"])
    plt.scatter(gs_results["MAX_DEPTH"], gs_results["INDI_RATE"])
    plt.savefig(os.path.join(output_path, "max_depth_chosen.pdf"))
    plt.close()

    optimal_depth = gs_results["MAX_DEPTH"][np.argmin(gs_results["INDI_RATE"])]
    logger.info("Optimal max depth: %d" % optimal_depth)

    return optimal_depth

def dream_prophet(dream_score_res, lib_cols, disc_model, top_k, n_threads, seed, dream_indicators, disc_dir, logger, fdr_precursor, fdr_protein):
    if not os.path.exists(disc_dir):
        os.mkdir(disc_dir)
    augmented_data = feature(dream_score_res, lib_cols, top_k, score0_cutoff = 500, score2_cutoff = 0.15)
    if dream_indicators:
        if disc_model == "xgboost":
            optimal_depth = dream_indicators_xgboost(augmented_data, lib_cols, n_threads, seed, disc_dir, logger, fdr_precursor)
        else:
            optimal_depth = dream_indicators_rf(augmented_data, lib_cols, n_threads, seed, disc_dir, logger, fdr_precursor)
    else:
        if disc_model == "xgboost":
            optimal_depth = heuristic_depth_xgboost
        else:
            optimal_depth = heuristic_depth_rf
    
    augmented_data, augmented_data_grouped = prophet(augmented_data, lib_cols, optimal_depth, disc_model, n_threads, seed)
    augmented_data_grouped["Intensity"] = augmented_data_grouped.apply(lambda df: quant(df["peak_group_rank"], 
                                                                                        df["ms2_areas"], 
                                                                                        df["lib_pearsons"]), axis = 1)
    augmented_data_grouped.to_csv(os.path.join(disc_dir, "Dream-DIA_all_results.tsv"), sep = "\t", index = False)
    fdrs, final_cut = calc_score_cut(augmented_data_grouped, score_column = "dr_score", label_column = "label", cut_off = fdr_precursor, 
                                     logger = logger, plot = True, plot_name = os.path.join(disc_dir, "optimal_score_distribution.pdf"))
    fdr_data = augmented_data_grouped[augmented_data_grouped["dr_score"] > final_cut]
    fdr_data.to_csv(os.path.join(disc_dir, "Dream-DIA_fdr_results.tsv"), sep = "\t", index = False)

    passed_filenames, passed_proteins, passed_fdrs, passed_n_targets, passed_n_decoys, passed_intensities = [], [], [], [], [], []
    for filename in np.unique(fdr_data["filename"]):
        fdr_data_part = fdr_data[fdr_data["filename"] == filename]
        passed_proteins_this_part = []
        stats_info_targets = fdr_data_part[fdr_data_part["label"] == "TARGET"][lib_cols["PROTEIN_NAME_COL"]].value_counts()
        stats_info_decoys = fdr_data_part[fdr_data_part["label"] == "DECOY"][lib_cols["PROTEIN_NAME_COL"]].value_counts()
        for target in list(stats_info_targets.index):
            if stats_info_targets[target] >= 3:
                if "DECOY_" + target in stats_info_decoys:
                    real_protein_fdr = stats_info_targets[target] / (stats_info_targets[target] + stats_info_decoys["DECOY_" + target])
                    if real_protein_fdr < fdr_protein:
                        passed_proteins.append(target)
                        passed_proteins_this_part.append(target)
                        passed_fdrs.append(real_protein_fdr)
                        passed_n_targets.append(stats_info_targets[target])
                        passed_n_decoys.append(stats_info_decoys["DECOY_" + target])
                else:
                    passed_proteins.append(target)
                    passed_proteins_this_part.append(target)
                    passed_fdrs.append(0)
                    passed_n_targets.append(stats_info_targets[target])
                    passed_n_decoys.append(0)
        passed_filenames.extend([filename] * len(passed_proteins))
        for target in passed_proteins_this_part:
            passed_intensities.append(sum(sorted(list(fdr_data_part[fdr_data_part[lib_cols["PROTEIN_NAME_COL"]] == target]["Intensity"]))[:3]))  
    protein_data = pd.DataFrame({"filename" : passed_filenames, "ProteinName" : passed_proteins, 
                                 "q_value" : passed_fdrs, "n_targets" : passed_n_targets, "n_decoys" : passed_n_decoys, 
                                 "Intensity" : passed_intensities})
    protein_data.to_csv(os.path.join(disc_dir, "Dream-DIA_protein_fdr_results.tsv"), sep = "\t", index = False)