"""
FCPA object decoding within load conditions (experiment 2).

For each ROI pair, the time courses of the four trials of the same object category from two runs of the same load condition
are concatenated into one FC sample. Object categories are then decoded within each load condition.
"""

import numpy as np
from scipy.stats import ttest_rel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import argparse
from itertools import combinations


def get_obj_order_mat(fmri_data_base_path, sub_base_folder_name_shapeload):
    """
    Read the object category of each trial from each run's .par file.

    fmri_data_base_path: folder containing the subject folders
    sub_base_folder_name_shapeload: subject folder names, in the same order as subj_1, ..., subj_12

    Returns:
    obj_category_idx_mat: object category IDs (1-8), in shape (n_subs, n_runs, n_trials)
    """

    n_runs = 16
    n_trials = 16

    obj_category_idx_mat = np.zeros([len(sub_base_folder_name_shapeload), n_runs, n_trials])

    for sub_count, cur_sub in enumerate(sub_base_folder_name_shapeload):

        run_num_file = f'{fmri_data_base_path}/{cur_sub}/bold/shapeload_all_runs.txt'

        # use the first 16 run folder names, sorted by name
        with open(run_num_file) as in_file:
            run_num_list = [line.strip() for line in in_file]
        run_num_list = sorted(run_num_list[:n_runs])
        if len(run_num_list) != n_runs or len(set(run_num_list)) != n_runs:
            raise ValueError('The first 16 run-list entries must contain 16 distinct runs.')

        for run_count, run_id in enumerate(run_num_list):

            run_id_param_file = f'{fmri_data_base_path}/{cur_sub}/bold/{run_id}/shapeload.par'

            with open(run_id_param_file) as in_file:
                obj_class_list = [int(line.strip().split('\t')[1]) % 8 for line in in_file] # categorize into the 8 object category classes (0-7)

            obj_class_list = [obj_id if obj_id != 0 else 8 for obj_id in obj_class_list] # replace 0 by 8

            obj_class_list = obj_class_list[1:] # remove the first (practice) trial
            if len(obj_class_list) != n_trials:
                raise ValueError('Each run must contain 16 trials after the practice trial.')

            obj_category_idx_mat[sub_count, run_count, :] = obj_class_list

    return obj_category_idx_mat


def extract_reorganize_ts_data(cur_roi_data, sub_idx, object_category_idx_mat):
    """
    Extract the time course of each task block and reorganize it by load condition and object category.

    cur_roi_data: raw time series, in shape (n_voxels, n_run_TRs, n_runs)
    sub_idx: the subject index, from 0 to n_subs-1; its parity selects the task schedule
    object_category_idx_mat: object category IDs (1-8) of all subjects, in shape (n_subs, n_runs, n_trials)

    Returns:
    task_ts: in shape (n_voxels, n_task_conditions, n_obj_categories, n_trials, n_task_block_TRs); trials are ordered by run quarter, then run, then occurrence
    """

    n_task_conds = 2
    n_obj_cats = 8
    n_trials = 16

    task_block_TR = 10
    fixation_block_TR = 10

    hemo_lag_TR = 7 # hemodynamic lag set to 7 TRs (5.6s)

    if sub_idx % 2 == 0:
        this_tasklist = [1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1] # task list for each run for odd (sub_idx+1) number subjects; 1 and 2 stand for shape 1-back and shape 2-back
    else:
        this_tasklist = [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2] # task list for each run for even (sub_idx+1) number subjects

    if cur_roi_data.ndim != 3 or cur_roi_data.shape[2] != 16 or cur_roi_data.shape[1] < 317:
        raise ValueError('Expected time series with shape (n_voxels, at least 317 TRs, 16 runs).')
    obj_order = np.asarray(object_category_idx_mat[sub_idx])
    if obj_order.shape != (16, 16) or not np.isin(obj_order, np.arange(1, 9)).all():
        raise ValueError('Object order must have shape (16, 16) per subject, with category IDs 1-8.')
    if not (np.sort(obj_order, axis=-1) == np.repeat(np.arange(1, 9), 2)).all():
        raise ValueError('Each run must contain exactly two occurrences of each object category.')

    n_voxels, n_total_run_TRs, n_runs = cur_roi_data.shape

    task_ts = np.zeros((n_voxels, n_task_conds, n_obj_cats, n_trials, task_block_TR))

    start_TR = hemo_lag_TR
    step_TR = task_block_TR + fixation_block_TR
    end_TR = start_TR + n_trials * step_TR # extract exactly 16 task blocks

    # extract each run's task block time courses
    for i_run in range(n_runs):

        task_idx = this_tasklist[i_run] - 1 # 0 or 1
        run_quarter = i_run // 4 # which of the four quarters is this run in
        run_idx_cur_task = run_quarter * 2 + i_run % 2 # run index of this run for the current task condition (0-7)

        obj_category_mat_cur_sub_run = object_category_idx_mat[sub_idx, i_run, :]

        obj_occurrence_count = np.zeros(n_obj_cats, dtype=int)
        for trial_count, cur_TR in enumerate(range(start_TR, end_TR, step_TR)):
            obj_cat_cur = int(obj_category_mat_cur_sub_run[trial_count]) - 1 # 0-7

            # the first or second occurrence of this object category within the run
            cur_combination_idx = 2 * run_idx_cur_task + obj_occurrence_count[obj_cat_cur]
            obj_occurrence_count[obj_cat_cur] += 1

            task_ts[:, task_idx, obj_cat_cur, cur_combination_idx, :] = cur_roi_data[:, cur_TR:cur_TR + task_block_TR, i_run]

    return task_ts


def reorganize_pattern_data(cur_roi_data, sub_idx):
    """
    Reorganize one subject's beta patterns by load condition and object category.

    cur_roi_data: raw beta patterns, in shape (n_voxels, 16 beta conditions, 16 runs); one beta per object exemplar (8 categories * 2 exemplars per run)
    sub_idx: the subject index, from 0 to n_subs-1; its parity selects the task schedule

    Returns:
    sorted_data_reshaped: the reorganized pattern data, in shape (n_voxels, n_task_conditions*n_obj_categories, n_trials),
        load-major then object category; trials are ordered by run quarter, then run, then exemplar
    """

    n_task_conds = 2
    n_obj_cats = 8

    if sub_idx % 2 == 0:
        this_tasklist = [1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1] # task list for each run for odd (sub_idx+1) number subjects
    else:
        this_tasklist = [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2] # task list for each run for even (sub_idx+1) number subjects

    if cur_roi_data.ndim != 3 or cur_roi_data.shape[1:] != (16, 16):
        raise ValueError('Expected pattern data with shape (n_voxels, 16, 16).')

    n_voxels, n_beta_conditions, n_trials = cur_roi_data.shape

    sorted_data = np.zeros((n_voxels, n_task_conds, n_obj_cats, n_trials))

    for i in range(n_trials):
        task_idx = this_tasklist[i] - 1 # 0 or 1
        run_quarter = i // 4 # which of the four quarters is this run in
        run_idx_in_quarter = i % 2 # first or second run for the current task condition within this quarter
        run_idx_cur_task = run_quarter * 2 + run_idx_in_quarter # run index of this run for the current task condition (0-7)

        for j in range(n_beta_conditions):
            obj_cat_idx = j % 8 # 0, 1, 2, ..., or 7
            thisrunitem = 2 * run_idx_cur_task + (j // 8) # j//8 identifies the first or second exemplar within the run

            sorted_data[:, task_idx, obj_cat_idx, thisrunitem] = cur_roi_data[:, j, i]

    sorted_data_reshaped = sorted_data.reshape(n_voxels, n_task_conds*n_obj_cats, n_trials)

    return sorted_data_reshaped


def zscore_2d(data, axis=0):
    """
    Z-score an array along a specified axis; raise an error for nonfinite or constant values
    data: in shape for instance, (n_voxels, n_trials)
    """

    data_std = data.std(axis=axis, keepdims=True)
    if (data_std == 0).any() or not np.isfinite(data).all():
        raise ValueError('Cannot z-score nonfinite data or constant values along the selected axis.')
    data_zscored = (data - data.mean(axis=axis, keepdims=True)) / data_std

    return data_zscored


def extract_pattern_helper(pattern_data, obj1_idx, obj2_idx, task_idx):
    """
    Extract and reshape the normalized pattern data of two object categories.

    pattern_data: normalized pattern data, in shape (n_voxels, n_obj_categories, n_task_conditions, 4 run quarters, 4 trials)
    obj1_idx, obj2_idx: int index for different object categories (0-7)
    task_idx: list of int index for different task conditions e.g., [0] for low load

    return: pattern_cond1, pattern_cond2: each in shape (n_voxels, 4 run quarters * n_corresponding_task_conditions * 4 trials)
    """

    # extract the corresponding object category and task condition pattern data, in shape (n_voxels, n_corresponding_task_conditions, 4 run quarters, 4 trials)
    pattern_cond1 = pattern_data[:, obj1_idx, task_idx, :, :]
    pattern_cond2 = pattern_data[:, obj2_idx, task_idx, :, :]

    # transpose to (n_voxels, 4 run quarters, n_corresponding_task_conditions, 4 trials)
    pattern_cond1 = pattern_cond1.transpose(0, 2, 1, 3)
    pattern_cond2 = pattern_cond2.transpose(0, 2, 1, 3)

    # flatten to (n_voxels, 4 run quarters * n_corresponding_task_conditions * 4 trials)
    # this way, the trials follow the run order of the same object category
    pattern_cond1 = pattern_cond1.reshape(pattern_cond1.shape[0], -1)
    pattern_cond2 = pattern_cond2.reshape(pattern_cond2.shape[0], -1)

    return pattern_cond1, pattern_cond2


def extract_ts_helper(ts_data, obj1_idx, obj2_idx, task_idx):
    """
    Extract and reshape the time series of two object categories in the corresponding task conditions.

    ts_data: time series data, in shape (n_voxels, n_obj_categories, n_task_conditions, 4 run quarters, 4 trials * n_task_block_TRs)
    obj1_idx, obj2_idx: int index for different object categories (0-7)
    task_idx: list of int index for different task conditions e.g., [0] for low load

    return: ts_cond1, ts_cond2: each in shape (n_voxels, 4 trials * n_task_block_TRs, 4 run quarters * n_corresponding_task_conditions)
    """

    # extract the corresponding object category and task condition time series data, in shape (n_voxels, n_corresponding_task_conditions, 4 run quarters, 4 trials * n_task_block_TRs)
    ts_cond1 = ts_data[:, obj1_idx, task_idx, :]
    ts_cond2 = ts_data[:, obj2_idx, task_idx, :]

    # transpose to (n_voxels, 4 run quarters, n_corresponding_task_conditions, 4 trials * n_task_block_TRs)
    ts_cond1 = ts_cond1.transpose(0, 2, 1, 3)
    ts_cond2 = ts_cond2.transpose(0, 2, 1, 3)

    # flatten to (n_voxels, 4 run quarters * n_corresponding_task_conditions, 4 trials * n_task_block_TRs)
    ts_cond1 = ts_cond1.reshape(ts_cond1.shape[0], -1, ts_cond1.shape[-1])
    ts_cond2 = ts_cond2.reshape(ts_cond2.shape[0], -1, ts_cond2.shape[-1])

    # transpose to (n_voxels, n_TRs, n_samples)
    ts_cond1 = ts_cond1.transpose(0, 2, 1)
    ts_cond2 = ts_cond2.transpose(0, 2, 1)

    return ts_cond1, ts_cond2


def calculate_fc_mat_3d(ts1, ts2, norm_pattern=True):
    """
    Calculate the voxel-wise FC (Pearson correlation) between two ROIs for each sample.

    ts1, ts2: shape (n_voxels1, n_TRs, n_samples) & (n_voxels2, n_TRs, n_samples)
    norm_pattern: z-score across FC features within each sample (default to True)

    Returns:
    fc_mat: shape (n_samples, n_voxels1 * n_voxels2)
    """

    if ts1.ndim != 3 or ts2.ndim != 3 or ts1.shape[1:] != ts2.shape[1:]:
        raise ValueError('Time series must have matching TR and sample dimensions.')
    if min(ts1.shape) == 0 or min(ts2.shape) == 0:
        raise ValueError('Time series dimensions must be nonempty.')
    n_TRs = ts1.shape[1]

    # z-score across time for each voxel and sample
    ts1_z = zscore_2d(ts1, axis=1)
    ts2_z = zscore_2d(ts2, axis=1)

    # correlation between every voxel pair for each sample, in shape (n_voxels1, n_voxels2, n_samples)
    fc_mat = np.einsum('xtk,ytk->xyk', ts1_z, ts2_z) / n_TRs

    # reorder the axis to (n_samples, n_voxels1, n_voxels2)
    fc_mat = np.transpose(fc_mat, (2, 0, 1))

    # flatten the last two dimensions to (n_samples, n_voxels1 * n_voxels2)
    fc_mat = fc_mat.reshape(fc_mat.shape[0], -1)

    if norm_pattern:
        # z-score across FC features for each sample
        fc_mat = zscore_2d(fc_mat, axis=1)

    return fc_mat


def within_pred_fc_mvpa(ts_cond1_roi1, ts_cond2_roi1, ts_cond1_roi2, ts_cond2_roi2,
                        pattern_cond1_roi1, pattern_cond2_roi1, pattern_cond1_roi2, pattern_cond2_roi2,
                        top_k_voxels=None, n_folds=4, normalized=True):
    """
    FCPA within-decoding using n-fold cross-validation. Voxels in each ROI are selected using the training voxel patterns,
    and FC features are computed from the selected voxels' time series.

    ts_*: time series of each class and ROI, in shape (n_voxels, n_TRs, n_samples)
    pattern_*: voxel patterns of each class and ROI, in shape (n_voxels, 4 * n_samples); each FC sample corresponds to four consecutive pattern samples
    top_k_voxels: number of voxels kept in each ROI, can be an integer or None (default to None)
    n_folds: number of cross-validation folds (default to 4)
    normalized: z-score across FC features within each sample (default to True)

    returns:
        acc_mat: accuracy for each fold of prediction, in shape (n_folds, )
    """

    n_trials_ts, n_trials_pattern = ts_cond1_roi1.shape[2], pattern_cond1_roi1.shape[1]
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2 or n_trials_ts < n_folds or n_trials_ts % n_folds or n_trials_pattern % n_folds:
        raise ValueError('n_folds must be at least 2 and divide both sample counts evenly.')
    if n_trials_pattern != 4 * n_trials_ts:
        raise ValueError('Each FC sample must correspond to four voxel-pattern samples.')
    if top_k_voxels is not None and (not isinstance(top_k_voxels, (int, np.integer)) or top_k_voxels < 1):
        raise ValueError('top_k_voxels must be a positive integer or None.')
    for ts1, ts2, pat1, pat2 in (
        (ts_cond1_roi1, ts_cond2_roi1, pattern_cond1_roi1, pattern_cond2_roi1),
        (ts_cond1_roi2, ts_cond2_roi2, pattern_cond1_roi2, pattern_cond2_roi2)
    ):
        if ts1.shape != ts2.shape or pat1.shape != pat2.shape or ts1.shape[0] != pat1.shape[0]:
            raise ValueError('Voxel and class dimensions must match within each ROI.')
        if ts1.shape[2] != n_trials_ts or pat1.shape[1] != n_trials_pattern:
            raise ValueError('Sample counts must match across ROIs.')
        if not all(np.isfinite(data).all() for data in (ts1, ts2, pat1, pat2)):
            raise ValueError('Decoding inputs must contain only finite values.')

    n_trials_per_fold_ts = n_trials_ts // n_folds
    n_trials_per_fold_pattern = n_trials_pattern // n_folds

    n_voxels_roi1, n_voxels_roi2 = pattern_cond1_roi1.shape[0], pattern_cond1_roi2.shape[0]

    acc_mat = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # split training and testing set, for both voxel patterns and FC samples
        test_idx_pattern = list(range(i_fold * n_trials_per_fold_pattern, (i_fold + 1) * n_trials_per_fold_pattern))
        train_idx_pattern = list(range(0, i_fold * n_trials_per_fold_pattern)) + list(range((i_fold + 1) * n_trials_per_fold_pattern, n_trials_pattern))

        test_idx_ts = list(range(i_fold * n_trials_per_fold_ts, (i_fold + 1) * n_trials_per_fold_ts))
        train_idx_ts = list(range(0, i_fold * n_trials_per_fold_ts)) + list(range((i_fold + 1) * n_trials_per_fold_ts, n_trials_ts))

        # do voxel selection based on training data only
        if top_k_voxels is not None:
            # get training pattern data for voxel selection
            pattern_cond1_roi1_tr, pattern_cond2_roi1_tr = pattern_cond1_roi1[:, train_idx_pattern], pattern_cond2_roi1[:, train_idx_pattern]
            pattern_cond1_roi2_tr, pattern_cond2_roi2_tr = pattern_cond1_roi2[:, train_idx_pattern], pattern_cond2_roi2[:, train_idx_pattern]

            _, p_vals_roi1 = ttest_rel(pattern_cond1_roi1_tr, pattern_cond2_roi1_tr, axis=1)
            _, p_vals_roi2 = ttest_rel(pattern_cond1_roi2_tr, pattern_cond2_roi2_tr, axis=1)

            if n_voxels_roi1 > top_k_voxels:
                selected_voxels_roi1 = p_vals_roi1.argsort()[:top_k_voxels]
                ts_cond1_roi1_top_k, ts_cond2_roi1_top_k = ts_cond1_roi1[selected_voxels_roi1], ts_cond2_roi1[selected_voxels_roi1]
            else:
                ts_cond1_roi1_top_k, ts_cond2_roi1_top_k = ts_cond1_roi1, ts_cond2_roi1

            if n_voxels_roi2 > top_k_voxels:
                selected_voxels_roi2 = p_vals_roi2.argsort()[:top_k_voxels]
                ts_cond1_roi2_top_k, ts_cond2_roi2_top_k = ts_cond1_roi2[selected_voxels_roi2], ts_cond2_roi2[selected_voxels_roi2]
            else:
                ts_cond1_roi2_top_k, ts_cond2_roi2_top_k = ts_cond1_roi2, ts_cond2_roi2

        else:
            ts_cond1_roi1_top_k, ts_cond2_roi1_top_k = ts_cond1_roi1, ts_cond2_roi1
            ts_cond1_roi2_top_k, ts_cond2_roi2_top_k = ts_cond1_roi2, ts_cond2_roi2

        # calculate fc matrices for each condition, each in shape (n_samples, n_voxels_roi1 * n_voxels_roi2)
        fc_cond1 = calculate_fc_mat_3d(ts_cond1_roi1_top_k, ts_cond1_roi2_top_k, norm_pattern=normalized)
        fc_cond2 = calculate_fc_mat_3d(ts_cond2_roi1_top_k, ts_cond2_roi2_top_k, norm_pattern=normalized)

        # split training and testing set
        cond1_tr = fc_cond1[train_idx_ts, :]
        cond1_te = fc_cond1[test_idx_ts, :]
        cond2_tr = fc_cond2[train_idx_ts, :]
        cond2_te = fc_cond2[test_idx_ts, :]

        # prepare for SVM training and testing data, each in shape (n_samples, n_features)
        train_data_roi = np.concatenate((cond1_tr, cond2_tr), axis=0)
        train_labels_roi = np.concatenate((np.zeros(cond1_tr.shape[0]), np.ones(cond2_tr.shape[0])), axis=0)

        test_data_roi = np.concatenate((cond1_te, cond2_te), axis=0)
        test_labels_roi = np.concatenate((np.zeros(cond1_te.shape[0]), np.ones(cond2_te.shape[0])), axis=0)

        if not np.isfinite(train_data_roi).all():
            raise ValueError(f'Nonfinite training FC features, fold {i_fold}')
        if not np.isfinite(test_data_roi).all():
            raise ValueError(f'Nonfinite test FC features, fold {i_fold}')

        # SVM training
        svm_cls_roi = SVC(kernel='linear')
        svm_cls_roi.fit(train_data_roi, train_labels_roi)

        # SVM testing
        pred_labels_roi = svm_cls_roi.predict(test_data_roi)
        acc_mat[i_fold] = accuracy_score(test_labels_roi, pred_labels_roi)

    return acc_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCPA object decoding within load conditions (experiment 2).')
    parser.add_argument('network_pairs_path', help='Text file with one comma-separated ROI pair per line')
    parser.add_argument('line_num', type=int, help='Zero-based ROI-pair line number')
    parser.add_argument('--ts-folder', default='time_series', help='Folder containing the input time-series .npz files')
    parser.add_argument('--pattern-folder', default='patterns', help='Folder containing the input pattern .npz files')
    parser.add_argument('--fmri-data-folder', default='fmri_data', help='Folder containing the subject folders with run lists and .par files')
    parser.add_argument('--subject-folders', required=True, help='Text file listing 12 subject folder names in subj_1 to subj_12 order')
    parser.add_argument('--results-folder', default='results/fcpa', help='Folder for decoding results')
    args = parser.parse_args()

    n_subs = 12
    n_folds = 4

    n_task_conds = 2
    n_obj_cats = 8
    n_trials = 16
    task_block_TR = 10

    top_k_voxels = 500 # set to None to use all voxels

    # read the roi pairs
    with open(args.network_pairs_path, 'r') as f:
        network_pairs = [line.strip() for line in f]

    if not 0 <= args.line_num < len(network_pairs):
        parser.error('line_num must identify an existing ROI-pair line (starting from 0).')
    cur_network_pair = network_pairs[args.line_num]

    if len(cur_network_pair.split(',')) != 2 or not all(name.strip() for name in cur_network_pair.split(',')):
        parser.error('Each selected line must contain two nonempty ROI names separated by a comma.')
    network1_name, network2_name = [name.strip() for name in cur_network_pair.split(',')] # e.g., 17Networks_LH_VisCent_Striate_1,17Networks_RH_VisCent_Striate_1

    print('roi pairs: ', network1_name, network2_name)

    # load the time series for FC and the voxel patterns for voxel selection
    data_network1 = np.load(f'{args.ts_folder}/{network1_name}_allsubs.npz')
    data_network2 = np.load(f'{args.ts_folder}/{network2_name}_allsubs.npz')

    pattern_file_data_network1 = np.load(f'{args.pattern_folder}/{network1_name}_allsubs.npz')
    pattern_file_data_network2 = np.load(f'{args.pattern_folder}/{network2_name}_allsubs.npz')

    # get the object order matrix for each subject, in shape (n_subs, n_runs, n_trials)
    with open(args.subject_folders) as f:
        sub_base_folder_name_shapeload = [line.strip() for line in f if line.strip()]
    if len(sub_base_folder_name_shapeload) != n_subs or len(set(sub_base_folder_name_shapeload)) != n_subs:
        parser.error('subject-folders must list 12 distinct folder names in subject order.')
    obj_cat_order_mat = get_obj_order_mat(args.fmri_data_folder, sub_base_folder_name_shapeload)

    decode_obj_cat_conditions = list(combinations(range(n_obj_cats), 2)) # all pairwise combinations of object categories

    low_load_task_idx = [0]
    high_load_task_idx = [1]

    accuracy_obj_decode_load_subcategory = np.zeros((n_subs, 2, len(decode_obj_cat_conditions), n_folds)) # 2 for low and high load conditions

    os.makedirs(args.results_folder, exist_ok=True)

    for i_sub in range(n_subs):

        print(f'Subject {i_sub+1}...')

        # load the time series for the current subject, each in shape (n_voxels, n_TRs, n_runs)
        ts_cur_sub_network1 = data_network1[f'subj_{i_sub+1}']
        ts_cur_sub_network2 = data_network2[f'subj_{i_sub+1}']

        # extract task time courses and reorganize the data, in shape (n_voxels, n_task_conditions, n_obj_categories, n_trials, n_task_block_TRs)
        task_ts_reorganized_network1 = extract_reorganize_ts_data(ts_cur_sub_network1, i_sub, obj_cat_order_mat)
        task_ts_reorganized_network2 = extract_reorganize_ts_data(ts_cur_sub_network2, i_sub, obj_cat_order_mat)

        # load the voxel patterns for each roi, in shape (n_voxels, n_beta_conditions, n_runs)
        pattern_data_cur_sub_network1 = pattern_file_data_network1[f'subj_{i_sub+1}']
        pattern_data_cur_sub_network2 = pattern_file_data_network2[f'subj_{i_sub+1}']

        # reorganize the pattern data to (n_voxels, n_task_conditions*n_obj_categories, n_trials)
        pattern_data_cur_sub_network1, pattern_data_cur_sub_network2 = reorganize_pattern_data(pattern_data_cur_sub_network1, i_sub), reorganize_pattern_data(pattern_data_cur_sub_network2, i_sub)

        # remove the voxels with nonfinite values in ts/pattern as well as with constant task block time courses
        valid_voxels_network1 = (
            np.isfinite(task_ts_reorganized_network1).all(axis=(1, 2, 3, 4))
            & (task_ts_reorganized_network1.std(axis=-1) > 0).all(axis=(1, 2, 3))
            & np.isfinite(pattern_data_cur_sub_network1).all(axis=(1, 2))
        )

        valid_voxels_network2 = (
            np.isfinite(task_ts_reorganized_network2).all(axis=(1, 2, 3, 4))
            & (task_ts_reorganized_network2.std(axis=-1) > 0).all(axis=(1, 2, 3))
            & np.isfinite(pattern_data_cur_sub_network2).all(axis=(1, 2))
        )

        print(f'Removing {(~valid_voxels_network1).sum()} / {valid_voxels_network1.size} voxels from {network1_name}, subject {i_sub+1}')
        print(f'Removing {(~valid_voxels_network2).sum()} / {valid_voxels_network2.size} voxels from {network2_name}, subject {i_sub+1}')

        # apply the valid voxel mask to the data
        task_ts_reorganized_network1 = task_ts_reorganized_network1[valid_voxels_network1]
        task_ts_reorganized_network2 = task_ts_reorganized_network2[valid_voxels_network2]
        pattern_data_cur_sub_network1 = pattern_data_cur_sub_network1[valid_voxels_network1]
        pattern_data_cur_sub_network2 = pattern_data_cur_sub_network2[valid_voxels_network2]

        n_voxels_network1, n_voxels_network2 = task_ts_reorganized_network1.shape[0], task_ts_reorganized_network2.shape[0]

        if not n_voxels_network1 > 1:
            raise ValueError(f'Too few valid voxels in {network1_name}')
        if not n_voxels_network2 > 1:
            raise ValueError(f'Too few valid voxels in {network2_name}')

        # z-score each task block across time
        task_ts_reorganized_network1 = zscore_2d(task_ts_reorganized_network1, axis=-1)
        task_ts_reorganized_network2 = zscore_2d(task_ts_reorganized_network2, axis=-1)

        if not (pattern_data_cur_sub_network1.std(axis=0) > 0).all():
            raise ValueError(f'Constant pattern across voxels in {network1_name}')
        if not (pattern_data_cur_sub_network2.std(axis=0) > 0).all():
            raise ValueError(f'Constant pattern across voxels in {network2_name}')

        # normalize the pattern data across voxels for each trial
        pattern_data_cur_sub_network1 = zscore_2d(pattern_data_cur_sub_network1, axis=0)
        pattern_data_cur_sub_network2 = zscore_2d(pattern_data_cur_sub_network2, axis=0)

        # first reshape the time series to (n_voxels, n_obj_categories, n_task_conditions, n_trials, n_task_block_TRs)
        task_ts_reshaped_network1 = task_ts_reorganized_network1.transpose(0, 2, 1, 3, 4)
        task_ts_reshaped_network2 = task_ts_reorganized_network2.transpose(0, 2, 1, 3, 4)

        # next, concat each 4 trials' time courses (two runs of the same load), in shape (n_voxels, n_obj_categories, n_task_conditions, 4 run quarters, 4 trials * n_task_block_TRs)
        task_ts_reshaped_network1 = task_ts_reshaped_network1.reshape(n_voxels_network1, n_obj_cats, n_task_conds, -1, 4*task_block_TR)
        task_ts_reshaped_network2 = task_ts_reshaped_network2.reshape(n_voxels_network2, n_obj_cats, n_task_conds, -1, 4*task_block_TR)

        # reshape pattern data to shape (n_voxels, n_obj_categories, n_task_conditions, n_trials)
        pattern_data_cur_sub_network1_reshaped = pattern_data_cur_sub_network1.reshape(n_voxels_network1, n_task_conds, n_obj_cats, n_trials).transpose(0, 2, 1, 3)
        pattern_data_cur_sub_network2_reshaped = pattern_data_cur_sub_network2.reshape(n_voxels_network2, n_task_conds, n_obj_cats, n_trials).transpose(0, 2, 1, 3)

        # reshape pattern data to shape (n_voxels, n_obj_categories, n_task_conditions, 4 run quarters, 4 trials)
        pattern_data_cur_sub_network1_reshaped = pattern_data_cur_sub_network1_reshaped.reshape(n_voxels_network1, n_obj_cats, n_task_conds, -1, 4)
        pattern_data_cur_sub_network2_reshaped = pattern_data_cur_sub_network2_reshaped.reshape(n_voxels_network2, n_obj_cats, n_task_conds, -1, 4)

        # do object decoding in each load condition
        for i_cond, (obj_cond1, obj_cond2) in enumerate(decode_obj_cat_conditions):
            print(f'Object pair {i_cond}')

            # extract the time series for the current object pair, each in shape (n_voxels, n_TRs, n_samples), n_samples equals 4 here
            data_obj1_network1_1back, data_obj2_network1_1back = extract_ts_helper(task_ts_reshaped_network1, obj_cond1, obj_cond2, low_load_task_idx)
            data_obj1_network2_1back, data_obj2_network2_1back = extract_ts_helper(task_ts_reshaped_network2, obj_cond1, obj_cond2, low_load_task_idx)

            data_obj1_network1_2back, data_obj2_network1_2back = extract_ts_helper(task_ts_reshaped_network1, obj_cond1, obj_cond2, high_load_task_idx)
            data_obj1_network2_2back, data_obj2_network2_2back = extract_ts_helper(task_ts_reshaped_network2, obj_cond1, obj_cond2, high_load_task_idx)

            # extract the normalized voxel patterns for voxel selection, each in shape (n_voxels, 4 run quarters * 4 trials)
            # run order for the same object category is preserved, and when split across 4-fold cv later, the test and training set use different runs of data
            pattern_cond1_network1_1back, pattern_cond2_network1_1back = extract_pattern_helper(pattern_data_cur_sub_network1_reshaped, obj_cond1, obj_cond2, low_load_task_idx)
            pattern_cond1_network2_1back, pattern_cond2_network2_1back = extract_pattern_helper(pattern_data_cur_sub_network2_reshaped, obj_cond1, obj_cond2, low_load_task_idx)

            pattern_cond1_network1_2back, pattern_cond2_network1_2back = extract_pattern_helper(pattern_data_cur_sub_network1_reshaped, obj_cond1, obj_cond2, high_load_task_idx)
            pattern_cond1_network2_2back, pattern_cond2_network2_2back = extract_pattern_helper(pattern_data_cur_sub_network2_reshaped, obj_cond1, obj_cond2, high_load_task_idx)

            # do voxel selection, fc, and cross-validation decoding, each in shape (n_folds, )
            obj_acc_1back = within_pred_fc_mvpa(data_obj1_network1_1back, data_obj2_network1_1back, data_obj1_network2_1back, data_obj2_network2_1back,
                                                pattern_cond1_network1_1back, pattern_cond2_network1_1back, pattern_cond1_network2_1back, pattern_cond2_network2_1back,
                                                top_k_voxels=top_k_voxels, n_folds=n_folds, normalized=True)

            obj_acc_2back = within_pred_fc_mvpa(data_obj1_network1_2back, data_obj2_network1_2back, data_obj1_network2_2back, data_obj2_network2_2back,
                                                pattern_cond1_network1_2back, pattern_cond2_network1_2back, pattern_cond1_network2_2back, pattern_cond2_network2_2back,
                                                top_k_voxels=top_k_voxels, n_folds=n_folds, normalized=True)

            accuracy_obj_decode_load_subcategory[i_sub, 0, i_cond, :] = obj_acc_1back
            accuracy_obj_decode_load_subcategory[i_sub, 1, i_cond, :] = obj_acc_2back

    # save results
    save_path = f'{args.results_folder}/{network1_name}_{network2_name}.npz'
    np.savez(save_path, accuracy_obj_decode_load_subcategory=accuracy_obj_decode_load_subcategory)
