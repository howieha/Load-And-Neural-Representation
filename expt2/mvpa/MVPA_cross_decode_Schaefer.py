import numpy as np
from scipy.stats import ttest_rel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import argparse
from itertools import combinations


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
        this_tasklist = [1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1] # task list for each run for odd (sub_idx+1) number subjects; 1 and 2 stand for shape 1-back and shape 2-back
    else:
        this_tasklist = [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2] # task list for each run for even (sub_idx+1) number subjects

    if cur_roi_data.ndim != 3 or cur_roi_data.shape[1:] != (16, 16) or cur_roi_data.shape[0] == 0:
        raise ValueError('Expected pattern data with shape (n_voxels, 16, 16), with at least one voxel.')
    if not np.isfinite(cur_roi_data).all():
        raise ValueError('Pattern data must contain only finite values.')

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


def zscore_2d(data, axis=1):
    """
    Z-score an array along a specified axis
    data: in shape for instance, (n_trials, n_voxels)
    """

    data_std = data.std(axis=axis, keepdims=True)
    data_std = np.where(data_std == 0, 1, data_std) # constant patterns become zeros
    data_zscored = (data - data.mean(axis=axis, keepdims=True)) / data_std

    return data_zscored


def cross_pred_cv(cond1_c1, cond1_c2, cond2_c1, cond2_c2, n_folds=4, top_k_voxels=None):
    """
    Voxel-pattern cross-prediction using n-fold cross-validation. Train on condition 1's data, test on condition 2's data.

    cond1_c1: data from condition 1 class 1, in shape (n_trials, n_voxels)
    cond1_c2: data from condition 1 class 2, same shape as above
    cond2_c1, cond2_c2: same as above, for condition 2
    n_folds: number of cross-validation folds (default to 4)
    top_k_voxels: number of voxels kept for decoding, can be an integer or None (default to None)

    returns:
        acc_mat: accuracy for each fold of prediction, in shape (n_folds, )
    """

    n_trials, n_voxels_roi = cond1_c1.shape

    if any(data.shape != cond1_c1.shape for data in (cond1_c2, cond2_c1, cond2_c2)):
        raise ValueError('All four class arrays must have the same shape.')
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2 or n_trials < n_folds or n_trials % n_folds != 0:
        raise ValueError('n_folds must be at least 2 and divide n_trials evenly.')
    if n_voxels_roi == 0 or any(not np.isfinite(data).all() for data in (cond1_c1, cond1_c2, cond2_c1, cond2_c2)):
        raise ValueError('Class arrays must have at least one voxel and contain only finite values.')
    if top_k_voxels is not None and (not isinstance(top_k_voxels, (int, np.integer)) or top_k_voxels < 1):
        raise ValueError('top_k_voxels must be a positive integer or None.')

    n_trials_per_fold = n_trials // n_folds

    acc_mat = np.zeros(n_folds)

    for i_fold in range(n_folds):

        # split training and testing set, each in shape (n_trials, n_voxels)
        test_idx = list(range(i_fold * n_trials_per_fold, (i_fold + 1) * n_trials_per_fold))
        train_idx = list(range(0, i_fold * n_trials_per_fold)) + list(range((i_fold + 1) * n_trials_per_fold, n_trials))

        cond1_c1_tr = cond1_c1[train_idx, :]
        cond1_c2_tr = cond1_c2[train_idx, :]
        cond2_c1_te = cond2_c1[test_idx, :]
        cond2_c2_te = cond2_c2[test_idx, :]

        # select up to top_k_voxels by paired t-test p-value using training data only
        if top_k_voxels is not None:
            _, p_vals_roi = ttest_rel(cond1_c1_tr, cond1_c2_tr, axis=0)

            if n_voxels_roi > top_k_voxels:
                selected_voxels = p_vals_roi.argsort()[:top_k_voxels]
                cond1_c1_tr = cond1_c1_tr[:, selected_voxels]
                cond1_c2_tr = cond1_c2_tr[:, selected_voxels]
                cond2_c1_te = cond2_c1_te[:, selected_voxels]
                cond2_c2_te = cond2_c2_te[:, selected_voxels]

        # prepare for SVM training and testing data, each in shape (n_trials, n_voxels)
        train_data_roi = np.concatenate((cond1_c1_tr, cond1_c2_tr), axis=0)
        train_labels_roi = np.concatenate((np.zeros(cond1_c1_tr.shape[0]), np.ones(cond1_c2_tr.shape[0])), axis=0)

        test_data_roi = np.concatenate((cond2_c1_te, cond2_c2_te), axis=0)
        test_labels_roi = np.concatenate((np.zeros(cond2_c1_te.shape[0]), np.ones(cond2_c2_te.shape[0])), axis=0)

        # SVM training
        svm_cls_roi = SVC(kernel='linear')
        svm_cls_roi.fit(train_data_roi, train_labels_roi)

        # SVM testing
        pred_labels_roi = svm_cls_roi.predict(test_data_roi)
        acc_mat[i_fold] = accuracy_score(test_labels_roi, pred_labels_roi)

    return acc_mat


def extract_obj_cross_load_data(data, load1_idx, load2_idx, obj1_idx, obj2_idx, zscore=True):
    """
    Extract the data of two object categories in two load conditions, and z-score them if needed.

    data: pattern data, in shape (n_voxels, n_task_conditions, n_obj_categories, n_trials)

    Returns:
    load1_obj1, load1_obj2, load2_obj1, load2_obj2 voxel patterns, each in shape (n_trials, n_voxels)
    """

    # each transposed to (n_trials, n_voxels)
    load1_obj1 = data[:, load1_idx, obj1_idx, :].T
    load1_obj2 = data[:, load1_idx, obj2_idx, :].T
    load2_obj1 = data[:, load2_idx, obj1_idx, :].T
    load2_obj2 = data[:, load2_idx, obj2_idx, :].T

    if zscore:
        load1_obj1 = zscore_2d(load1_obj1, axis=1)
        load1_obj2 = zscore_2d(load1_obj2, axis=1)
        load2_obj1 = zscore_2d(load2_obj1, axis=1)
        load2_obj2 = zscore_2d(load2_obj2, axis=1)

    return load1_obj1, load1_obj2, load2_obj1, load2_obj2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-decode experiment 2 voxel patterns for one Schaefer ROI.')
    parser.add_argument('rois_path', help='Text file with one ROI name per line')
    parser.add_argument('line_num', type=int, help='Zero-based ROI line number')
    parser.add_argument('--pattern-folder', default='patterns', help='Folder containing the input .npz files')
    parser.add_argument('--results-folder', default='results/cross_decode', help='Folder for decoding results')
    args = parser.parse_args()

    n_subs = 12
    n_task_conditions, n_obj_categories = 2, 8
    n_trials = 16
    n_folds = 4

    top_k_voxels = 500 # set to None to use all voxels
    normalize_pattern = True # whether to z-score across voxels within each trial before decoding

    # read the roi names
    with open(args.rois_path, 'r') as f:
        roi_names = [line.strip() for line in f]

    if not 0 <= args.line_num < len(roi_names) or not roi_names[args.line_num]:
        parser.error('line_num must identify a nonempty ROI line (starting from 0).')
    roi_name = roi_names[args.line_num] # e.g., 17Networks_LH_ContA_IPS_1

    print('ROI: ', roi_name)

    os.makedirs(args.results_folder, exist_ok=True)

    pattern_file_data = np.load(f'{args.pattern_folder}/{roi_name}_allsubs.npz')

    # get all pairs of object categories
    obj_cat_pairs = list(combinations(range(n_obj_categories), 2))
    load_pair = (0, 1) # low load (1-back), high load (2-back)

    obj_cross_load = np.zeros((n_subs, len(obj_cat_pairs), n_folds))
    load_cross_obj = np.zeros((n_subs, len(obj_cat_pairs), n_folds))

    for i_sub in range(n_subs):
        print(f'ROI: {roi_name}, sub{i_sub+1}')

        # load pattern data in shape (n_voxels, n_beta_conditions, n_runs)
        pattern_data_cur_sub = pattern_file_data[f'subj_{i_sub+1}']

        # reorganize the pattern data to (n_voxels, n_task_conditions*n_obj_categories, n_trials)
        cur_roi_data = reorganize_pattern_data(pattern_data_cur_sub, i_sub)
        n_voxels_roi = cur_roi_data.shape[0]

        # reshape to (n_voxels, n_task_conditions, n_obj_categories, n_trials)
        cur_roi_data = cur_roi_data.reshape(n_voxels_roi, n_task_conditions, n_obj_categories, n_trials)

        for i_obj_cat_pair, (obj1, obj2) in enumerate(obj_cat_pairs):

            # extract the four conditions' data, each in shape (n_trials, n_voxels)
            load1_obj1, load1_obj2, load2_obj1, load2_obj2 = extract_obj_cross_load_data(cur_roi_data, load_pair[0], load_pair[1], obj1, obj2, zscore=normalize_pattern)

            # object decoding across load, in both directions, each in shape (n_folds, )
            acc_load12 = cross_pred_cv(load1_obj1, load1_obj2, load2_obj1, load2_obj2, n_folds=n_folds, top_k_voxels=top_k_voxels)
            acc_load21 = cross_pred_cv(load2_obj1, load2_obj2, load1_obj1, load1_obj2, n_folds=n_folds, top_k_voxels=top_k_voxels)
            obj_cross_load[i_sub, i_obj_cat_pair, :] = (acc_load12 + acc_load21) / 2

            # load decoding across object category, in both directions, each in shape (n_folds, )
            acc_obj12 = cross_pred_cv(load1_obj1, load2_obj1, load1_obj2, load2_obj2, n_folds=n_folds, top_k_voxels=top_k_voxels)
            acc_obj21 = cross_pred_cv(load1_obj2, load2_obj2, load1_obj1, load2_obj1, n_folds=n_folds, top_k_voxels=top_k_voxels)
            load_cross_obj[i_sub, i_obj_cat_pair, :] = (acc_obj12 + acc_obj21) / 2

    # save both cross-decoding accuracy matrices
    if top_k_voxels is None:
        save_path = f'{args.results_folder}/{roi_name}.npz'
    else:
        save_path = f'{args.results_folder}/{roi_name}_top{top_k_voxels}.npz'

    np.savez(save_path, obj_cross_load=obj_cross_load, load_cross_obj=load_cross_obj, obj_cat_pairs=np.array(obj_cat_pairs))
