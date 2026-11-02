import numpy as np
from scipy.stats import ttest_rel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import argparse


def reorganize_pattern_data(cur_roi_data, sub_idx):
    """
    Reorganize one subject's beta patterns by task condition and object category.

    cur_roi_data: raw beta patterns, in shape (n_voxels, 16 beta conditions, 16 runs)
    sub_idx: the subject index, from 0 to n_subs-1; its parity selects the task schedule

    Returns:
    sorted_data_reshaped: the reorganized pattern data, in shape (n_voxels, n_task_conditions*n_obj_categories, n_trials),
        task-major then object category; trials are ordered by run quarter
    """

    n_task_conds = 4
    n_obj_cats = 4

    if sub_idx % 2 == 0:
        this_tasklist = [1, 2, 3, 4, 3, 1, 4, 2, 2, 4, 1, 3, 4, 3, 2, 1] # task list for each run for odd (sub_idx+1) number subjects; 1, 2, 3, 4 stand for color 1-back, color 2-back, shape 1-back, shape 2-back
    else:
        this_tasklist = [2, 4, 1, 3, 4, 3, 2, 1, 1, 2, 3, 4, 3, 1, 4, 2] # task list for each run for even (sub_idx+1) number subjects

    if cur_roi_data.ndim != 3 or cur_roi_data.shape[1:] != (16, 16) or cur_roi_data.shape[0] == 0:
        raise ValueError('Expected pattern data with shape (n_voxels, 16, 16), with at least one voxel.')
    if not np.isfinite(cur_roi_data).all():
        raise ValueError('Pattern data must contain only finite values.')

    n_voxels, n_task_obj_conds, n_trials = cur_roi_data.shape

    sorted_data = np.zeros((n_voxels, n_task_conds, n_obj_cats, n_trials))

    for i in range(n_trials):
        task_idx = this_tasklist[i] - 1 # 0, 1, 2, or 3
        run_quarter = i // 4 # which of the four quarters is this run in

        for j in range(n_task_obj_conds):
            obj_cat_idx = j % 4 # 0, 1, 2, or 3
            thisrunitem = 4 * run_quarter + (j // 4)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode experiment 1 voxel patterns for one Schaefer ROI.')
    parser.add_argument('rois_path', help='Text file with one ROI name per line')
    parser.add_argument('line_num', type=int, help='Zero-based ROI line number')
    parser.add_argument('--pattern-folder', default='patterns', help='Folder containing the input .npz files')
    parser.add_argument('--results-folder', default='results/within_decode', help='Folder for decoding results')
    args = parser.parse_args()

    n_subs = 12
    n_trials = 16
    n_conditions = 16 # 4 task conditions * 4 object categories

    n_folds = 4
    n_trials_per_fold = n_trials // n_folds

    top_k_voxels = 500 # set to None to use all voxels
    normalize_pattern = True # whether to z-score across voxels within each trial before decoding

    # read the roi names
    with open(args.rois_path, 'r') as f:
        roi_names = [line.strip() for line in f]

    if not 0 <= args.line_num < len(roi_names) or not roi_names[args.line_num]:
        parser.error('line_num must identify a nonempty ROI line (starting from 0).')
    roi_name = roi_names[args.line_num] # e.g., 17Networks_RH_SomMotB_S2_2

    print('ROI: ', roi_name)

    os.makedirs(args.results_folder, exist_ok=True)

    pattern_file_data = np.load(f'{args.pattern_folder}/{roi_name}_allsubs.npz')

    # initiate the accuracy matrix for the current ROI
    acc_mat_all = np.zeros((n_subs, n_conditions, n_conditions, n_folds))

    for i_sub in range(n_subs):
        print(f'ROI: {roi_name}, sub{i_sub+1}')

        # load pattern data in shape (n_voxels, n_beta_conditions, n_runs)
        pattern_data_cur_sub = pattern_file_data[f'subj_{i_sub+1}']

        # reorganize the pattern data to (n_voxels, n_task_conditions*n_obj_categories, n_trials)
        cur_roi_data = reorganize_pattern_data(pattern_data_cur_sub, i_sub)
        n_voxels_roi = cur_roi_data.shape[0]

        # pairwise decoding between all 16 conditions (upper triangle only)
        for cur_condition1 in range(n_conditions):

            # get current condition's pattern, transposed to (n_trials, n_voxels)
            voxel_pattern_cond1 = cur_roi_data[:, cur_condition1, :].T

            if normalize_pattern:
                voxel_pattern_cond1 = zscore_2d(voxel_pattern_cond1, axis=1)

            for cur_condition2 in range(cur_condition1+1, n_conditions):

                voxel_pattern_cond2 = cur_roi_data[:, cur_condition2, :].T

                if normalize_pattern:
                    voxel_pattern_cond2 = zscore_2d(voxel_pattern_cond2, axis=1)

                for i_fold in range(n_folds):

                    # split training and testing set, each in shape (n_trials, n_voxels)
                    test_idx = list(range(i_fold * n_trials_per_fold, (i_fold + 1) * n_trials_per_fold))
                    train_idx = list(range(0, i_fold * n_trials_per_fold)) + list(range((i_fold + 1) * n_trials_per_fold, n_trials))

                    cond1_voxel_pattern_tr, cond1_voxel_pattern_te = voxel_pattern_cond1[train_idx, :], voxel_pattern_cond1[test_idx, :]
                    cond2_voxel_pattern_tr, cond2_voxel_pattern_te = voxel_pattern_cond2[train_idx, :], voxel_pattern_cond2[test_idx, :]

                    # select up to top_k_voxels by paired t-test p-value using training data only
                    if top_k_voxels is not None:
                        _, p_vals_voxel_pattern = ttest_rel(cond1_voxel_pattern_tr, cond2_voxel_pattern_tr, axis=0)

                        if n_voxels_roi > top_k_voxels:
                            selected_voxels = p_vals_voxel_pattern.argsort()[:top_k_voxels]
                            cond1_voxel_pattern_tr = cond1_voxel_pattern_tr[:, selected_voxels]
                            cond1_voxel_pattern_te = cond1_voxel_pattern_te[:, selected_voxels]
                            cond2_voxel_pattern_tr = cond2_voxel_pattern_tr[:, selected_voxels]
                            cond2_voxel_pattern_te = cond2_voxel_pattern_te[:, selected_voxels]

                    # prepare for SVM training and testing data, each in shape (n_trials, n_voxels)
                    train_data_voxel_pattern = np.concatenate((cond1_voxel_pattern_tr, cond2_voxel_pattern_tr), axis=0)
                    train_labels = np.concatenate((np.zeros(cond1_voxel_pattern_tr.shape[0]), np.ones(cond2_voxel_pattern_tr.shape[0])), axis=0)

                    test_data_voxel_pattern = np.concatenate((cond1_voxel_pattern_te, cond2_voxel_pattern_te), axis=0)
                    test_labels = np.concatenate((np.zeros(cond1_voxel_pattern_te.shape[0]), np.ones(cond2_voxel_pattern_te.shape[0])), axis=0)

                    # SVM training
                    svm_voxel_pattern = SVC(kernel='linear')
                    svm_voxel_pattern.fit(train_data_voxel_pattern, train_labels)

                    # SVM testing
                    pred_labels_voxel_pattern = svm_voxel_pattern.predict(test_data_voxel_pattern)
                    cur_acc = accuracy_score(test_labels, pred_labels_voxel_pattern)

                    acc_mat_all[i_sub, cur_condition1, cur_condition2, i_fold] = cur_acc

    # save acc_mat_all for the current ROI (upper triangle only; other entries remain zero)
    if top_k_voxels is None:
        save_path = f'{args.results_folder}/{roi_name}.npy'
    else:
        save_path = f'{args.results_folder}/{roi_name}_top{top_k_voxels}.npy'
    np.save(save_path, acc_mat_all)
