# MVPA and FCPA decoding code

Python code for the voxel-pattern (MVPA) and functional-connectivity-pattern (FCPA) decoding analyses of two fMRI experiments.

```
expt1/                          # experiment 1: color/shape attention x 1-back/2-back load, 4 object categories
    mvpa/MVPA_within_decode_Schaefer.py
    mvpa/MVPA_cross_decode_Schaefer.py
    fcpa/FCPA_within_obj_decode.py
expt2/                          # experiment 2: shape 1-back/2-back load, 8 object categories
    mvpa/MVPA_within_decode_Schaefer.py
    mvpa/MVPA_cross_decode_Schaefer.py
    fcpa/FCPA_within_obj_decode.py
```

Requires Python 3, NumPy, SciPy, and scikit-learn (tested with NumPy 1.26, SciPy 1.13, and scikit-learn 1.3).

## Usage

Each script analyzes one ROI (MVPA) or one ROI pair (FCPA), selected by a zero-based line number in a text file, so that ROIs can be run in parallel (e.g., as a job array).

```bash
# MVPA: rois.txt has one ROI name per line, e.g., 17Networks_LH_ContA_IPS_1
python MVPA_within_decode_Schaefer.py rois.txt 0 --pattern-folder patterns --results-folder results/within_decode
python MVPA_cross_decode_Schaefer.py rois.txt 0 --pattern-folder patterns --results-folder results/cross_decode

# FCPA: roi_pairs.txt has one comma-separated ROI pair per line
python FCPA_within_obj_decode.py roi_pairs.txt 0 \
    --ts-folder time_series --pattern-folder patterns \
    --fmri-data-folder fmri_data --subject-folders subject_folders.local.txt \
    --results-folder results/fcpa
```

Folder arguments default to the values shown above (`--fmri-data-folder` defaults to `fmri_data`) and are relative to the working directory. `--subject-folders` is required for FCPA: a text file listing the 12 subject folder names in `subj_1` to `subj_12` order. Existing results with the same name are overwritten. Analysis settings such as `top_k_voxels` are set at the top of each script's main block.

## Inputs

All data files are named `<ROI>_allsubs.npz` and contain one array per subject, keyed `subj_1` to `subj_12`.

- **Voxel patterns** (`--pattern-folder`): beta values in shape `(n_voxels, 16 beta conditions, 16 runs)`. In experiment 1, beta index modulo 4 is the object category and integer division by 4 is the repetition within a run. In experiment 2, beta index modulo 8 is the object category and integer division by 8 is the exemplar (first or second) within a run.
- **Time series** (FCPA only, `--ts-folder`): shape `(n_voxels, n_TRs, 16 runs)`, with at least 317 TRs. Sixteen 10-TR task blocks are taken starting at TR 7 (a 7-TR hemodynamic lag), with a 20-TR block onset interval. Voxel order must match the voxel patterns.
- **Trial order** (FCPA only, `--fmri-data-folder`): each subject folder contains `bold/<task>_all_runs.txt` listing the run folders (`<task>` is `colordot` in experiment 1, `shapeload` in experiment 2). The 16 run names (in experiment 2, the first 16 entries) are sorted by name, and this order must match the run axis of the data. Each run folder contains `<task>.par`, a tab-separated file whose second column is the condition ID. The first row is a practice trial and is discarded.

Runs are assigned to task conditions by the two task schedules defined in each script, selected by subject-number parity. Task conditions are ordered color 1-back, color 2-back, shape 1-back, shape 2-back in experiment 1, and 1-back (low load), 2-back (high load) in experiment 2.

## Analysis

- **MVPA**: patterns are z-scored across voxels within each trial. Each pair of conditions is decoded with a linear SVM (`SVC(kernel='linear')`) using 4-fold cross-validation, where each fold holds out one quarter of the run schedule. In each fold, up to 500 voxels are selected by paired t-test on the training data only. For cross-decoding, the classifier is trained on one condition and tested on the other, and accuracy is averaged across both train/test directions.
- **FCPA**: voxels with nonfinite values or constant task blocks are removed. Each task block is z-scored across time, and each pattern across voxels. The four trials of the same object category within a run (experiment 1) or within two runs of the same load (experiment 2) are concatenated into one 40-TR sample. In each fold, up to 500 voxels per ROI are selected by paired t-test on the training voxel patterns. FC features are the Pearson correlations between all selected voxel pairs across the two ROIs, z-scored within each sample, and decoded with a linear SVM using 4-fold cross-validation.

## Outputs

MVPA results are saved as `<ROI>_top500.npy` / `.npz` (`<ROI>.npy` / `.npz` if `top_k_voxels = None`); FCPA results as `<ROI1>_<ROI2>.npz`. Object-category pairs follow `itertools.combinations` order. The last axis of every array is the cross-validation fold.

| Script | Array | Shape | Axes |
| --- | --- | --- | --- |
| expt1 MVPA within | (`.npy`) | (12, 16, 16, 4) | subject, condition 1, condition 2 (task-major, then object), fold; upper triangle only |
| expt1 MVPA cross | `accuracy_obj_cross_decode_roi` | (12, 4, 6, 4) | subject, task pair `(0,1), (2,3), (0,2), (1,3)`, object pair, fold |
| | `accuracy_attn_load_cross_obj_roi` | (12, 6, 4, 4) | subject, object pair, decoded task pair `(0,2), (1,3), (0,1), (2,3)`, fold |
| | `accuracy_cross_attn_load` | (12, 4, 2, 4) | subject, object, attention across load / load across attention, fold |
| expt1 FCPA | `accuracy_obj_decode_load_subcategory` | (12, 2, 6, 4) | subject, low / high load, object pair, fold |
| | `accuracy_obj_decode_attn_subcategory` | (12, 2, 6, 4) | subject, color / shape attention, object pair, fold |
| expt2 MVPA within | (`.npy`) | (12, 16, 16, 4) | subject, condition 1, condition 2 (load-major, then object), fold; upper triangle only |
| expt2 MVPA cross | `obj_cross_load` | (12, 28, 4) | subject, object pair, fold |
| | `load_cross_obj` | (12, 28, 4) | subject, object pair, fold |
| | `obj_cat_pairs` | (28, 2) | object pairs |
| expt2 FCPA | `accuracy_obj_decode_load_subcategory` | (12, 2, 28, 4) | subject, low / high load, object pair, fold |

In the within-decoding matrices, the diagonal and lower triangle are zero placeholders and should be excluded when summarizing accuracy.

## License

MIT License; see [LICENSE](LICENSE).
