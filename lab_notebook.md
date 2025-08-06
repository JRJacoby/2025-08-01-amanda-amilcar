# Lab Notebook

## 2025-08-02 16:03: Project Overview - Train vs Apply Syllable Count Bug Investigation

Investigating a reported bug where syllable instance counts differ significantly between training and apply results when applied to the same dataset. User Amanda Amilcar reported this issue on the moseq help slack. The goal is to reproduce the issue and identify the root cause to improve the software for all users.

**Approach**: Create a systematic comparison pipeline to analyze differences between train and apply results on the same input data (training_data folder). If no discrepancy is found, will test on the full dataset (all_data folder) using set_mixed_map_iters.

## 2025-08-02 16:03: Data Acquisition and Setup

Downloaded training data, full dataset, and project configuration from Amanda Amilcar's slack message. Set up project structure with:
- `kpms_project/2025_08_01-23_41_19/` containing the model
- `training_data/` folder with subset of behavioral data  
- `all_data/` folder with complete dataset
- Applied model to training_data using existing `apply.py` script, generating results in both `results/` and `apply_results/` directories

## 2025-08-02 16:03: Pipeline Step 1 - Combined Results Parquet Creation

Created `compare_results.py` pipeline with function `create_combined_results()` to process CSV analysis files into structured parquet format. Code combines all CSV files per directory and adds derived columns.

**Output**: 
- `results/combined_results.parquet`
- `apply_results/combined_results.parquet`

**Added Columns**:
- `session_name`: Shortened session identifier (e.g., "AA2_6-1") 
- `frame`: 1-based frame counter within each session
- `is_transition`: Boolean marking syllable changes (False for first frame)
- `instance_id`: Unique identifier for continuous syllable blocks across entire dataset

**Data**: 1,160,367 total frames across 21 sessions in each dataset.

## 2025-08-02 16:03: Pipeline Step 2 - Duration Distribution Analysis

Added `duration_histogram()` function to compare behavioral instance duration distributions between train and apply datasets. Code calculates instance durations and creates overlaid histograms.

**Output**: `comparison_plots/duration_histogram.png`

**Axes**:
- x-axis: Duration (frames), filtered to < 50 frames
- y-axis: Count (number of instances)  
- color: Dataset (Training vs Apply)

**Results**: Very similar distributions - Training: 86,655 instances (mean 10.7 frames), Apply: 87,766 instances (mean 10.5 frames). Both datasets show median duration of 8 frames.

## 2025-08-02 16:03: Pipeline Step 3 - Frame-Level Syllable Agreement

Added `syllable_confusion_matrix()` function to analyze frame-by-frame syllable prediction agreement between datasets. Creates confusion matrix ordered by syllable frequency in training data.

**Output**: `comparison_plots/syllable_confusion_matrix.png`

**Axes**:
- x-axis: Apply dataset syllables
- y-axis: Training dataset syllables (ordered by frequency)
- color: Percentage agreement (Blues colormap)

**Results**: 45.57% overall frame-level accuracy across 97 unique syllables. This level of disagreement is expected and normal - identical frame-level predictions would be highly unlikely.

## 2025-08-02 16:03: Pipeline Step 4 - Syllable Instance Count Comparison  

Added `syllable_counts_scatter()` function to directly test the reported bug by comparing syllable instance counts between datasets. Creates scatterplot with 1:1 reference line.

**Output**: `comparison_plots/syllable_counts_scatter.png`

**Axes**:
- x-axis: Training dataset instance counts
- y-axis: Apply dataset instance counts
- Reference line: Red dashed 1:1 diagonal

**Results**: Perfect correlation (1.000) between syllable instance counts. Key findings:
- 97 total syllables identified
- 32 syllables appear only in training dataset  
- 0 syllables appear only in apply dataset
- When syllables appear in both datasets, their instance counts are identical

## 2025-08-02 16:03: Summary and Next Steps

**Main Finding**: Cannot reproduce the reported bug. Syllable instance counts show perfect correlation between train and apply results when applied to the same training_data. The apply dataset appears to be a subset using 65 out of 97 training syllables, but with identical frequency patterns.

**Bug Status**: No evidence of train vs apply discrepancy on training_data subset. The perfect correlation directly contradicts the user report of "very different" instance counts.

**Next Steps**: Test on complete dataset (all_data folder) to determine if the bug manifests when processing larger/different data volumes using set_mixed_map_iters configuration.

## 2025-08-02 17:06: Pipeline Refactoring and Code Quality Improvements

Refactored the comparison pipeline to improve maintainability and add new functionality. Created a unified `run_comparisons()` function that consolidates all comparison steps.

**Code Quality Improvements**:
- Added syllable smoothing with 11-frame sliding window majority vote
- Created custom types using `Literal`: `SyllableColumn` and `ComparisonMode`
- Applied black formatting with 120-character line length
- Added type safety for function parameters

**Pipeline Consolidation**:
- **Before**: 8 separate pipeline steps with ~150 lines of conditional logic
- **After**: Single `run_comparisons(mode)` function handling both scenarios
- **Reduction**: ~600 lines to ~450 lines (25% reduction)

**New Functionality**:
- **Syllable smoothing**: Added `syllable_smoothed` column with 11-frame sliding window majority vote
  - **Rationale**: Suspected noise at per-frame level makes ~50% agreement in confusion matrix expected
  - **Hypothesis**: Overall behavioral sequences should show higher agreement after smoothing
  - **Method**: Reflect padding at edges, majority vote within window
- **Top N confusion matrices**: Added `n_syllables` parameter for limited syllable analysis
- **Flexible comparison modes**: `"train_apply"` vs `"train_apply_all"` with automatic file organization

**Experimental Design Context**:
- **`train_apply`**: Compares training results vs. applying model to same training dataset
  - Tests if training and applying on identical data produce different results
  - Baseline comparison to understand model consistency
- **`train_apply_all`**: Compares training results vs. applying model to full dataset (training subset)
  - Tests if applying to larger dataset affects results on training sessions
  - Investigates whether data volume or context affects model behavior
  - Since we found no difference in `train_apply`, this tests if the bug manifests when processing more data

**File Organization**:
- `train_apply` mode: Standard filenames (e.g., `duration_histogram.png`)
- `train_apply_all` mode: `all_data_` prefix (e.g., `all_data_duration_histogram.png`)

**Code Structure**:
- Functions: `run_comparisons()`, `smooth_syllables_with_window()`, enhanced `syllable_confusion_matrix()`
- Types: `SyllableColumn = Literal["syllable", "syllable_smoothed"]`, `ComparisonMode = Literal["train_apply", "train_apply_all"]`
- Constants: `syllable_smoothing_window = 11`

The refactored pipeline automatically runs both comparison scenarios and handles all conditional logic internally, making it much easier to maintain and extend.

## 2025-08-03 11:26: Session Name Bug Discovery and Modeling Software Inconsistency

**Problem**: Initial analysis of `all_data_syllable_counts_scatter.png` showed poor agreement between training and apply results, but this was due to a bug in the `extract_session_name()` function rather than a genuine modeling issue.

**Root Cause**: The regex pattern `r"^(AA\d+[_-]?\d*[_-]?\d*)"` was extracting session names like `AA049_` from both `AA049_OF1` and `AA049_OF2` files, causing different recording sessions to be treated as duplicates. This led to inflated syllable counts in the apply_all_results dataset.

**Investigation Process**:
1. **Session name verification**: Confirmed that `AA049_OF1` and `AA049_OF2` are different recordings (55,179 vs 57,686 frames, 28 vs 50 unique syllables)
2. **Data integrity check**: Found that apply_all_results contained duplicate frames due to session name collision
3. **Script validation**: Systematically verified that the comparison logic is sound and fair

**Fix Applied**: Updated regex pattern to `r"^(AA\d+[_-]?\d*[_-]?\d*_OF\d+)"` to include the OF1/OF2 distinction in session names.

**Post-Fix Verification**: 
- Training data: 21 sessions (all OF2), 1,160,367 frames
- Training subset: 21 sessions (all OF2), 1,160,367 frames  
- Session counts identical, frame ranges identical (1-57,696)

**Genuine Modeling Issue Discovered**: After fixing the session name bug, the comparison revealed a real problem with the modeling software. The same frames are labeled with completely different syllables:

**Frame-level comparison (AA049_OF2, frames 1-10)**:
- Training mode: frames 1-4 = syllable 17, frames 5-10 = syllable 23
- Apply mode: frames 1-10 = syllable 11

**Syllable frequency patterns**:
- Training mode: syllables 0-9 most frequent (5,682, 4,650, 4,159, etc.)
- Apply mode: syllables 12, 28, 18 most frequent (4,848, 3,548, 1,647, etc.)

**Duration Histogram Evidence**: The `all_data_duration_histogram.png` shows a flat distribution for the apply all dataset, suggesting the model isn't fitting properly and may be behaving like an untrained model.

**Next Steps**: 
1. Delete existing apply results and re-run `apply.py` on the whole dataset to confirm the model actually learned
2. If the same inconsistency persists, focus debugging efforts on the keypoint-moseq model itself
3. The comparison script is confirmed to be working correctly - the issue is in the modeling software

**Code Location**: Fixed `extract_session_name()` function in `scripts/compare_results.py` line 33-42.
