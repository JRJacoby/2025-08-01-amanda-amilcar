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

## 2025-01-27 18:30 - Extended Comparison Analysis with HTML Reporting

**Problem**: The existing comparison script provided basic statistical comparisons and visualizations, but lacked comprehensive analysis of behavioral sequences, latent space representations, advanced statistical testing, and user-friendly reporting capabilities.

**Solution**: Created `scripts/compare_results_extended.py` - a comprehensive analysis pipeline that extends the existing comparison functionality with advanced statistical methods, machine learning analysis, and professional HTML reporting.

**Key Features**:

*Statistical Analysis*:
- Frame-by-frame accuracy assessment using sklearn classification metrics
- Adjusted Rand Index and Normalized Mutual Information for clustering agreement
- Spearman and Pearson correlation analysis for syllable count distributions
- Kolmogorov-Smirnov test for behavioral bout duration distributions
- Full classification report with precision, recall, and F1-scores per syllable

*Behavioral Sequence Analysis*:
- Syllable transition probability matrices and entropy calculations
- Behavioral bout duration statistics (mean, median, maximum, standard deviation)
- Transition pattern comparison between training and apply datasets
- Top 10 most frequent transitions identification

*Latent Space Analysis*:
- PCA dimensionality reduction on 4D latent state representations
- t-SNE embedding for non-linear structure visualization
- Dataset and syllable-based clustering visualization
- Explained variance analysis for principal components

*Advanced Visualizations*:
- Static plots: latent space PCA/t-SNE, transition heatmaps, syllable repertoire analysis, session-wise comparisons
- Interactive plots: Plotly-based confusion matrices and latent space explorers with hover functionality
- Professional matplotlib styling with consistent color schemes

*HTML Report Generation*:
- Comprehensive single-page report with embedded visualizations
- Executive summary with key findings and interpretations
- Responsive design with modern CSS styling
- Base64-encoded images for standalone report portability
- Interactive plot links for detailed exploration

**Analysis Workflow**:
1. Data loading and validation with comprehensive statistics
2. Frame-by-frame alignment for direct comparison accuracy
3. Statistical testing suite with multiple correlation and distribution tests
4. Behavioral pattern analysis including transitions and bout characteristics
5. Dimensionality reduction and clustering analysis of latent representations
6. Static and interactive visualization generation
7. JSON results serialization with numpy type conversion
8. Professional HTML report compilation with Jinja2 templating

**Code Structure**:
- Main class: `ExtendedAnalysis` with modular analysis methods
- Analysis methods: `statistical_comparisons()`, `behavioral_sequence_analysis()`, `latent_space_analysis()`
- Visualization methods: `generate_advanced_plots()`, `generate_interactive_plots()`
- Plotting functions: `_plot_latent_space_pca()`, `_plot_transition_heatmaps()`, `_plot_syllable_repertoire()`
- Report generation: `generate_html_report()` with comprehensive templating
- Main function: `run_extended_analysis()` with configurable modes

**Output Files**:
- HTML report: `kpms_project/{model_name}/extended_analysis/comparison_report.html`
- Static plots: `kpms_project/{model_name}/extended_comparison_plots/*.png`
- Interactive plots: `kpms_project/{model_name}/extended_comparison_plots/*.html`
- Results JSON: `kpms_project/{model_name}/extended_analysis/extended_analysis_results.json`

**Dependencies**: Extends existing polars/matplotlib stack with scipy, sklearn, plotly, jinja2, and pandas for comprehensive analysis capabilities. Compatible with both "train_apply" and "train_apply_all" comparison modes from the base script.

**Technical Notes**: The script handles large datasets efficiently through strategic subsampling for t-SNE (max 5000 points) and interactive plots (max 10000 points). All numpy data types are properly converted for JSON serialization, and comprehensive error handling ensures robust execution.