import polars as pl
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns
import numpy as np
from typing import Literal

# Custom types
SyllableColumn = Literal["syllable", "syllable_smoothed"]
ComparisonMode = Literal["train_apply", "train_apply_all"]

project_dir = Path("kpms_project")
model_name = "2025_08_01-23_41_19"
results_dir = project_dir / model_name / "results"
apply_results_dir = project_dir / model_name / "apply_results"
apply_all_results_dir = project_dir / model_name / "apply_all_results"
plots_dir = project_dir / model_name / "comparison_plots"
syllable_smoothing_window = 11
combined_filename = "combined_results.parquet"
training_subset_filename = "combined_results_training_subset.parquet"
duration_histogram_filename = "duration_histogram.png"
confusion_matrix_filename = "syllable_confusion_matrix.png"
confusion_matrix_smoothed_filename = "syllable_confusion_matrix_smoothed.png"
confusion_matrix_top50_filename = "syllable_confusion_matrix_top50.png"
confusion_matrix_smoothed_top50_filename = "syllable_confusion_matrix_smoothed_top50.png"
syllable_counts_scatter_filename = "syllable_counts_scatter.png"
syllable_counts_scatter_training_subset_filename = "syllable_counts_scatter_training_subset.png"


def extract_session_name(filename: str) -> str:
    """Extract shortened session name from CSV filename."""
    # Extract the part before the first timestamp (e.g., AA2_6-1_OF2.000_BehaviorVid)
    # Then take just the animal/session identifier (e.g., AA2_6-1)
    # Updated to include OF1/OF2 part to distinguish different recordings
    match = re.match(r"^(AA\d+[_-]?\d*[_-]?\d*_OF\d+)", filename)
    if match:
        return match.group(1)
    # Fallback: take everything before the first timestamp
    return filename.split("2025-")[0].rstrip("_").rstrip("Vid")


def smooth_syllables_with_window(syllables: pl.Series, window_size: int = syllable_smoothing_window) -> pl.Series:
    """
    Apply sliding window majority vote smoothing to syllable sequence.
    Uses reflect padding strategy at edges.
    """
    syllable_list = syllables.to_list()
    n = len(syllable_list)
    half_window = window_size // 2
    smoothed = []

    for i in range(n):
        # Calculate window bounds
        start = i - half_window
        end = i + half_window + 1

        # Create window with reflect padding
        window_syllables = []
        for j in range(start, end):
            if j < 0:
                # Reflect at left edge
                window_syllables.append(syllable_list[-j])
            elif j >= n:
                # Reflect at right edge
                window_syllables.append(syllable_list[2 * n - j - 2])
            else:
                window_syllables.append(syllable_list[j])

        # Find most common syllable in window (majority vote)
        from collections import Counter

        counter = Counter(window_syllables)
        most_common_syllable = counter.most_common(1)[0][0]
        smoothed.append(most_common_syllable)

    return pl.Series(smoothed)


def create_combined_results(input_dir: Path, output_file: Path) -> None:
    """
    Concatenate all CSV files in directory into a single parquet with additional columns.
    """
    print(f"Creating combined results for {input_dir}")

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    all_data = []
    global_instance_id = 0

    for csv_file in sorted(csv_files):
        print(f"Processing {csv_file.name}")

        # Read CSV and add session_name
        session_name = extract_session_name(csv_file.stem)
        df = pl.read_csv(csv_file).with_columns(pl.lit(session_name).alias("session_name"))

        # Add frame column (1-based counter)
        df = df.with_columns((pl.int_range(len(df)) + 1).alias("frame"))

        # Add is_transition column (True when syllable changes from previous row)
        df = df.with_columns(
            pl.concat([pl.lit(False), pl.col("syllable").diff().ne(0).slice(1)]).alias(  # First row is always False
                "is_transition"
            )
        )

        # Add instance_id column (unique identifier for continuous syllable blocks)
        # Calculate cumulative sum of transitions to create block IDs
        transition_cumsum = df.select(pl.col("is_transition").cast(pl.Int32).cum_sum()).to_series()

        df = df.with_columns((transition_cumsum + global_instance_id).alias("instance_id"))

        # Add syllable smoothing
        df = df.with_columns(smooth_syllables_with_window(df.select("syllable").to_series()).alias("syllable_smoothed"))

        # Update global instance_id for next session
        global_instance_id = df.select(pl.col("instance_id").max()).item() + 1

        all_data.append(df)

    # Concatenate all sessions
    combined_df = pl.concat(all_data, how="vertical")

    # Reorder columns to put new columns after syllable
    column_order = [
        "session_name",
        "frame",
        "syllable",
        "syllable_smoothed",
        "is_transition",
        "instance_id",
        "centroid x",
        "centroid y",
        "heading",
        "latent_state 0",
        "latent_state 1",
        "latent_state 2",
        "latent_state 3",
    ]
    combined_df = combined_df.select(column_order)

    # Save as parquet
    combined_df.write_parquet(output_file)
    print(f"Saved combined results to {output_file}")
    print(f"Total rows: {len(combined_df)}, Total sessions: {len(csv_files)}")


def should_run_step(output_file: Path, dependencies) -> bool:
    """
    Check if step should run based on file timestamps.

    Args:
        output_file: The output file to check
        dependencies: Either a Path (directory to check for CSVs) or list of Paths (specific files)

    Returns:
        True if output doesn't exist or is older than any dependency
    """
    if not output_file.exists():
        return True

    output_mtime = os.path.getmtime(output_file)

    # Handle directory dependency (check all CSV files)
    if isinstance(dependencies, Path):
        input_files = list(dependencies.glob("*.csv"))
    # Handle list of file dependencies
    else:
        input_files = dependencies

    for input_file in input_files:
        if input_file.exists() and os.path.getmtime(input_file) > output_mtime:
            return True

    return False


def calculate_instance_durations(parquet_file: Path) -> pl.DataFrame:
    """
    Calculate the duration (number of frames) for each instance_id.
    Returns DataFrame with instance_id and duration columns.
    """
    df = pl.read_parquet(parquet_file)
    durations = df.group_by("instance_id").agg(pl.len().alias("duration")).sort("instance_id")
    return durations


def duration_histogram(results_parquet: Path, apply_results_parquet: Path, output_file: Path) -> None:
    """
    Create overlaid histograms comparing instance durations between results and apply_results.
    """
    print(f"Creating duration histogram: {output_file}")

    # Calculate durations for both datasets
    results_durations = calculate_instance_durations(results_parquet)
    apply_durations = calculate_instance_durations(apply_results_parquet)

    # Add dataset labels and combine
    results_labeled = results_durations.with_columns(pl.lit("Training").alias("dataset"))
    apply_labeled = apply_durations.with_columns(pl.lit("Apply").alias("dataset"))

    combined_durations = pl.concat([results_labeled, apply_labeled])

    # Filter to only include durations < 50 frames
    max_duration = 50
    filtered_durations = combined_durations.filter(pl.col("duration") < max_duration)

    # Create histogram plot using seaborn.objects
    plot = (
        so.Plot(filtered_durations.to_pandas(), x="duration", color="dataset")
        .add(so.Bars(alpha=0.7), so.Hist(bins=30))
        .layout(size=(10, 6))
        .label(
            x="Duration (frames)",
            y="Count",
            title=f"Distribution of Behavioral Syllable Block Durations (< {max_duration} frames)",
        )
        .scale(color=so.Nominal(["#2E8B57", "#FF6347"]))  # SeaGreen, Tomato
    )

    # Save the plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output_file, dpi=300, bbox_inches="tight")

    # Print summary statistics for filtered data
    stats_summary = filtered_durations.group_by("dataset").agg(
        [
            pl.col("duration").min().alias("min_duration"),
            pl.col("duration").max().alias("max_duration"),
            pl.col("duration").mean().alias("mean_duration"),
            pl.col("duration").median().alias("median_duration"),
            pl.len().alias("total_instances"),
        ]
    )

    print(f"Duration statistics by dataset (< {max_duration} frames):")
    print(stats_summary)
    print(f"Plot saved to: {output_file}")


def syllable_confusion_matrix(
    results_parquet: Path,
    apply_results_parquet: Path,
    output_file: Path,
    syllable_column: SyllableColumn = "syllable",
    n_syllables: int = None,
) -> None:
    """
    Create a confusion matrix comparing syllable predictions between training and apply datasets.

    Args:
        results_parquet: Path to training results parquet
        apply_results_parquet: Path to apply results parquet
        output_file: Path to save confusion matrix plot
        syllable_column: Column name to use for comparison (one of the literal strings defining the SyllableColumn type)
        n_syllables: Number of most frequent syllables to include (None for all)
    """
    print(f"Creating syllable confusion matrix ({syllable_column}): {output_file}")

    # Load both datasets
    results_df = pl.read_parquet(results_parquet)
    apply_df = pl.read_parquet(apply_results_parquet)

    # Get syllable frequency order from training dataset (by number of instances)
    syllable_freq = (
        results_df.group_by(syllable_column)
        .agg(pl.col("instance_id").n_unique().alias("num_instances"))
        .sort("num_instances", descending=True)
    )

    # Limit to top N syllables if specified
    if n_syllables is not None:
        syllable_freq = syllable_freq.head(n_syllables)
        print(f"Limiting to top {n_syllables} most frequent syllables")

    syllable_order = syllable_freq.select(syllable_column).to_series().to_list()

    # Create frame-aligned comparison
    # Sort both datasets by session_name and frame to ensure alignment
    results_sorted = (
        results_df.sort(["session_name", "frame"])
        .select(["session_name", "frame", syllable_column])
        .rename({syllable_column: "train_syllable"})
    )
    apply_sorted = (
        apply_df.sort(["session_name", "frame"])
        .select(["session_name", "frame", syllable_column])
        .rename({syllable_column: "apply_syllable"})
    )

    # Join on session_name and frame to align predictions
    aligned_df = results_sorted.join(apply_sorted, on=["session_name", "frame"], how="inner")

    # Create confusion matrix using polars
    confusion_counts = aligned_df.group_by(["train_syllable", "apply_syllable"]).agg(pl.len().alias("count"))

    # Convert to pandas for easier matrix manipulation
    confusion_pandas = confusion_counts.to_pandas()

    # Create full confusion matrix with syllables ordered by frequency in training
    # Include any syllables that appear in apply but not training (rare case)
    # When limiting to top N, only include apply syllables that are in the training top N
    apply_syllables = set(apply_df.select(syllable_column).unique().to_series().to_list())
    if n_syllables is not None:
        # Only include apply syllables that are in the training top N
        all_syllables = syllable_order
        # Filter confusion data to only include syllables in the allowed set
        allowed_syllables = set(syllable_order)
        confusion_pandas = confusion_pandas[
            confusion_pandas["train_syllable"].isin(allowed_syllables) & 
            confusion_pandas["apply_syllable"].isin(allowed_syllables)
        ]
    else:
        all_syllables = syllable_order + [s for s in sorted(apply_syllables) if s not in syllable_order]
    confusion_matrix = np.zeros((len(all_syllables), len(all_syllables)))

    # Fill in the counts
    syllable_to_idx = {syll: i for i, syll in enumerate(all_syllables)}
    for _, row in confusion_pandas.iterrows():
        train_idx = syllable_to_idx[row["train_syllable"]]
        apply_idx = syllable_to_idx[row["apply_syllable"]]
        confusion_matrix[train_idx, apply_idx] = row["count"]

    # Normalize by rows (training syllables) to get percentages
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_percent = (confusion_matrix / row_sums) * 100

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion_percent,
        xticklabels=all_syllables,
        yticklabels=all_syllables,
        annot=False,
        cmap="Blues",
        cbar_kws={"label": "Percentage (%)"},
    )

    column_label = "Smoothed " if syllable_column == "syllable_smoothed" else ""
    top_n_label = f" (Top {n_syllables})" if n_syllables is not None else ""
    plt.title(f"{column_label}Syllable Prediction Confusion Matrix{top_n_label}\n(Training vs Apply)")
    plt.xlabel(f"Apply Dataset {column_label}Syllables{top_n_label}")
    plt.ylabel(f"Training Dataset {column_label}Syllables{top_n_label}")
    plt.tight_layout()

    # Save the plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    total_frames = len(aligned_df)
    diagonal_sum = np.trace(confusion_matrix)
    accuracy = (diagonal_sum / total_frames) * 100

    print(f"Total aligned frames: {total_frames}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Number of unique syllables: {len(all_syllables)}")
    print(f"Plot saved to: {output_file}")


def syllable_counts_scatter(results_parquet: Path, apply_results_parquet: Path, output_file: Path) -> None:
    """
    Create a scatterplot comparing syllable instance counts between training and apply datasets.
    """
    print(f"Creating syllable counts scatterplot: {output_file}")

    # Load both datasets
    results_df = pl.read_parquet(results_parquet)
    apply_df = pl.read_parquet(apply_results_parquet)

    # Calculate instance counts for each syllable in both datasets
    results_counts = results_df.group_by("syllable").agg(pl.col("instance_id").n_unique().alias("train_instances"))

    apply_counts = apply_df.group_by("syllable").agg(pl.col("instance_id").n_unique().alias("apply_instances"))

    # Join the counts together
    combined_counts = results_counts.join(apply_counts, on="syllable", how="full").fill_null(0)

    # Convert to pandas for plotting
    counts_pandas = combined_counts.to_pandas()

    # Create the scatterplot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        counts_pandas["train_instances"],
        counts_pandas["apply_instances"],
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add 1:1 reference line
    max_count = max(counts_pandas[["train_instances", "apply_instances"]].max())
    plt.plot([0, max_count], [0, max_count], "r--", linewidth=2, alpha=0.8, label="1:1 line")

    # Formatting
    plt.xlabel("Training Dataset Instance Counts")
    plt.ylabel("Apply Dataset Instance Counts")
    plt.title("Syllable Instance Counts: Training vs Apply")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Make axes equal and start from 0
    plt.axis("equal")
    plt.xlim(0, None)
    plt.ylim(0, None)

    # Save the plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate correlation and summary stats
    correlation = counts_pandas[["train_instances", "apply_instances"]].corr().iloc[0, 1]
    total_syllables = len(counts_pandas)
    train_only = len(counts_pandas[counts_pandas["apply_instances"] == 0])
    apply_only = len(counts_pandas[counts_pandas["train_instances"] == 0])

    print(f"Total syllables: {total_syllables}")
    print(f"Syllables only in training: {train_only}")
    print(f"Syllables only in apply: {apply_only}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Plot saved to: {output_file}")


def get_training_session_names():
    """
    Get the list of training session names from the training_data directory.
    """
    training_files = list(Path("training_data").glob("*.h5"))
    session_names = [extract_session_name(f.stem) for f in training_files]
    return session_names


def filter_and_save_training_subset(input_parquet: Path, output_parquet: Path) -> None:
    """
    Filter a combined results parquet to only include training session names and save.
    """
    print(f"Filtering to training subset: {output_parquet}")

    # Load the full parquet
    full_df = pl.read_parquet(input_parquet)

    # Get training session names
    training_sessions = get_training_session_names()

    # Filter to only training sessions
    filtered_df = full_df.filter(pl.col("session_name").is_in(training_sessions))

    # Save filtered results
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.write_parquet(output_parquet)

    print(f"Filtered from {len(full_df)} to {len(filtered_df)} rows")
    print(f"Training sessions: {len(training_sessions)}")
    print(f"Subset saved to: {output_parquet}")


def run_comparisons(mode: ComparisonMode = "train_apply") -> None:
    """
    Run all comparison analyses between training and apply datasets.
    
    Args:
        mode: Comparison mode - "train_apply" for results vs apply_results,
              "train_apply_all" for results vs apply_all_results (with training subset)
    """
    print(f"Running comparisons in {mode} mode")
    
    # Define paths based on mode
    if mode == "train_apply":
        train_parquet = results_dir / combined_filename
        apply_parquet = apply_results_dir / combined_filename
        apply_dir = apply_results_dir
        output_prefix = ""
    else:  # train_apply_all
        train_parquet = results_dir / combined_filename
        apply_all_parquet = apply_all_results_dir / combined_filename
        apply_subset_parquet = apply_all_results_dir / training_subset_filename
        apply_dir = apply_all_results_dir
        output_prefix = "all_data_"
    
    # Step 1: Create train parquet if needed
    if should_run_step(train_parquet, results_dir):
        create_combined_results(results_dir, train_parquet)
    else:
        print(f"Train parquet up to date: {train_parquet}")
    
    # Step 2: Create apply parquet if needed
    if mode == "train_apply":
        if should_run_step(apply_parquet, apply_dir):
            create_combined_results(apply_dir, apply_parquet)
        else:
            print(f"Apply parquet up to date: {apply_parquet}")
    else:  # train_apply_all
        # Create apply_all parquet
        if apply_dir.exists():
            if should_run_step(apply_all_parquet, apply_dir):
                create_combined_results(apply_dir, apply_all_parquet)
            else:
                print(f"Apply all parquet up to date: {apply_all_parquet}")
        else:
            print(f"Apply all directory does not exist: {apply_dir}")
            return
        
        # Create training subset
        if apply_all_parquet.exists():
            if should_run_step(apply_subset_parquet, [apply_all_parquet]):
                filter_and_save_training_subset(apply_all_parquet, apply_subset_parquet)
            else:
                print(f"Training subset up to date: {apply_subset_parquet}")
        else:
            print("Apply all parquet not available for subset creation")
            return
        
        # Use subset for comparisons
        apply_parquet = apply_subset_parquet
    
    # Step 3: Create duration histogram
    duration_output = plots_dir / f"{output_prefix}{duration_histogram_filename}"
    if train_parquet.exists() and apply_parquet.exists():
        if should_run_step(duration_output, [train_parquet, apply_parquet]):
            duration_histogram(train_parquet, apply_parquet, duration_output)
        else:
            print(f"Duration histogram up to date: {duration_output}")
    else:
        print("Skipping duration histogram - required parquets not available")
    
    # Step 4: Create confusion matrices
    confusion_outputs = [
        (plots_dir / f"{output_prefix}{confusion_matrix_filename}", "syllable"),
        (plots_dir / f"{output_prefix}{confusion_matrix_smoothed_filename}", "syllable_smoothed"),
        (plots_dir / f"{output_prefix}{confusion_matrix_top50_filename}", "syllable", 50),
        (plots_dir / f"{output_prefix}{confusion_matrix_smoothed_top50_filename}", "syllable_smoothed", 50),
    ]
    
    for output_file, syllable_col, *args in confusion_outputs:
        if train_parquet.exists() and apply_parquet.exists():
            if should_run_step(output_file, [train_parquet, apply_parquet]):
                syllable_confusion_matrix(train_parquet, apply_parquet, output_file, syllable_col, *args)
            else:
                print(f"Confusion matrix up to date: {output_file}")
        else:
            print(f"Skipping confusion matrix - required parquets not available: {output_file}")
    
    # Step 5: Create syllable counts scatter
    scatter_output = plots_dir / f"{output_prefix}{syllable_counts_scatter_filename}"
    if train_parquet.exists() and apply_parquet.exists():
        if should_run_step(scatter_output, [train_parquet, apply_parquet]):
            syllable_counts_scatter(train_parquet, apply_parquet, scatter_output)
        else:
            print(f"Syllable counts scatter up to date: {scatter_output}")
    else:
        print("Skipping syllable counts scatter - required parquets not available")
    
    print(f"Completed {mode} comparisons")

run_comparisons("train_apply")  # For results vs apply_results
run_comparisons("train_apply_all")  # For results vs apply_all_results (training subset)