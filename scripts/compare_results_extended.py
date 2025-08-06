import polars as pl
import pandas as pd
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn.objects as so
import seaborn as sns
import numpy as np
from typing import Literal, Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Import from existing comparison script
from compare_results import (
    extract_session_name, 
    smooth_syllables_with_window, 
    create_combined_results,
    should_run_step,
    get_training_session_names,
    filter_and_save_training_subset,
    ComparisonMode
)

# Configuration
project_dir = Path("kpms_project")
model_name = "2025_08_01-23_41_19"
results_dir = project_dir / model_name / "results"
apply_results_dir = project_dir / model_name / "apply_results"
apply_all_results_dir = project_dir / model_name / "apply_all_results"
comparison_plots_dir = project_dir / model_name / "comparison_plots"
analysis_dir = project_dir / model_name / "analysis"
html_report_path = analysis_dir / "comparison_report.html"

# File naming
combined_filename = "combined_results.parquet"
training_subset_filename = "combined_results_training_subset.parquet"
analysis_results_json = "analysis_results.json"

# Plot filenames
latent_pca_filename = "latent_space_pca.png"
latent_tsne_filename = "latent_space_tsne.png"
transition_heatmaps_filename = "transition_heatmaps.png"
syllable_repertoire_filename = "syllable_repertoire.png"
session_comparison_filename = "session_comparison.png"
interactive_confusion_filename = "interactive_confusion_matrix.html"
interactive_latent_filename = "interactive_latent_space.html"


def calculate_statistical_metrics(train_df: pl.DataFrame, apply_df: pl.DataFrame) -> Dict:
    """Calculate comprehensive statistical comparison metrics."""
    print("Calculating statistical metrics...")
    
    # Align datasets for frame-by-frame comparison
    train_aligned = train_df.sort(["session_name", "frame"]).select([
        "session_name", "frame", "syllable", "syllable_smoothed"
    ])
    apply_aligned = apply_df.sort(["session_name", "frame"]).select([
        "session_name", "frame", "syllable", "syllable_smoothed"
    ])
    
    aligned = train_aligned.join(
        apply_aligned, 
        on=["session_name", "frame"], 
        how="inner",
        suffix="_apply"
    )
    
    if len(aligned) == 0:
        print("Warning: No aligned frames found between datasets")
        return {}
    
    # Classification metrics
    y_true = aligned["syllable"].to_numpy()
    y_pred = aligned["syllable_apply"].to_numpy()
    
    # Get unique labels from both datasets
    all_labels = sorted(set(np.concatenate([y_true, y_pred])))
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, 
        labels=all_labels,
        output_dict=True,
        zero_division=0
    )
    
    # Agreement metrics
    accuracy = np.mean(y_true == y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    
    # Statistical tests on syllable distributions
    train_syll_counts = train_df.group_by("syllable").len().sort("syllable")
    apply_syll_counts = apply_df.group_by("syllable").len().sort("syllable")
    
    # Align syllable counts
    all_syllables = sorted(set(train_syll_counts["syllable"].to_list() + 
                             apply_syll_counts["syllable"].to_list()))
    
    train_counts_aligned = []
    apply_counts_aligned = []
    
    for syll in all_syllables:
        train_count = train_syll_counts.filter(pl.col("syllable") == syll)
        apply_count = apply_syll_counts.filter(pl.col("syllable") == syll)
        
        train_counts_aligned.append(train_count["len"][0] if len(train_count) > 0 else 0)
        apply_counts_aligned.append(apply_count["len"][0] if len(apply_count) > 0 else 0)
    
    # Correlation tests
    if len(train_counts_aligned) > 1:
        spearman_corr, spearman_p = stats.spearmanr(train_counts_aligned, apply_counts_aligned)
        pearson_corr, pearson_p = stats.pearsonr(train_counts_aligned, apply_counts_aligned)
    else:
        spearman_corr = spearman_p = pearson_corr = pearson_p = np.nan
    
    # Distribution tests
    train_durations = train_df.group_by("instance_id").len()["len"].to_numpy()
    apply_durations = apply_df.group_by("instance_id").len()["len"].to_numpy()
    
    ks_stat, ks_p = stats.ks_2samp(train_durations, apply_durations)
    
    return {
        "frame_accuracy": accuracy,
        "adjusted_rand_index": ari_score,
        "normalized_mutual_info": nmi_score,
        "spearman_correlation": {"statistic": spearman_corr, "p_value": spearman_p},
        "pearson_correlation": {"statistic": pearson_corr, "p_value": pearson_p},
        "ks_test_durations": {"statistic": ks_stat, "p_value": ks_p},
        "classification_report": class_report,
        "total_aligned_frames": len(aligned)
    }


def analyze_behavioral_sequences(train_df: pl.DataFrame, apply_df: pl.DataFrame) -> Dict:
    """Analyze behavioral sequence properties and transitions."""
    print("Analyzing behavioral sequences...")
    
    def calculate_transitions(df, label):
        """Calculate syllable transitions within a dataset."""
        transitions = []
        for session in df["session_name"].unique():
            session_data = df.filter(pl.col("session_name") == session).sort("frame")
            syllables = session_data["syllable"].to_list()
            
            for i in range(len(syllables) - 1):
                transitions.append((syllables[i], syllables[i + 1]))
        
        # Count transitions
        transition_counts = {}
        for from_syll, to_syll in transitions:
            key = f"{from_syll}→{to_syll}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Calculate entropy
        total_transitions = len(transitions)
        transition_probs = [count / total_transitions for count in transition_counts.values()]
        entropy = -sum(p * np.log2(p) for p in transition_probs if p > 0)
        
        return {
            "total_transitions": total_transitions,
            "unique_transitions": len(transition_counts),
            "transition_entropy": entropy,
            "top_transitions": sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    train_transitions = calculate_transitions(train_df, "train")
    apply_transitions = calculate_transitions(apply_df, "apply")
    
    # Bout analysis
    def calculate_bout_stats(df, label):
        """Calculate syllable bout characteristics."""
        bout_stats = df.group_by(["session_name", "instance_id", "syllable"]).agg([
            pl.len().alias("duration"),
            pl.col("frame").min().alias("start_frame"),
            pl.col("frame").max().alias("end_frame")
        ])
        
        return {
            "mean_bout_duration": bout_stats["duration"].mean(),
            "median_bout_duration": bout_stats["duration"].median(),
            "bout_duration_std": bout_stats["duration"].std(),
            "max_bout_duration": bout_stats["duration"].max(),
            "total_bouts": len(bout_stats)
        }
    
    train_bouts = calculate_bout_stats(train_df, "train")
    apply_bouts = calculate_bout_stats(apply_df, "apply")
    
    return {
        "train": train_transitions,
        "apply": apply_transitions,
        "bout_analysis": {
            "train": train_bouts,
            "apply": apply_bouts
        }
    }


def analyze_latent_space(train_df: pl.DataFrame, apply_df: pl.DataFrame) -> Dict:
    """Analyze latent space representations."""
    print("Analyzing latent space representations...")
    
    # Extract latent features
    latent_cols = ["latent_state 0", "latent_state 1", "latent_state 2", "latent_state 3"]
    
    train_latent = train_df.select(latent_cols + ["syllable"]).to_pandas()
    apply_latent = apply_df.select(latent_cols + ["syllable"]).to_pandas()
    
    # Combine datasets
    combined_latent = pd.concat([
        train_latent.assign(dataset="train"),
        apply_latent.assign(dataset="apply")
    ])
    
    # PCA analysis
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(combined_latent[latent_cols])
    
    # t-SNE analysis (subsample for efficiency)
    max_samples = 5000
    if len(combined_latent) > max_samples:
        sample_idx = np.random.choice(len(combined_latent), max_samples, replace=False)
        tsne_data = combined_latent.iloc[sample_idx]
    else:
        tsne_data = combined_latent
        sample_idx = np.arange(len(combined_latent))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)//4))
    latent_tsne = tsne.fit_transform(tsne_data[latent_cols])
    
    return {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "total_samples": len(combined_latent),
        "tsne_samples": len(tsne_data),
        "pca_data": {
            "coordinates": latent_pca,
            "labels": combined_latent["syllable"].values,
            "dataset": combined_latent["dataset"].values
        },
        "tsne_data": {
            "coordinates": latent_tsne,
            "labels": tsne_data["syllable"].values,
            "dataset": tsne_data["dataset"].values,
            "sample_idx": sample_idx
        }
    }


def plot_latent_space_pca(latent_results: Dict, output_file: Path) -> None:
    """Plot PCA of latent space colored by syllable and dataset."""
    pca_data = latent_results["pca_data"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot by dataset
    for dataset in ["train", "apply"]:
        mask = pca_data["dataset"] == dataset
        ax1.scatter(
            pca_data["coordinates"][mask, 0],
            pca_data["coordinates"][mask, 1],
            alpha=0.6, s=1, label=dataset
        )
    
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Latent Space PCA - by Dataset")
    ax1.legend()
    
    # Plot by syllable (top 10 most frequent)
    syllable_counts = pd.Series(pca_data["labels"]).value_counts()
    top_syllables = syllable_counts.head(10).index
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_syllables)))
    for i, syll in enumerate(top_syllables):
        mask = pca_data["labels"] == syll
        ax2.scatter(
            pca_data["coordinates"][mask, 0],
            pca_data["coordinates"][mask, 1],
            alpha=0.6, s=1, color=colors[i], label=f"Syllable {syll}"
        )
    
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Latent Space PCA - by Top 10 Syllables")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latent_space_tsne(latent_results: Dict, output_file: Path) -> None:
    """Plot t-SNE of latent space."""
    tsne_data = latent_results["tsne_data"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot by dataset
    for dataset in ["train", "apply"]:
        mask = tsne_data["dataset"] == dataset
        ax1.scatter(
            tsne_data["coordinates"][mask, 0],
            tsne_data["coordinates"][mask, 1],
            alpha=0.6, s=1, label=dataset
        )
    
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.set_title("Latent Space t-SNE - by Dataset")
    ax1.legend()
    
    # Plot by syllable (top 10 most frequent)
    syllable_counts = pd.Series(tsne_data["labels"]).value_counts()
    top_syllables = syllable_counts.head(10).index
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_syllables)))
    for i, syll in enumerate(top_syllables):
        mask = tsne_data["labels"] == syll
        ax2.scatter(
            tsne_data["coordinates"][mask, 0],
            tsne_data["coordinates"][mask, 1],
            alpha=0.6, s=1, color=colors[i], label=f"Syllable {syll}"
        )
    
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("Latent Space t-SNE - by Top 10 Syllables")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_transition_heatmaps(behavioral_results: Dict, output_file: Path) -> None:
    """Plot syllable transition probability heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (dataset, data) in enumerate([
        ("Training", behavioral_results["train"]),
        ("Apply", behavioral_results["apply"])
    ]):
        transitions = data["top_transitions"]
        
        # Extract unique syllables
        syllables = set()
        for trans, _ in transitions:
            from_syll, to_syll = trans.split("→")
            syllables.add(from_syll)
            syllables.add(to_syll)
        
        syllables = sorted(list(syllables))
        n_syll = len(syllables)
        
        if n_syll == 0:
            continue
            
        # Create matrix
        trans_matrix = np.zeros((n_syll, n_syll))
        syll_to_idx = {syll: i for i, syll in enumerate(syllables)}
        
        total_count = sum(count for _, count in transitions)
        for trans, count in transitions:
            from_syll, to_syll = trans.split("→")
            i, j = syll_to_idx[from_syll], syll_to_idx[to_syll]
            trans_matrix[i, j] = count / total_count
        
        # Plot heatmap
        ax = ax1 if idx == 0 else ax2
        im = ax.imshow(trans_matrix, cmap="Blues", aspect="auto")
        
        # Set ticks
        ax.set_xticks(range(n_syll))
        ax.set_yticks(range(n_syll))
        ax.set_xticklabels(syllables, rotation=45)
        ax.set_yticklabels(syllables)
        
        ax.set_title(f"{dataset} Transition Probabilities")
        ax.set_xlabel("To Syllable")
        ax.set_ylabel("From Syllable")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_syllable_repertoire(statistical_results: Dict, output_file: Path) -> None:
    """Plot syllable repertoire comparison."""
    class_report = statistical_results["classification_report"]
    
    syllables = []
    precision = []
    recall = []
    f1_score = []
    
    for syll, metrics in class_report.items():
        if syll not in ["accuracy", "macro avg", "weighted avg"]:
            syllables.append(syll)
            precision.append(metrics.get("precision", 0))
            recall.append(metrics.get("recall", 0))
            f1_score.append(metrics.get("f1-score", 0))
    
    if not syllables:
        return
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision by syllable
    ax1.bar(range(len(syllables)), precision, alpha=0.7, color="skyblue")
    ax1.set_xlabel("Syllable")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision by Syllable")
    ax1.set_xticks(range(len(syllables)))
    ax1.set_xticklabels(syllables, rotation=45)
    
    # Recall by syllable
    ax2.bar(range(len(syllables)), recall, alpha=0.7, color="lightcoral")
    ax2.set_xlabel("Syllable")
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall by Syllable")
    ax2.set_xticks(range(len(syllables)))
    ax2.set_xticklabels(syllables, rotation=45)
    
    # F1-score by syllable
    ax3.bar(range(len(syllables)), f1_score, alpha=0.7, color="lightgreen")
    ax3.set_xlabel("Syllable")
    ax3.set_ylabel("F1-Score")
    ax3.set_title("F1-Score by Syllable")
    ax3.set_xticks(range(len(syllables)))
    ax3.set_xticklabels(syllables, rotation=45)
    
    # Precision vs Recall scatter
    ax4.scatter(precision, recall, alpha=0.7, s=50)
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax4.set_xlabel("Precision")
    ax4.set_ylabel("Recall")
    ax4.set_title("Precision vs Recall")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_session_comparison(data_overview: Dict, statistical_results: Dict, behavioral_results: Dict, output_file: Path) -> None:
    """Plot session-wise comparison metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Data overview bars
    categories = ["Total Frames", "Sessions", "Unique Syllables", "Instances"]
    train_values = [
        data_overview["train_total_frames"],
        data_overview["train_sessions"],
        data_overview["train_unique_syllables"],
        data_overview["train_instances"]
    ]
    apply_values = [
        data_overview["apply_total_frames"],
        data_overview["apply_sessions"],
        data_overview["apply_unique_syllables"],
        data_overview["apply_instances"]
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, train_values, width, label="Training", alpha=0.7)
    ax1.bar(x + width/2, apply_values, width, label="Apply", alpha=0.7)
    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Count")
    ax1.set_title("Dataset Overview Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.set_yscale("log")
    
    # Statistical metrics
    metrics = ["Frame Accuracy", "Adjusted Rand Index", "Normalized Mutual Info"]
    values = [
        statistical_results["frame_accuracy"],
        statistical_results["adjusted_rand_index"],
        statistical_results["normalized_mutual_info"]
    ]
    
    ax2.bar(metrics, values, alpha=0.7, color="green")
    ax2.set_ylabel("Score")
    ax2.set_title("Agreement Metrics")
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    
    # Correlation metrics
    correlations = ["Spearman", "Pearson"]
    corr_values = [
        statistical_results["spearman_correlation"]["statistic"],
        statistical_results["pearson_correlation"]["statistic"]
    ]
    p_values = [
        statistical_results["spearman_correlation"]["p_value"],
        statistical_results["pearson_correlation"]["p_value"]
    ]
    
    bars = ax3.bar(correlations, corr_values, alpha=0.7, color="orange")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_title("Syllable Count Correlations")
    ax3.set_ylim(-1, 1)
    
    # Add significance indicators
    for i, (v, p) in enumerate(zip(corr_values, p_values)):
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax3.text(i, v + 0.05, f"{v:.3f}{significance}", ha="center", va="bottom")
    
    # Bout duration comparison
    bout_data = behavioral_results["bout_analysis"]
    bout_metrics = ["Mean Duration", "Median Duration", "Max Duration"]
    train_bout_values = [
        bout_data["train"]["mean_bout_duration"],
        bout_data["train"]["median_bout_duration"],
        bout_data["train"]["max_bout_duration"]
    ]
    apply_bout_values = [
        bout_data["apply"]["mean_bout_duration"],
        bout_data["apply"]["median_bout_duration"],
        bout_data["apply"]["max_bout_duration"]
    ]
    
    x = np.arange(len(bout_metrics))
    width = 0.35
    
    ax4.bar(x - width/2, train_bout_values, width, label="Training", alpha=0.7)
    ax4.bar(x + width/2, apply_bout_values, width, label="Apply", alpha=0.7)
    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Duration (frames)")
    ax4.set_title("Bout Duration Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(bout_metrics)
    ax4.legend()
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def create_interactive_confusion_matrix(statistical_results: Dict, output_file: Path) -> None:
    """Create interactive confusion matrix with Plotly."""
    class_report = statistical_results["classification_report"]
    
    syllables = [syll for syll in class_report.keys() 
                if syll not in ["accuracy", "macro avg", "weighted avg"]]
    
    if not syllables:
        return
    
    # Create confusion matrix data
    precision = [class_report[syll]["precision"] for syll in syllables]
    recall = [class_report[syll]["recall"] for syll in syllables]
    f1 = [class_report[syll]["f1-score"] for syll in syllables]
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=precision,
        y=recall,
        mode="markers+text",
        text=syllables,
        textposition="top center",
        marker=dict(
            size=10,
            color=f1,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="F1-Score")
        ),
        hovertemplate="<b>Syllable %{text}</b><br>" +
                     "Precision: %{x:.3f}<br>" +
                     "Recall: %{y:.3f}<br>" +
                     "F1-Score: %{marker.color:.3f}<extra></extra>"
    ))
    
    # Add diagonal line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="red", dash="dash")
    )
    
    fig.update_layout(
        title="Interactive Syllable Performance Metrics",
        xaxis_title="Precision",
        yaxis_title="Recall",
        width=800,
        height=600
    )
    
    # Save as HTML
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)


def create_interactive_latent_space(latent_results: Dict, output_file: Path) -> None:
    """Create interactive latent space plot with Plotly."""
    pca_data = latent_results["pca_data"]
    
    # Subsample for performance
    max_points = 10000
    if len(pca_data["coordinates"]) > max_points:
        indices = np.random.choice(len(pca_data["coordinates"]), max_points, replace=False)
        coords = pca_data["coordinates"][indices]
        labels = pca_data["labels"][indices]
        datasets = pca_data["dataset"][indices]
    else:
        coords = pca_data["coordinates"]
        labels = pca_data["labels"]
        datasets = pca_data["dataset"]
    
    # Create DataFrame for easier handling
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Syllable": labels,
        "Dataset": datasets
    })
    
    # Create interactive plot
    fig = px.scatter(
        plot_df, 
        x="PC1", y="PC2", 
        color="Syllable", 
        symbol="Dataset",
        title="Interactive Latent Space PCA",
        hover_data=["Syllable", "Dataset"],
        opacity=0.6
    )
    
    fig.update_layout(width=1000, height=700)
    
    # Save as HTML
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string for HTML embedding."""
    if not image_path.exists():
        return ""
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def generate_html_report(results: Dict, output_file: Path) -> None:
    """Generate comprehensive HTML report."""
    print(f"Generating HTML report: {output_file}")
    
    # HTML template
    html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keypoint-MoSeq Comparison Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .timestamp {
            color: #7f8c8d;
            font-style: italic;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .metric-card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #27ae60;
        }
        .plot-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .plot-caption {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .stats-table th, .stats-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #bdc3c7;
        }
        .stats-table th {
            background-color: #3498db;
            color: white;
        }
        .stats-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .summary-box {
            background-color: #e8f5e8;
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .summary-box h3 {
            color: #27ae60;
            margin-top: 0;
        }
        .interactive-link {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin: 5px;
        }
        .interactive-link:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Keypoint-MoSeq Comparison Report</h1>
            <p class="timestamp">Generated on {{ timestamp }}</p>
            <p><strong>Model:</strong> {{ model_name }} | <strong>Project:</strong> {{ project_dir }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Frame-level Accuracy:</strong> {{ "%.1f"|format(results.statistical_tests.frame_accuracy * 100) }}% agreement between training and apply predictions</li>
                    <li><strong>Behavioral Consistency:</strong> {{ "%.3f"|format(results.statistical_tests.adjusted_rand_index) }} Adjusted Rand Index indicates {{ 
                        "excellent" if results.statistical_tests.adjusted_rand_index > 0.8 else
                        "good" if results.statistical_tests.adjusted_rand_index > 0.6 else
                        "moderate" if results.statistical_tests.adjusted_rand_index > 0.4 else
                        "limited" }} clustering agreement</li>
                    <li><strong>Syllable Repertoire:</strong> Training dataset contains {{ results.data_overview.train_unique_syllables }} unique syllables, apply dataset contains {{ results.data_overview.apply_unique_syllables }}</li>
                    <li><strong>Data Scale:</strong> Analysis covers {{ "{:,}"|format(results.data_overview.train_total_frames + results.data_overview.apply_total_frames) }} total frames across {{ results.data_overview.train_sessions + results.data_overview.apply_sessions }} sessions</li>
                </ul>
            </div>
        </div>

        <!-- Data Overview -->
        <div class="section">
            <h2>📈 Data Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Training Dataset</h3>
                    <div class="metric-value">{{ "{:,}"|format(results.data_overview.train_total_frames) }}</div>
                    <p>Total frames</p>
                    <p><strong>Sessions:</strong> {{ results.data_overview.train_sessions }}</p>
                    <p><strong>Syllables:</strong> {{ results.data_overview.train_unique_syllables }}</p>
                    <p><strong>Instances:</strong> {{ "{:,}"|format(results.data_overview.train_instances) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Apply Dataset</h3>
                    <div class="metric-value">{{ "{:,}"|format(results.data_overview.apply_total_frames) }}</div>
                    <p>Total frames</p>
                    <p><strong>Sessions:</strong> {{ results.data_overview.apply_sessions }}</p>
                    <p><strong>Syllables:</strong> {{ results.data_overview.apply_unique_syllables }}</p>
                    <p><strong>Instances:</strong> {{ "{:,}"|format(results.data_overview.apply_instances) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Overlap Analysis</h3>
                    <div class="metric-value">{{ results.data_overview.overlapping_sessions }}</div>
                    <p>Overlapping sessions</p>
                    <p><strong>Aligned frames:</strong> {{ "{:,}"|format(results.statistical_tests.total_aligned_frames) }}</p>
                </div>
            </div>
        </div>

        <!-- Statistical Analysis -->
        <div class="section">
            <h2>🔬 Statistical Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Agreement Metrics</h3>
                    <p><strong>Frame Accuracy:</strong> {{ "%.3f"|format(results.statistical_tests.frame_accuracy) }}</p>
                    <p><strong>Adjusted Rand Index:</strong> {{ "%.3f"|format(results.statistical_tests.adjusted_rand_index) }}</p>
                    <p><strong>Normalized Mutual Information:</strong> {{ "%.3f"|format(results.statistical_tests.normalized_mutual_info) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Correlation Analysis</h3>
                    <p><strong>Spearman Correlation:</strong> {{ "%.3f"|format(results.statistical_tests.spearman_correlation.statistic) }}</p>
                    <p><em>p-value: {{ "%.2e"|format(results.statistical_tests.spearman_correlation.p_value) }}</em></p>
                    <p><strong>Pearson Correlation:</strong> {{ "%.3f"|format(results.statistical_tests.pearson_correlation.statistic) }}</p>
                    <p><em>p-value: {{ "%.2e"|format(results.statistical_tests.pearson_correlation.p_value) }}</em></p>
                </div>
                <div class="metric-card">
                    <h3>Distribution Tests</h3>
                    <p><strong>K-S Test (Durations):</strong></p>
                    <p>Statistic: {{ "%.3f"|format(results.statistical_tests.ks_test_durations.statistic) }}</p>
                    <p>p-value: {{ "%.2e"|format(results.statistical_tests.ks_test_durations.p_value) }}</p>
                    <p><em>{{ "Significant difference" if results.statistical_tests.ks_test_durations.p_value < 0.05 else "No significant difference" }} in duration distributions</em></p>
                </div>
            </div>
        </div>

        <!-- Behavioral Analysis -->
        {% if results.behavioral_sequences %}
        <div class="section">
            <h2>🎭 Behavioral Sequence Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Training Sequences</h3>
                    <p><strong>Total Transitions:</strong> {{ "{:,}"|format(results.behavioral_sequences.train.total_transitions) }}</p>
                    <p><strong>Unique Transitions:</strong> {{ results.behavioral_sequences.train.unique_transitions }}</p>
                    <p><strong>Transition Entropy:</strong> {{ "%.2f"|format(results.behavioral_sequences.train.transition_entropy) }} bits</p>
                </div>
                <div class="metric-card">
                    <h3>Apply Sequences</h3>
                    <p><strong>Total Transitions:</strong> {{ "{:,}"|format(results.behavioral_sequences.apply.total_transitions) }}</p>
                    <p><strong>Unique Transitions:</strong> {{ results.behavioral_sequences.apply.unique_transitions }}</p>
                    <p><strong>Transition Entropy:</strong> {{ "%.2f"|format(results.behavioral_sequences.apply.transition_entropy) }} bits</p>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Bout Analysis -->
        {% if results.behavioral_sequences.bout_analysis %}
        <div class="section">
            <h2>⏱️ Behavioral Bout Analysis</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Training Dataset</th>
                        <th>Apply Dataset</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Mean Bout Duration</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.train.mean_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.apply.mean_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.apply.mean_bout_duration - results.behavioral_sequences.bout_analysis.train.mean_bout_duration) }} frames</td>
                    </tr>
                    <tr>
                        <td>Median Bout Duration</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.train.median_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.apply.median_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(results.behavioral_sequences.bout_analysis.apply.median_bout_duration - results.behavioral_sequences.bout_analysis.train.median_bout_duration) }} frames</td>
                    </tr>
                    <tr>
                        <td>Max Bout Duration</td>
                        <td>{{ results.behavioral_sequences.bout_analysis.train.max_bout_duration }} frames</td>
                        <td>{{ results.behavioral_sequences.bout_analysis.apply.max_bout_duration }} frames</td>
                        <td>{{ results.behavioral_sequences.bout_analysis.apply.max_bout_duration - results.behavioral_sequences.bout_analysis.train.max_bout_duration }} frames</td>
                    </tr>
                    <tr>
                        <td>Total Bouts</td>
                        <td>{{ "{:,}"|format(results.behavioral_sequences.bout_analysis.train.total_bouts) }}</td>
                        <td>{{ "{:,}"|format(results.behavioral_sequences.bout_analysis.apply.total_bouts) }}</td>
                        <td>{{ "{:,}"|format(results.behavioral_sequences.bout_analysis.apply.total_bouts - results.behavioral_sequences.bout_analysis.train.total_bouts) }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Visualizations -->
        <div class="section">
            <h2>📊 Visualizations</h2>
            
            {% for plot_name, plot_data in results.plots.items() %}
            <div class="plot-container">
                <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
                <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
                <div class="plot-caption">{{ plot_name.replace('_', ' ').title() }}</div>
            </div>
            {% endfor %}
        </div>

        <!-- Interactive Content -->
        {% if results.interactive_plots %}
        <div class="section">
            <h2>🎯 Interactive Visualizations</h2>
            <p>Click the links below to view interactive plots:</p>
            {% for plot_name in results.interactive_plots %}
            <a href="{{ plot_name }}" class="interactive-link" target="_blank">{{ plot_name.replace('_', ' ').title() }}</a>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Technical Details -->
        <div class="section">
            <h2>⚙️ Technical Details</h2>
            <div class="metric-card">
                <h3>Analysis Parameters</h3>
                <p><strong>Model Name:</strong> {{ model_name }}</p>
                <p><strong>Project Directory:</strong> {{ project_dir }}</p>
                <p><strong>Analysis Timestamp:</strong> {{ timestamp }}</p>
                {% if results.latent_space %}
                <p><strong>PCA Explained Variance:</strong> {{ "%.1f"|format(results.latent_space.pca_explained_variance[0] * 100) }}% (PC1), {{ "%.1f"|format(results.latent_space.pca_explained_variance[1] * 100) }}% (PC2)</p>
                <p><strong>t-SNE Samples:</strong> {{ "{:,}"|format(results.latent_space.tsne_samples) }} of {{ "{:,}"|format(results.latent_space.total_samples) }} total</p>
                {% endif %}
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d;">
            <p>Generated by Keypoint-MoSeq Analysis Pipeline</p>
        </div>
    </div>
</body>
</html>
    """)
    
    # Prepare data for template
    template_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "project_dir": str(project_dir),
        "results": results
    }
    
    # Render template
    html_content = html_template.render(**template_data)
    
    # Save HTML report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")


def save_analysis_results(results: Dict, output_file: Path) -> None:
    """Save analysis results to JSON."""
    print(f"Saving analysis results: {output_file}")
    
    # Prepare data for JSON serialization
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "metadata": {
            "model_name": model_name,
            "project_dir": str(project_dir),
            "analysis_version": "1.0"
        }
    }
    
    # Convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    json_data = convert_numpy(json_data)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)


def run_comprehensive_analysis(mode: ComparisonMode = "train_apply_all") -> Dict:
    """
    Run comprehensive analysis and generate HTML report.
    
    Args:
        mode: Either "train_apply" or "train_apply_all"
    """
    print(f"Starting comprehensive analysis in {mode} mode...")
    
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
    
    # Step 1: Ensure base parquet files exist
    print("Preparing data files...")
    if should_run_step(train_parquet, results_dir):
        create_combined_results(results_dir, train_parquet)
    
    if mode == "train_apply_all":
        if apply_dir.exists():
            if should_run_step(apply_all_parquet, apply_dir):
                create_combined_results(apply_dir, apply_all_parquet)
            
            if apply_all_parquet.exists():
                if should_run_step(apply_subset_parquet, [apply_all_parquet]):
                    filter_and_save_training_subset(apply_all_parquet, apply_subset_parquet)
            
            apply_parquet = apply_subset_parquet
        else:
            print(f"Apply directory does not exist: {apply_dir}")
            return {}
    else:
        if should_run_step(apply_parquet, apply_dir):
            create_combined_results(apply_dir, apply_parquet)
    
    # Verify required files exist
    if not train_parquet.exists() or not apply_parquet.exists():
        print("Required parquet files not found. Please run basic comparison first.")
        return {}
    
    # Step 2: Load data
    print("Loading datasets...")
    train_df = pl.read_parquet(train_parquet)
    apply_df = pl.read_parquet(apply_parquet)
    
    # Basic data validation
    train_sessions = set(train_df["session_name"].unique())
    apply_sessions = set(apply_df["session_name"].unique())
    
    data_overview = {
        "train_total_frames": len(train_df),
        "apply_total_frames": len(apply_df),
        "train_sessions": len(train_sessions),
        "apply_sessions": len(apply_sessions),
        "overlapping_sessions": len(train_sessions.intersection(apply_sessions)),
        "train_unique_syllables": train_df["syllable"].n_unique(),
        "apply_unique_syllables": apply_df["syllable"].n_unique(),
        "train_instances": train_df["instance_id"].n_unique(),
        "apply_instances": apply_df["instance_id"].n_unique()
    }
    
    print(f"Training data: {len(train_df):,} frames, {len(train_sessions)} sessions")
    print(f"Apply data: {len(apply_df):,} frames, {len(apply_sessions)} sessions")
    
    # Step 3: Run analyses
    statistical_results = calculate_statistical_metrics(train_df, apply_df)
    behavioral_results = analyze_behavioral_sequences(train_df, apply_df)
    latent_results = analyze_latent_space(train_df, apply_df)
    
    # Step 4: Generate plots
    print("Generating plots...")
    plots = {}
    
    # Static plots
    if latent_results:
        latent_pca_file = comparison_plots_dir / f"{output_prefix}{latent_pca_filename}"
        if should_run_step(latent_pca_file, [train_parquet, apply_parquet]):
            plot_latent_space_pca(latent_results, latent_pca_file)
        plots["latent_space_pca"] = encode_image_to_base64(latent_pca_file)
        
        latent_tsne_file = comparison_plots_dir / f"{output_prefix}{latent_tsne_filename}"
        if should_run_step(latent_tsne_file, [train_parquet, apply_parquet]):
            plot_latent_space_tsne(latent_results, latent_tsne_file)
        plots["latent_space_tsne"] = encode_image_to_base64(latent_tsne_file)
    
    if behavioral_results:
        transition_file = comparison_plots_dir / f"{output_prefix}{transition_heatmaps_filename}"
        if should_run_step(transition_file, [train_parquet, apply_parquet]):
            plot_transition_heatmaps(behavioral_results, transition_file)
        plots["transition_heatmaps"] = encode_image_to_base64(transition_file)
    
    if statistical_results:
        repertoire_file = comparison_plots_dir / f"{output_prefix}{syllable_repertoire_filename}"
        if should_run_step(repertoire_file, [train_parquet, apply_parquet]):
            plot_syllable_repertoire(statistical_results, repertoire_file)
        plots["syllable_repertoire"] = encode_image_to_base64(repertoire_file)
    
    session_file = comparison_plots_dir / f"{output_prefix}{session_comparison_filename}"
    if should_run_step(session_file, [train_parquet, apply_parquet]):
        plot_session_comparison(data_overview, statistical_results, behavioral_results, session_file)
    plots["session_comparison"] = encode_image_to_base64(session_file)
    
    # Interactive plots
    interactive_plots = []
    if statistical_results:
        interactive_conf_file = comparison_plots_dir / f"{output_prefix}{interactive_confusion_filename}"
        if should_run_step(interactive_conf_file, [train_parquet, apply_parquet]):
            create_interactive_confusion_matrix(statistical_results, interactive_conf_file)
        interactive_plots.append(interactive_conf_file.name)
    
    if latent_results:
        interactive_latent_file = comparison_plots_dir / f"{output_prefix}{interactive_latent_filename}"
        if should_run_step(interactive_latent_file, [train_parquet, apply_parquet]):
            create_interactive_latent_space(latent_results, interactive_latent_file)
        interactive_plots.append(interactive_latent_file.name)
    
    # Step 5: Compile results
    results = {
        "data_overview": data_overview,
        "statistical_tests": statistical_results,
        "behavioral_sequences": behavioral_results,
        "latent_space": latent_results,
        "plots": plots,
        "interactive_plots": interactive_plots
    }
    
    # Step 6: Save results and generate report
    results_file = analysis_dir / f"{output_prefix}{analysis_results_json}"
    save_analysis_results(results, results_file)
    
    report_file = analysis_dir / f"{output_prefix}comparison_report.html"
    generate_html_report(results, report_file)
    
    print("Analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print(f"HTML report: {report_file}")
    print(f"Plots saved to: {comparison_plots_dir}")
    
    return results


if __name__ == "__main__":
    # Run comprehensive analysis
    results = run_comprehensive_analysis("train_apply_all")