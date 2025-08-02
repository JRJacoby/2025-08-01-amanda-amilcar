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
    filter_and_save_training_subset
)

# Configuration
project_dir = Path("kpms_project")
model_name = "2025_08_01-23_41_19"
results_dir = project_dir / model_name / "results"
apply_results_dir = project_dir / model_name / "apply_results"
apply_all_results_dir = project_dir / model_name / "apply_all_results"
extended_plots_dir = project_dir / model_name / "extended_comparison_plots"
extended_analysis_dir = project_dir / model_name / "extended_analysis"
html_report_path = extended_analysis_dir / "comparison_report.html"

# File naming
combined_filename = "combined_results.parquet"
training_subset_filename = "combined_results_training_subset.parquet"
extended_results_json = "extended_analysis_results.json"

class ExtendedAnalysis:
    """Comprehensive analysis class for keypoint-moseq results comparison."""
    
    def __init__(self):
        self.results = {}
        self.plots = {}
        self.statistics = {}
        
    def load_data(self, train_parquet: Path, apply_parquet: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load and validate training and apply datasets."""
        print("Loading datasets...")
        train_df = pl.read_parquet(train_parquet)
        apply_df = pl.read_parquet(apply_parquet)
        
        # Basic data validation
        train_sessions = set(train_df["session_name"].unique())
        apply_sessions = set(apply_df["session_name"].unique())
        
        self.statistics["data_overview"] = {
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
        
        return train_df, apply_df
    
    def behavioral_sequence_analysis(self, train_df: pl.DataFrame, apply_df: pl.DataFrame):
        """Analyze behavioral sequence properties and transitions."""
        print("Performing behavioral sequence analysis...")
        
        def analyze_transitions(df, label):
            """Analyze syllable transitions within a dataset."""
            # Calculate transition probabilities
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
            
            # Calculate entropy of transition matrix
            total_transitions = len(transitions)
            transition_probs = [count / total_transitions for count in transition_counts.values()]
            entropy = -sum(p * np.log2(p) for p in transition_probs if p > 0)
            
            return {
                "total_transitions": total_transitions,
                "unique_transitions": len(transition_counts),
                "transition_entropy": entropy,
                "top_transitions": sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        
        train_transitions = analyze_transitions(train_df, "train")
        apply_transitions = analyze_transitions(apply_df, "apply")
        
        self.statistics["behavioral_sequences"] = {
            "train": train_transitions,
            "apply": apply_transitions
        }
        
        # Syllable bout analysis
        def analyze_bouts(df, label):
            """Analyze syllable bout characteristics."""
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
        
        train_bouts = analyze_bouts(train_df, "train")
        apply_bouts = analyze_bouts(apply_df, "apply")
        
        self.statistics["bout_analysis"] = {
            "train": train_bouts,
            "apply": apply_bouts
        }
    
    def statistical_comparisons(self, train_df: pl.DataFrame, apply_df: pl.DataFrame):
        """Perform comprehensive statistical comparisons."""
        print("Performing statistical comparisons...")
        
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
            return
        
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
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
        
        # Agreement metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Clustering metrics (treating syllables as cluster assignments)
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
        
        # Spearman correlation
        if len(train_counts_aligned) > 1:
            spearman_corr, spearman_p = stats.spearmanr(train_counts_aligned, apply_counts_aligned)
            pearson_corr, pearson_p = stats.pearsonr(train_counts_aligned, apply_counts_aligned)
        else:
            spearman_corr = spearman_p = pearson_corr = pearson_p = np.nan
        
        # Kolmogorov-Smirnov test for duration distributions
        train_durations = train_df.group_by("instance_id").len()["len"].to_numpy()
        apply_durations = apply_df.group_by("instance_id").len()["len"].to_numpy()
        
        ks_stat, ks_p = stats.ks_2samp(train_durations, apply_durations)
        
        self.statistics["statistical_tests"] = {
            "frame_accuracy": accuracy,
            "adjusted_rand_index": ari_score,
            "normalized_mutual_info": nmi_score,
            "spearman_correlation": {"statistic": spearman_corr, "p_value": spearman_p},
            "pearson_correlation": {"statistic": pearson_corr, "p_value": pearson_p},
            "ks_test_durations": {"statistic": ks_stat, "p_value": ks_p},
            "classification_report": class_report,
            "total_aligned_frames": len(aligned)
        }
    
    def latent_space_analysis(self, train_df: pl.DataFrame, apply_df: pl.DataFrame):
        """Analyze latent space representations."""
        print("Analyzing latent space representations...")
        
        # Extract latent features
        latent_cols = ["latent_state 0", "latent_state 1", "latent_state 2", "latent_state 3"]
        
        train_latent = train_df.select(latent_cols + ["syllable"]).to_pandas()
        apply_latent = apply_df.select(latent_cols + ["syllable"]).to_pandas()
        
        # PCA on latent features
        combined_latent = pd.concat([
            train_latent.assign(dataset="train"),
            apply_latent.assign(dataset="apply")
        ])
        
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(combined_latent[latent_cols])
        
        # t-SNE on latent features (subsample for efficiency)
        max_samples = 5000
        if len(combined_latent) > max_samples:
            sample_idx = np.random.choice(len(combined_latent), max_samples, replace=False)
            tsne_data = combined_latent.iloc[sample_idx]
        else:
            tsne_data = combined_latent
            sample_idx = np.arange(len(combined_latent))
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)//4))
        latent_tsne = tsne.fit_transform(tsne_data[latent_cols])
        
        self.statistics["latent_space"] = {
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "total_samples": len(combined_latent),
            "tsne_samples": len(tsne_data)
        }
        
        # Store data for plotting
        self.results["latent_pca"] = {
            "coordinates": latent_pca,
            "labels": combined_latent["syllable"].values,
            "dataset": combined_latent["dataset"].values
        }
        
        self.results["latent_tsne"] = {
            "coordinates": latent_tsne,
            "labels": tsne_data["syllable"].values,
            "dataset": tsne_data["dataset"].values,
            "sample_idx": sample_idx
        }
    
    def generate_advanced_plots(self):
        """Generate advanced visualization plots."""
        print("Generating advanced plots...")
        extended_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Latent space PCA plot
        if "latent_pca" in self.results:
            self._plot_latent_space_pca()
        
        # 2. Latent space t-SNE plot
        if "latent_tsne" in self.results:
            self._plot_latent_space_tsne()
        
        # 3. Transition heatmaps
        self._plot_transition_heatmaps()
        
        # 4. Syllable repertoire comparison
        self._plot_syllable_repertoire()
        
        # 5. Session-wise comparison
        self._plot_session_comparison()
        
    def _plot_latent_space_pca(self):
        """Plot PCA of latent space colored by syllable and dataset."""
        pca_data = self.results["latent_pca"]
        
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
        plt.savefig(extended_plots_dir / "latent_space_pca.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.plots["latent_space_pca"] = "latent_space_pca.png"
    
    def _plot_latent_space_tsne(self):
        """Plot t-SNE of latent space."""
        tsne_data = self.results["latent_tsne"]
        
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
        plt.savefig(extended_plots_dir / "latent_space_tsne.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.plots["latent_space_tsne"] = "latent_space_tsne.png"
    
    def _plot_transition_heatmaps(self):
        """Plot syllable transition probability heatmaps."""
        if "behavioral_sequences" not in self.statistics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        for idx, (dataset, data) in enumerate([
            ("Training", self.statistics["behavioral_sequences"]["train"]),
            ("Apply", self.statistics["behavioral_sequences"]["apply"])
        ]):
            # Create transition matrix
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
        plt.savefig(extended_plots_dir / "transition_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.plots["transition_heatmaps"] = "transition_heatmaps.png"
    
    def _plot_syllable_repertoire(self):
        """Plot syllable repertoire comparison."""
        if "statistical_tests" not in self.statistics:
            return
        
        # Extract syllable frequencies from classification report
        class_report = self.statistics["statistical_tests"]["classification_report"]
        
        syllables = []
        train_support = []
        apply_support = []
        precision = []
        recall = []
        f1_score = []
        
        for syll, metrics in class_report.items():
            if syll not in ["accuracy", "macro avg", "weighted avg"]:
                syllables.append(syll)
                # Note: support in classification report is for predicted labels
                # We'll use this as a proxy for syllable frequency
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
        plt.savefig(extended_plots_dir / "syllable_repertoire.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.plots["syllable_repertoire"] = "syllable_repertoire.png"
    
    def _plot_session_comparison(self):
        """Plot session-wise comparison metrics."""
        # This would require session-specific analysis
        # For now, create a placeholder plot showing data overview
        
        data_overview = self.statistics["data_overview"]
        
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
        if "statistical_tests" in self.statistics:
            stats_data = self.statistics["statistical_tests"]
            metrics = ["Frame Accuracy", "Adjusted Rand Index", "Normalized Mutual Info"]
            values = [
                stats_data["frame_accuracy"],
                stats_data["adjusted_rand_index"],
                stats_data["normalized_mutual_info"]
            ]
            
            ax2.bar(metrics, values, alpha=0.7, color="green")
            ax2.set_ylabel("Score")
            ax2.set_title("Agreement Metrics")
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        
        # Correlation metrics
        if "statistical_tests" in self.statistics:
            stats_data = self.statistics["statistical_tests"]
            correlations = ["Spearman", "Pearson"]
            corr_values = [
                stats_data["spearman_correlation"]["statistic"],
                stats_data["pearson_correlation"]["statistic"]
            ]
            p_values = [
                stats_data["spearman_correlation"]["p_value"],
                stats_data["pearson_correlation"]["p_value"]
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
        if "bout_analysis" in self.statistics:
            bout_data = self.statistics["bout_analysis"]
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
        plt.savefig(extended_plots_dir / "session_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.plots["session_comparison"] = "session_comparison.png"
    
    def generate_interactive_plots(self):
        """Generate interactive plots using Plotly."""
        print("Generating interactive plots...")
        
        # Interactive confusion matrix
        if "statistical_tests" in self.statistics:
            self._create_interactive_confusion_matrix()
        
        # Interactive latent space plot
        if "latent_pca" in self.results:
            self._create_interactive_latent_space()
    
    def _create_interactive_confusion_matrix(self):
        """Create interactive confusion matrix with Plotly."""
        class_report = self.statistics["statistical_tests"]["classification_report"]
        
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
        fig.write_html(extended_plots_dir / "interactive_confusion_matrix.html")
        self.plots["interactive_confusion_matrix"] = "interactive_confusion_matrix.html"
    
    def _create_interactive_latent_space(self):
        """Create interactive latent space plot with Plotly."""
        pca_data = self.results["latent_pca"]
        
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
        fig.write_html(extended_plots_dir / "interactive_latent_space.html")
        self.plots["interactive_latent_space"] = "interactive_latent_space.html"
    
    def save_results(self):
        """Save analysis results to JSON."""
        print("Saving analysis results...")
        extended_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.statistics,
            "plots": self.plots,
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
        
        with open(extended_analysis_dir / extended_results_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to {extended_analysis_dir / extended_results_json}")

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string for HTML embedding."""
    if not image_path.exists():
        return ""
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def generate_html_report(analysis: ExtendedAnalysis):
    """Generate comprehensive HTML report."""
    print("Generating HTML report...")
    
    # HTML template
    html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keypoint-MoSeq Extended Comparison Report</title>
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
            <h1>🧬 Keypoint-MoSeq Extended Comparison Report</h1>
            <p class="timestamp">Generated on {{ timestamp }}</p>
            <p><strong>Model:</strong> {{ model_name }} | <strong>Project:</strong> {{ project_dir }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Frame-level Accuracy:</strong> {{ "%.1f"|format(statistics.statistical_tests.frame_accuracy * 100) }}% agreement between training and apply predictions</li>
                    <li><strong>Behavioral Consistency:</strong> {{ "%.3f"|format(statistics.statistical_tests.adjusted_rand_index) }} Adjusted Rand Index indicates {{ 
                        "excellent" if statistics.statistical_tests.adjusted_rand_index > 0.8 else
                        "good" if statistics.statistical_tests.adjusted_rand_index > 0.6 else
                        "moderate" if statistics.statistical_tests.adjusted_rand_index > 0.4 else
                        "limited" }} clustering agreement</li>
                    <li><strong>Syllable Repertoire:</strong> Training dataset contains {{ statistics.data_overview.train_unique_syllables }} unique syllables, apply dataset contains {{ statistics.data_overview.apply_unique_syllables }}</li>
                    <li><strong>Data Scale:</strong> Analysis covers {{ "{:,}"|format(statistics.data_overview.train_total_frames + statistics.data_overview.apply_total_frames) }} total frames across {{ statistics.data_overview.train_sessions + statistics.data_overview.apply_sessions }} sessions</li>
                </ul>
            </div>
        </div>

        <!-- Data Overview -->
        <div class="section">
            <h2>📈 Data Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Training Dataset</h3>
                    <div class="metric-value">{{ "{:,}"|format(statistics.data_overview.train_total_frames) }}</div>
                    <p>Total frames</p>
                    <p><strong>Sessions:</strong> {{ statistics.data_overview.train_sessions }}</p>
                    <p><strong>Syllables:</strong> {{ statistics.data_overview.train_unique_syllables }}</p>
                    <p><strong>Instances:</strong> {{ "{:,}"|format(statistics.data_overview.train_instances) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Apply Dataset</h3>
                    <div class="metric-value">{{ "{:,}"|format(statistics.data_overview.apply_total_frames) }}</div>
                    <p>Total frames</p>
                    <p><strong>Sessions:</strong> {{ statistics.data_overview.apply_sessions }}</p>
                    <p><strong>Syllables:</strong> {{ statistics.data_overview.apply_unique_syllables }}</p>
                    <p><strong>Instances:</strong> {{ "{:,}"|format(statistics.data_overview.apply_instances) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Overlap Analysis</h3>
                    <div class="metric-value">{{ statistics.data_overview.overlapping_sessions }}</div>
                    <p>Overlapping sessions</p>
                    <p><strong>Aligned frames:</strong> {{ "{:,}"|format(statistics.statistical_tests.total_aligned_frames) }}</p>
                </div>
            </div>
        </div>

        <!-- Statistical Analysis -->
        <div class="section">
            <h2>🔬 Statistical Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Agreement Metrics</h3>
                    <p><strong>Frame Accuracy:</strong> {{ "%.3f"|format(statistics.statistical_tests.frame_accuracy) }}</p>
                    <p><strong>Adjusted Rand Index:</strong> {{ "%.3f"|format(statistics.statistical_tests.adjusted_rand_index) }}</p>
                    <p><strong>Normalized Mutual Information:</strong> {{ "%.3f"|format(statistics.statistical_tests.normalized_mutual_info) }}</p>
                </div>
                <div class="metric-card">
                    <h3>Correlation Analysis</h3>
                    <p><strong>Spearman Correlation:</strong> {{ "%.3f"|format(statistics.statistical_tests.spearman_correlation.statistic) }}</p>
                    <p><em>p-value: {{ "%.2e"|format(statistics.statistical_tests.spearman_correlation.p_value) }}</em></p>
                    <p><strong>Pearson Correlation:</strong> {{ "%.3f"|format(statistics.statistical_tests.pearson_correlation.statistic) }}</p>
                    <p><em>p-value: {{ "%.2e"|format(statistics.statistical_tests.pearson_correlation.p_value) }}</em></p>
                </div>
                <div class="metric-card">
                    <h3>Distribution Tests</h3>
                    <p><strong>K-S Test (Durations):</strong></p>
                    <p>Statistic: {{ "%.3f"|format(statistics.statistical_tests.ks_test_durations.statistic) }}</p>
                    <p>p-value: {{ "%.2e"|format(statistics.statistical_tests.ks_test_durations.p_value) }}</p>
                    <p><em>{{ "Significant difference" if statistics.statistical_tests.ks_test_durations.p_value < 0.05 else "No significant difference" }} in duration distributions</em></p>
                </div>
            </div>
        </div>

        <!-- Behavioral Analysis -->
        {% if statistics.behavioral_sequences %}
        <div class="section">
            <h2>🎭 Behavioral Sequence Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Training Sequences</h3>
                    <p><strong>Total Transitions:</strong> {{ "{:,}"|format(statistics.behavioral_sequences.train.total_transitions) }}</p>
                    <p><strong>Unique Transitions:</strong> {{ statistics.behavioral_sequences.train.unique_transitions }}</p>
                    <p><strong>Transition Entropy:</strong> {{ "%.2f"|format(statistics.behavioral_sequences.train.transition_entropy) }} bits</p>
                </div>
                <div class="metric-card">
                    <h3>Apply Sequences</h3>
                    <p><strong>Total Transitions:</strong> {{ "{:,}"|format(statistics.behavioral_sequences.apply.total_transitions) }}</p>
                    <p><strong>Unique Transitions:</strong> {{ statistics.behavioral_sequences.apply.unique_transitions }}</p>
                    <p><strong>Transition Entropy:</strong> {{ "%.2f"|format(statistics.behavioral_sequences.apply.transition_entropy) }} bits</p>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Bout Analysis -->
        {% if statistics.bout_analysis %}
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
                        <td>{{ "%.2f"|format(statistics.bout_analysis.train.mean_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(statistics.bout_analysis.apply.mean_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(statistics.bout_analysis.apply.mean_bout_duration - statistics.bout_analysis.train.mean_bout_duration) }} frames</td>
                    </tr>
                    <tr>
                        <td>Median Bout Duration</td>
                        <td>{{ "%.2f"|format(statistics.bout_analysis.train.median_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(statistics.bout_analysis.apply.median_bout_duration) }} frames</td>
                        <td>{{ "%.2f"|format(statistics.bout_analysis.apply.median_bout_duration - statistics.bout_analysis.train.median_bout_duration) }} frames</td>
                    </tr>
                    <tr>
                        <td>Max Bout Duration</td>
                        <td>{{ statistics.bout_analysis.train.max_bout_duration }} frames</td>
                        <td>{{ statistics.bout_analysis.apply.max_bout_duration }} frames</td>
                        <td>{{ statistics.bout_analysis.apply.max_bout_duration - statistics.bout_analysis.train.max_bout_duration }} frames</td>
                    </tr>
                    <tr>
                        <td>Total Bouts</td>
                        <td>{{ "{:,}"|format(statistics.bout_analysis.train.total_bouts) }}</td>
                        <td>{{ "{:,}"|format(statistics.bout_analysis.apply.total_bouts) }}</td>
                        <td>{{ "{:,}"|format(statistics.bout_analysis.apply.total_bouts - statistics.bout_analysis.train.total_bouts) }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Visualizations -->
        <div class="section">
            <h2>📊 Visualizations</h2>
            
            {% for plot_name, plot_file in plots.items() %}
            {% if plot_file.endswith('.png') %}
            <div class="plot-container">
                <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
                <img src="data:image/png;base64,{{ plot_images[plot_name] }}" alt="{{ plot_name }}">
                <div class="plot-caption">{{ plot_name.replace('_', ' ').title() }}</div>
            </div>
            {% endif %}
            {% endfor %}
        </div>

        <!-- Interactive Content -->
        {% if interactive_plots %}
        <div class="section">
            <h2>🎯 Interactive Visualizations</h2>
            <p>Click the links below to view interactive plots:</p>
            {% for plot_name, plot_file in plots.items() %}
            {% if plot_file.endswith('.html') %}
            <a href="{{ plot_file }}" class="interactive-link" target="_blank">{{ plot_name.replace('_', ' ').title() }}</a>
            {% endif %}
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
                {% if statistics.latent_space %}
                <p><strong>PCA Explained Variance:</strong> {{ "%.1f"|format(statistics.latent_space.pca_explained_variance[0] * 100) }}% (PC1), {{ "%.1f"|format(statistics.latent_space.pca_explained_variance[1] * 100) }}% (PC2)</p>
                <p><strong>t-SNE Samples:</strong> {{ "{:,}"|format(statistics.latent_space.tsne_samples) }} of {{ "{:,}"|format(statistics.latent_space.total_samples) }} total</p>
                {% endif %}
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d;">
            <p>Generated by Keypoint-MoSeq Extended Analysis Pipeline</p>
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
        "statistics": analysis.statistics,
        "plots": analysis.plots,
        "plot_images": {},
        "interactive_plots": any(plot.endswith('.html') for plot in analysis.plots.values())
    }
    
    # Encode images to base64
    for plot_name, plot_file in analysis.plots.items():
        if plot_file.endswith('.png'):
            image_path = extended_plots_dir / plot_file
            template_data["plot_images"][plot_name] = encode_image_to_base64(image_path)
    
    # Render template
    html_content = html_template.render(**template_data)
    
    # Save HTML report
    html_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {html_report_path}")
    return html_report_path

def run_extended_analysis(mode: str = "train_apply_all"):
    """
    Run comprehensive extended analysis and generate HTML report.
    
    Args:
        mode: Either "train_apply" or "train_apply_all"
    """
    print(f"🚀 Starting extended analysis in {mode} mode...")
    
    # Define paths based on mode
    if mode == "train_apply":
        train_parquet = results_dir / combined_filename
        apply_parquet = apply_results_dir / combined_filename
        apply_dir = apply_results_dir
    else:  # train_apply_all
        train_parquet = results_dir / combined_filename
        apply_all_parquet = apply_all_results_dir / combined_filename
        apply_subset_parquet = apply_all_results_dir / training_subset_filename
        apply_dir = apply_all_results_dir
    
    # Step 1: Ensure base parquet files exist
    print("📁 Preparing data files...")
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
            print(f"❌ Apply directory does not exist: {apply_dir}")
            return
    else:
        if should_run_step(apply_parquet, apply_dir):
            create_combined_results(apply_dir, apply_parquet)
    
    # Verify required files exist
    if not train_parquet.exists() or not apply_parquet.exists():
        print("❌ Required parquet files not found. Please run basic comparison first.")
        return
    
    # Step 2: Initialize analysis
    analysis = ExtendedAnalysis()
    
    # Step 3: Load and analyze data
    train_df, apply_df = analysis.load_data(train_parquet, apply_parquet)
    
    # Step 4: Run comprehensive analyses
    print("🔬 Running statistical comparisons...")
    analysis.statistical_comparisons(train_df, apply_df)
    
    print("🎭 Analyzing behavioral sequences...")
    analysis.behavioral_sequence_analysis(train_df, apply_df)
    
    print("🧠 Analyzing latent space...")
    analysis.latent_space_analysis(train_df, apply_df)
    
    # Step 5: Generate visualizations
    print("📊 Generating static plots...")
    analysis.generate_advanced_plots()
    
    print("🎯 Generating interactive plots...")
    analysis.generate_interactive_plots()
    
    # Step 6: Save results
    analysis.save_results()
    
    # Step 7: Generate HTML report
    print("📄 Generating HTML report...")
    report_path = generate_html_report(analysis)
    
    print("✅ Extended analysis complete!")
    print(f"📊 Results saved to: {extended_analysis_dir}")
    print(f"📄 HTML report: {report_path}")
    print(f"🖼️  Plots saved to: {extended_plots_dir}")
    
    return analysis, report_path

if __name__ == "__main__":
    # Run extended analysis
    analysis, report_path = run_extended_analysis("train_apply_all")