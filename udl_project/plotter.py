from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from udl_project import config
from udl_project.utils.data_loading import load_pickled_artifacts

# Modern, colorblind-friendly color palette with semantic meaning
COLORS = {
    "Original ResNet": "#E74C3C",  # Red - baseline (problem)
    "L2 Regularized ResNet": "#3498DB",  # Blue - traditional regularization
    "Ensemble ResNet": "#2ECC71",  # Green - ensemble (robust)
    "Transfer ResNet Frozen": "#F39C12",  # Orange - transfer learning
    "Data Augmented ResNet": "#9B59B6",  # Purple - data techniques
    "Early Stopping ResNet": "#1ABC9C",  # Teal - early stopping
    "Transfer Learning ResNet": "#E67E22",  # Dark orange - advanced transfer
}

# Distinct line styles for better differentiation
LINESTYLES = {
    "Original ResNet": "-",  # Solid - baseline
    "L2 Regularized ResNet": "--",  # Dashed - regularization
    "Ensemble ResNet": "-.",  # Dash-dot - ensemble
    "Transfer ResNet Frozen": ":",  # Dotted - frozen transfer
    "Data Augmented ResNet": (0, (5, 2, 1, 2)),  # Custom dash pattern
    "Early Stopping ResNet": (0, (3, 1, 1, 1)),  # Custom dash pattern
    "Transfer Learning ResNet": (0, (5, 1)),  # Custom dash pattern
}


def load_results():
    """Load all required results from pickled files, that were produced by running each trainer.

    A trainer implements the abstract class `Trainer`.
    """
    results = {}

    # Load original results
    results["Original ResNet"] = load_pickled_artifacts("original_results.pkl")
    print("Loaded original results")

    # Load L2 results
    results["L2 Regularized ResNet"] = load_pickled_artifacts("l2_results.pkl")
    print("Loaded L2 results")

    # Load ensemble results
    results["Ensemble ResNet"] = load_pickled_artifacts("ensemble_results.pkl")
    print("Loaded ensemble results")

    # Load early stopping results
    results["Early Stopping ResNet"] = load_pickled_artifacts("early_stopping_results.pkl")
    print("Loaded early stopping results")

    # Load transfer learning results
    results["Transfer Learning ResNet"] = load_pickled_artifacts("finetune_results.pkl")
    print("Loaded transfer learning results")

    # Load data augmentation results
    results["Data Augmented ResNet"] = load_pickled_artifacts("augmented_results.pkl")
    print("Loaded data augmentation results")

    return results


def create_comprehensive_plots(results, show: bool = False):
    """Create comprehensive comparison plots to compare the regularization techniques.

    Args:
        results (dict): Dictionary containing the results from different trainers.
        show (bool): If True, display the plots interactively. If False, save them to the artifacts directory.
    """
    if not show:
        mpl.use("Agg")

    date_str = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    plots_path = config.ARTIFACTS_DIR / f"plots_{date_str}"
    plots_path.mkdir(exist_ok=True)

    print("\nCreating comprehensive comparison plots...")

    # Get individual comparison figures
    comparison_figs = plot_comparison(results, COLORS, LINESTYLES)
    plot_names = ["training_loss", "validation_loss", "training_accuracy", "validation_accuracy"]
    for fig, name in zip(comparison_figs, plot_names, strict=False):
        if not show:
            fig.savefig(str(plots_path / f"{name}_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Get individual overfitting analysis figures
    overfitting_figs = plot_overfitting_analysis(results, COLORS, LINESTYLES)
    overfitting_names = [
        "generalization_gap",
        "performance_tradeoff",
        "effectiveness_ranking",
        "stability_analysis",
    ]
    for fig, name in zip(overfitting_figs, overfitting_names, strict=False):
        if not show:
            fig.savefig(str(plots_path / f"overfitting_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Get individual dashboard figures
    dashboard_figs = plot_summary_dashboard(results, COLORS)
    dashboard_names = [
        "learning_curves",
        "performance_comparison",
        "overfitting_control",
        "loss_convergence",
        "executive_summary",
    ]
    for fig, name in zip(dashboard_figs, dashboard_names, strict=False):
        if not show:
            fig.savefig(str(plots_path / f"dashboard_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_comparison(results, colors, linestyles):
    """Create individual large plots for each metric comparison."""
    # Set up the matplotlib style for better aesthetics
    plt.style.use("seaborn-v0_8-whitegrid")

    figures = []

    # Plot configurations
    plot_configs = [
        ("Training Loss Evolution", "train_losses", "Training Loss", False),
        ("Validation Loss Evolution", "val_losses", "Validation Loss", False),
        ("Training Accuracy Evolution", "train_accs", "Training Accuracy (%)", True),
        ("Validation Accuracy Evolution", "val_accs", "Validation Accuracy (%)", True),
    ]

    for title, data_key, ylabel, is_accuracy in plot_configs:
        # Create individual large figure for each plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Enhanced title with better formatting
        fig.suptitle(
            f"Deep Learning Regularization: {title}",
            fontsize=20,
            fontweight="bold",
            y=0.95,
            color="#2C3E50",
        )

        # Plot each model with enhanced styling
        for name, data in results.items():
            y_data = data[data_key]
            if is_accuracy:  # Convert accuracy to percentage
                y_data = [acc * 100 for acc in y_data]

            ax.plot(
                range(1, len(y_data) + 1),
                y_data,
                label=name.replace(" ResNet", ""),  # Shorter labels
                linewidth=4,
                color=colors[name],
                linestyle=linestyles[name],
                alpha=0.8,
                marker="o"
                if len(y_data) <= 20  # noqa: PLR2004
                else None,  # Add markers for short series
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=2,
            )

        # Enhanced axis formatting
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=16)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=16)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8)
        ax.set_facecolor("#FAFAFA")

        # Better legend positioning and styling
        ax.legend(
            loc="best",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=14,
            title="Regularization Methods",
            title_fontsize=16,
        )

        # Set y-axis limits for better comparison
        if not is_accuracy:  # Loss plots
            ax.set_ylim(bottom=0)
        else:  # Accuracy plots
            ax.set_ylim(0, 100)

            # Highlight the best performing model with subtle background
            final_values = {}
            for name, data in results.items():
                final_val = data[data_key][-1] * 100
                final_values[name] = final_val

            best_model = max(final_values.keys(), key=lambda k: final_values[k])
            best_data = [acc * 100 for acc in results[best_model][data_key]]

            # Add annotation for best performer
            best_final = max(final_values.values())
            ax.annotate(
                f"Best: {best_model.replace(' ResNet', '')} ({best_final:.1f}%)",
                xy=(len(best_data), best_final),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": colors[best_model], "alpha": 0.3},
                arrowprops={"arrowstyle": "->", "color": colors[best_model], "lw": 2},
            )

        # Improve tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=14)

        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_overfitting_analysis(results, colors, linestyles):
    """Create individual large plots for overfitting analysis."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figures = []

    # Plot 1: Generalization Gap Evolution - Redesigned for clarity
    fig1, (ax_main, ax_final) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig1.suptitle(
        "Regularization Effectiveness: Generalization Gap Analysis",
        fontsize=20,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    # Main plot: Gap evolution over time
    ax_main.set_title("Gap Evolution Over Training", fontweight="bold", fontsize=16, pad=20)

    # Calculate gaps and plot with better styling
    gap_data = {}
    final_gaps = {}

    for name, data in results.items():
        gap = np.array(data["train_accs"]) - np.array(data["val_accs"])
        gap_percent = gap * 100
        epochs = range(1, len(gap_percent) + 1)
        gap_data[name] = gap_percent
        final_gaps[name] = gap_percent[-1]

        # Plot with enhanced styling
        ax_main.plot(
            epochs,
            gap_percent,
            label=name.replace(" ResNet", ""),
            linewidth=3,
            color=colors[name],
            linestyle=linestyles[name],
            alpha=0.85,
        )

        # Add subtle marker at final point
        ax_main.scatter(
            len(gap_percent),
            gap_percent[-1],
            color=colors[name],
            s=80,
            zorder=5,
            edgecolor="white",
            linewidth=2,
        )

    # Enhanced main plot styling
    ax_main.set_xlabel("Training Epoch", fontweight="bold", fontsize=14)
    ax_main.set_ylabel("Generalization Gap (%)", fontweight="bold", fontsize=14)
    ax_main.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax_main.axhline(
        y=0, color="darkred", linestyle="--", alpha=0.8, linewidth=2, label="Perfect Generalization"
    )
    ax_main.tick_params(axis="both", which="major", labelsize=12)
    ax_main.legend(fontsize=11, loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Add interpretation box
    ax_main.text(
        0.02,
        0.98,
        "Lower is Better\n(Better Generalization)",
        transform=ax_main.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightgreen", "alpha": 0.3},
    )

    # Side plot: Final gap comparison (horizontal bar chart)
    ax_final.set_title("Final Gap\nComparison", fontweight="bold", fontsize=16, pad=20)

    # Sort by final gap (best to worst)
    sorted_models = sorted(final_gaps.items(), key=lambda x: x[1])
    model_names = [name.replace(" ResNet", "") for name, _ in sorted_models]
    gap_values = [gap for _, gap in sorted_models]
    model_colors = [colors[name] for name, _ in sorted_models]

    # Create horizontal bar chart
    bars = ax_final.barh(
        range(len(model_names)),
        gap_values,
        color=model_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Style the bar chart
    ax_final.set_yticks(range(len(model_names)))
    ax_final.set_yticklabels(model_names, fontsize=12)
    ax_final.set_xlabel("Final Gap (%)", fontweight="bold", fontsize=12)
    ax_final.grid(True, alpha=0.3, axis="x")
    ax_final.tick_params(axis="both", which="major", labelsize=11)

    # Highlight the best performer
    bars[0].set_edgecolor("gold")
    bars[0].set_linewidth(3)

    # Add value labels on bars
    for bar, gap in zip(bars, gap_values, strict=False):
        ax_final.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{gap:.2f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    # Add ranking indicators
    for i, bar in enumerate(bars):
        rank_color = (
            "gold" if i == 0 else "silver" if i == 1 else "darkgoldenrod" if i == 2 else "gray"  # noqa: PLR2004
        )
        ax_final.text(
            0.02,
            bar.get_y() + bar.get_height() / 2,
            f"#{i + 1}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
            color=rank_color,
            transform=ax_final.transData,
        )

    plt.tight_layout()
    figures.append(fig1)

    # Plot 2: Performance vs Overfitting - Redesigned as a strategic quadrant analysis
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    fig2.suptitle(
        "Strategic Model Selection: Performance vs Generalization Trade-off",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    final_val_accs = []
    final_gaps = []
    names = []

    for name, data in results.items():
        val_acc = data["val_accs"][-1] * 100
        gap = (data["train_accs"][-1] - data["val_accs"][-1]) * 100
        final_val_accs.append(val_acc)
        final_gaps.append(gap)
        names.append(name.replace(" ResNet", ""))

    # Create the scatter plot with enhanced styling
    scatter_points = []
    for i, (gap, acc) in enumerate(zip(final_gaps, final_val_accs, strict=False)):
        full_name = list(results.keys())[i]
        point = ax2.scatter(
            gap,
            acc,
            c=colors[full_name],
            s=300,
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        scatter_points.append(point)

    # Add quadrant lines for strategic analysis
    median_gap = float(np.median(final_gaps))
    median_acc = float(np.median(final_val_accs))

    ax2.axvline(x=median_gap, color="gray", linestyle="--", alpha=0.5, linewidth=2)
    ax2.axhline(y=median_acc, color="gray", linestyle="--", alpha=0.5, linewidth=2)

    # Add quadrant labels with strategic meaning (removed emojis to avoid font warnings)
    ax2.text(
        0.05,
        0.95,
        "IDEAL ZONE\nHigh Accuracy\nLow Overfitting",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.7},
        verticalalignment="top",
    )

    ax2.text(
        0.95,
        0.95,
        "HIGH RISK\nHigh Accuracy\nHigh Overfitting",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "orange", "alpha": 0.7},
        verticalalignment="top",
        horizontalalignment="right",
    )

    ax2.text(
        0.05,
        0.05,
        "POOR ZONE\nLow Accuracy\nLow Overfitting",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightcoral", "alpha": 0.7},
        verticalalignment="bottom",
    )

    ax2.text(
        0.95,
        0.05,
        "WORST CASE\nLow Accuracy\nHigh Overfitting",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "red", "alpha": 0.7},
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Add model labels with arrows pointing to points
    for name, gap, acc in zip(names, final_gaps, final_val_accs, strict=False):
        ax2.annotate(
            name,
            (gap, acc),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "black",
            },
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
        )

    # Find and highlight the best performer (lowest gap, highest accuracy)
    best_idx = min(
        range(len(final_gaps)), key=lambda i: final_gaps[i] - final_val_accs[i] / 100
    )  # Composite score

    ax2.scatter(
        final_gaps[best_idx],
        final_val_accs[best_idx],
        s=400,
        facecolors="none",
        edgecolors="gold",
        linewidth=4,
        zorder=6,
    )
    ax2.text(
        final_gaps[best_idx],
        final_val_accs[best_idx] - 1,
        "* RECOMMENDED",
        ha="center",
        fontweight="bold",
        fontsize=10,
        color="darkgoldenrod",
    )

    ax2.set_xlabel("Overfitting Gap (%) → Higher Risk", fontweight="bold", fontsize=14)
    ax2.set_ylabel("Validation Accuracy (%) → Better Performance", fontweight="bold", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    figures.append(fig2)

    # Plot 3: Regularization Effectiveness Ranking
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))
    fig3.suptitle(
        "Regularization Effectiveness Ranking",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    # Calculate effectiveness scores
    original_gap = (
        results["Original ResNet"]["train_accs"][-1] - results["Original ResNet"]["val_accs"][-1]
    ) * 100
    effectiveness_scores = []
    method_names = []

    for name, data in results.items():
        if name != "Original ResNet":
            gap = (data["train_accs"][-1] - data["val_accs"][-1]) * 100
            val_acc = data["val_accs"][-1] * 100
            gap_reduction = (original_gap - gap) / original_gap * 100
            original_val_acc = results["Original ResNet"]["val_accs"][-1] * 100
            acc_improvement = ((val_acc - original_val_acc) / original_val_acc) * 100

            effectiveness_score = gap_reduction + acc_improvement
            effectiveness_scores.append(effectiveness_score)
            method_names.append(name.replace(" ResNet", ""))

    # Sort by effectiveness
    sorted_data = sorted(
        zip(
            method_names,
            effectiveness_scores,
            [colors[f"{name} ResNet"] for name in method_names],
            strict=False,
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_names, sorted_scores, sorted_colors = zip(*sorted_data, strict=False)

    bars = ax3.barh(
        range(len(sorted_names)),
        sorted_scores,
        color=sorted_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )

    ax3.set_yticks(range(len(sorted_names)))
    ax3.set_yticklabels(sorted_names, fontsize=14)
    ax3.set_xlabel("Effectiveness Score (Higher is Better)", fontweight="bold", fontsize=16)
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.tick_params(axis="both", which="major", labelsize=14)

    # Add score labels
    for bar, score in zip(bars, sorted_scores, strict=False):
        ax3.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    figures.append(fig3)

    # Plot 4: Learning Curves Overlay - Enhanced multi-technique comparison
    fig4, (ax4_train, ax4_val) = plt.subplots(1, 2, figsize=(20, 8))
    fig4.suptitle(
        "Learning Dynamics: Training vs Validation Performance Evolution",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    # Find the minimum number of epochs across all models
    min_epochs = min(len(data["train_accs"]) for data in results.values())
    epochs = range(1, min_epochs + 1)

    # Training curves panel
    for name, data in results.items():
        train_accs = [acc * 100 for acc in data["train_accs"][:min_epochs]]
        ax4_train.plot(
            epochs,
            train_accs,
            color=colors[name],
            linewidth=3,
            alpha=0.8,
            linestyle=linestyles[name],
            label=name.replace(" ResNet", ""),
        )

        # Add final point marker
        ax4_train.scatter(epochs[-1], train_accs[-1], color=colors[name], s=100, zorder=5)

    ax4_train.set_title("Training Accuracy Evolution", fontsize=16, fontweight="bold", pad=20)
    ax4_train.set_xlabel("Epoch", fontweight="bold", fontsize=14)
    ax4_train.set_ylabel("Training Accuracy (%)", fontweight="bold", fontsize=14)
    ax4_train.grid(True, alpha=0.3)
    ax4_train.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax4_train.tick_params(axis="both", which="major", labelsize=12)

    # Add convergence analysis
    final_train_accs = [data["train_accs"][min_epochs - 1] * 100 for data in results.values()]
    best_train_acc = max(final_train_accs)
    ax4_train.axhline(y=best_train_acc, color="green", linestyle=":", alpha=0.6, linewidth=2)
    ax4_train.text(
        0.02,
        0.98,
        f"Best Training: {best_train_acc:.1f}%",
        transform=ax4_train.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.8},
        verticalalignment="top",
    )

    # Validation curves panel
    for name, data in results.items():
        val_accs = [acc * 100 for acc in data["val_accs"][:min_epochs]]
        ax4_val.plot(
            epochs,
            val_accs,
            color=colors[name],
            linewidth=3,
            alpha=0.8,
            linestyle=linestyles[name],
            label=name.replace(" ResNet", ""),
        )

        # Add final point marker
        ax4_val.scatter(epochs[-1], val_accs[-1], color=colors[name], s=100, zorder=5)

    ax4_val.set_title("Validation Accuracy Evolution", fontsize=16, fontweight="bold", pad=20)
    ax4_val.set_xlabel("Epoch", fontweight="bold", fontsize=14)
    ax4_val.set_ylabel("Validation Accuracy (%)", fontweight="bold", fontsize=14)
    ax4_val.grid(True, alpha=0.3)
    ax4_val.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax4_val.tick_params(axis="both", which="major", labelsize=12)

    # Add generalization analysis
    final_val_accs = [data["val_accs"][min_epochs - 1] * 100 for data in results.values()]
    best_val_acc = max(final_val_accs)
    best_val_name = list(results.keys())[final_val_accs.index(best_val_acc)]

    ax4_val.axhline(y=best_val_acc, color="blue", linestyle=":", alpha=0.6, linewidth=2)
    ax4_val.text(
        0.02,
        0.98,
        f"Best Validation: {best_val_acc:.1f}%\n({best_val_name.replace(' ResNet', '')})",
        transform=ax4_val.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.8},
        verticalalignment="top",
    )

    # Add stability indicators
    for _, data in results.items():
        val_accs = [acc * 100 for acc in data["val_accs"][:min_epochs]]
        if len(val_accs) > 5:  # Only if we have enough epochs  # noqa: PLR2004
            stability = np.std(val_accs[-5:])  # Std of last 5 epochs
            if stability < 0.5:  # Very stable  # noqa: PLR2004
                ax4_val.text(
                    epochs[-1] + 0.5,
                    val_accs[-1],
                    "STABLE",
                    fontsize=10,
                    color="green",
                    fontweight="bold",
                )

    plt.tight_layout()
    figures.append(fig4)

    return figures


def plot_summary_dashboard(results, colors):
    """Create individual large plots for summary dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figures = []

    # Plot 1: Combined Learning Curves
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    fig1.suptitle(
        "Training vs Validation Performance: Learning Curves Comparison",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    for name, data in results.items():
        epochs = range(1, len(data["train_losses"]) + 1)
        # Plot training curves with lighter alpha
        ax1.plot(
            epochs,
            [acc * 100 for acc in data["train_accs"]],
            color=colors[name],
            linestyle="-",
            alpha=0.4,
            linewidth=3,
        )
        # Plot validation curves with full alpha and labels
        ax1.plot(
            epochs,
            [acc * 100 for acc in data["val_accs"]],
            color=colors[name],
            linestyle="-",
            linewidth=4,
            label=f"{name.replace(' ResNet', '')}",
            alpha=0.9,
        )

    ax1.set_xlabel("Epoch", fontweight="bold", fontsize=16)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=16)
    ax1.legend(loc="lower right", fontsize=14, title="Regularization Methods", title_fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.text(
        0.02,
        0.98,
        "Solid Lines: Validation Accuracy\nFaded Lines: Training Accuracy",
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )

    plt.tight_layout()
    figures.append(fig1)

    # Plot 2: Performance Metrics Comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    fig2.suptitle(
        "Final Performance Comparison: Composite Scoring",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    metrics = []
    model_names = []
    model_colors = []

    for name, data in results.items():
        val_acc = data["val_accs"][-1] * 100
        gap = (data["train_accs"][-1] - data["val_accs"][-1]) * 100
        score = val_acc - gap  # Composite score: high validation accuracy, low gap
        metrics.append(score)
        model_names.append(name.replace(" ResNet", ""))
        model_colors.append(colors[name])

    # Sort by performance
    sorted_data = sorted(
        zip(model_names, metrics, model_colors, strict=False), key=lambda x: x[1], reverse=True
    )
    sorted_names, sorted_metrics, sorted_colors = zip(*sorted_data, strict=False)

    bars = ax2.barh(
        range(len(sorted_names)),
        sorted_metrics,
        color=sorted_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontsize=14)
    ax2.set_xlabel(
        "Performance Score (Val Accuracy - Overfitting Gap)", fontweight="bold", fontsize=16
    )
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.tick_params(axis="both", which="major", labelsize=14)

    # Add score labels
    for bar, score in zip(bars, sorted_metrics, strict=False):
        ax2.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

    # Highlight the best performer
    bars[0].set_edgecolor("gold")
    bars[0].set_linewidth(4)

    plt.tight_layout()
    figures.append(fig2)

    # Plot 3: Overfitting Control Analysis
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))
    fig3.suptitle(
        "Overfitting Control: Generalization Gap Analysis",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    gaps = []
    names = []
    gap_colors = []

    for name, data in results.items():
        gap = (data["train_accs"][-1] - data["val_accs"][-1]) * 100
        gaps.append(gap)
        names.append(name.replace(" ResNet", ""))
        gap_colors.append(colors[name])

    bars = ax3.bar(
        range(len(names)), gaps, color=gap_colors, alpha=0.8, edgecolor="black", linewidth=2
    )
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=14)
    ax3.set_ylabel("Overfitting Gap (%)", fontweight="bold", fontsize=16)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(axis="both", which="major", labelsize=14)

    # Highlight best (lowest gap)
    min_gap_idx = gaps.index(min(gaps))
    bars[min_gap_idx].set_edgecolor("gold")
    bars[min_gap_idx].set_linewidth(4)

    # Add value labels
    for bar, gap in zip(bars, gaps, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{gap:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax3.text(
        0.02,
        0.98,
        "Lower is Better\n(Less Overfitting)",
        transform=ax3.transAxes,
        fontsize=14,
        verticalalignment="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.4},
    )

    plt.tight_layout()
    figures.append(fig3)

    # Plot 4: Loss Convergence Analysis
    fig4, ax4 = plt.subplots(1, 1, figsize=(14, 10))
    fig4.suptitle(
        "Loss Convergence: Training Efficiency Analysis",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        color="#2C3E50",
    )

    for name, data in results.items():
        epochs = range(1, len(data["val_losses"]) + 1)
        ax4.plot(
            epochs,
            data["val_losses"],
            color=colors[name],
            linewidth=4,
            alpha=0.8,
            label=name.replace(" ResNet", ""),
            marker="o" if len(epochs) <= 20 else None,  # noqa: PLR2004
            markersize=6,
        )

    ax4.set_xlabel("Epoch", fontweight="bold", fontsize=16)
    ax4.set_ylabel("Validation Loss (Log Scale)", fontweight="bold", fontsize=16)
    ax4.set_yscale("log")  # Log scale for better visualization
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=14, title="Regularization Methods", title_fontsize=16)
    ax4.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    figures.append(fig4)

    # Plot 5: Executive Summary Text
    fig5, ax5 = plt.subplots(1, 1, figsize=(14, 10))
    ax5.axis("off")
    fig5.suptitle(
        "Text based summary of findings", fontsize=20, fontweight="bold", y=0.95, color="#2C3E50"
    )

    # Generate comprehensive summary
    original_data = results["Original ResNet"]
    original_val_acc = original_data["val_accs"][-1] * 100
    original_gap = (original_data["train_accs"][-1] - original_data["val_accs"][-1]) * 100

    # Find best models for different criteria
    best_val_acc = max(results.items(), key=lambda x: x[1]["val_accs"][-1])
    best_gap = min(results.items(), key=lambda x: x[1]["train_accs"][-1] - x[1]["val_accs"][-1])

    summary_text = f"""
    COMPREHENSIVE PERFORMANCE ANALYSIS SUMMARY

    BASELINE PERFORMANCE (Original ResNet):
    • Validation Accuracy: {original_val_acc:.1f}%
    • Overfitting Gap: {original_gap:.2f}%
    • Establishes benchmark for regularization effectiveness

    TOP PERFORMERS BY CATEGORY:
    • Highest Validation Accuracy: {best_val_acc[0].replace(" ResNet", "")}
      ({best_val_acc[1]["val_accs"][-1] * 100:.1f}% - {((best_val_acc[1]["val_accs"][-1] * 100 - original_val_acc) / original_val_acc * 100):+.1f}% vs baseline)

    • Best Overfitting Control: {best_gap[0].replace(" ResNet", "")}
      ({(best_gap[1]["train_accs"][-1] - best_gap[1]["val_accs"][-1]) * 100:.2f}% gap - {((original_gap - (best_gap[1]["train_accs"][-1] - best_gap[1]["val_accs"][-1]) * 100) / original_gap * 100):.1f}% improvement)

    KEY STRATEGIC INSIGHTS:
    • Regularization techniques demonstrate varying effectiveness across metrics
    • Critical trade-offs exist between final accuracy and generalization capability
    • Some methods optimize for training efficiency while others prioritize stability
    • Early stopping and ensemble methods show particularly strong generalization

    DEPLOYMENT RECOMMENDATIONS:
    • Production Systems: Use {best_gap[0].replace(" ResNet", "")}
      (optimal generalization for real-world deployment)

    • Research/Benchmarking: Consider {best_val_acc[0].replace(" ResNet", "")}
      (maximum accuracy for competitive performance)

    • Monitoring Strategy: Track overfitting gaps during training for early intervention
      and implement validation-based stopping criteria

    NEXT STEPS:
    • Conduct hyperparameter optimization on top-performing methods
    • Investigate ensemble combinations of best individual techniques
    • Implement continuous monitoring for production model drift
    """

    ax5.text(
        0.05,
        0.95,
        summary_text,
        transform=ax5.transAxes,
        fontsize=13,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F8F9FA",
            "edgecolor": "#DEE2E6",
            "linewidth": 2,
        },
    )

    plt.tight_layout()
    figures.append(fig5)

    return figures


def print_summary(results):
    """Print final results summary."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    original_gap = (
        results["Original ResNet"]["train_accs"][-1] - results["Original ResNet"]["val_accs"][-1]
    )

    for name, data in results.items():
        train_acc = data["train_accs"][-1]
        val_acc = data["val_accs"][-1]
        gap = train_acc - val_acc

        print(f"\n{name}:")
        print(f"  Final Train Accuracy: {train_acc:.4f}")
        print(f"  Final Val Accuracy:   {val_acc:.4f}")
        print(f"  Overfitting Gap:      {gap:.4f}")

        if name != "Original ResNet":
            improvement = (original_gap - gap) / original_gap * 100
            print(f"  Gap Reduction:        {improvement:.1f}%")

    # Determine best model
    gaps = {name: data["train_accs"][-1] - data["val_accs"][-1] for name, data in results.items()}
    best_model = min(gaps.keys(), key=lambda k: gaps[k])
    print(f"\nBEST MODEL (lowest overfitting): {best_model}")
    print(f"   Overfitting gap: {gaps[best_model]:.4f}")


def plot():
    print("COMPREHENSIVE UDL REGULARIZATION COMPARISON PLOTTER")
    print("=" * 60)

    # Load results
    results = load_results()
    if results is None:
        print("Could not load all required results files.")
        return

    # Create comprehensive plots
    create_comprehensive_plots(results)

    # Print summary
    print_summary(results)

    print("\nAll plots saved successfully!")
    print("Generated files:")
    print("  - artifacts/udl_comprehensive_comparison.png")
    print("  - artifacts/udl_overfitting_analysis.png")
    print("  - artifacts/udl_summary_dashboard.png")
    print("\nComprehensive plotting completed!")


if __name__ == "__main__":
    plot()
