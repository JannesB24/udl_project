from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from udl_project import config
from udl_project.utils.data_loading import load_pickled_artifacts

COLORS = {
    "Original ResNet": "red",
    "L2 Regularized ResNet": "blue",
    "Ensemble ResNet": "green",
    "Data Augmented ResNet": "purple",
}
LINESTYLES = {
    "Original ResNet": "-",
    "L2 Regularized ResNet": "--",
    "Ensemble ResNet": "-.",
    "Data Augmented ResNet": ":",
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

    fig1 = plot_comparison(results, COLORS, LINESTYLES)
    if not show:
        fig1.savefig(plots_path / "udl_comprehensive_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_overfitting_analysis(results, COLORS, LINESTYLES)
    if not show:
        fig2.savefig(plots_path / "udl_overfitting_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plot_summary_dashboard(results, COLORS)
    if not show:
        fig3.savefig(plots_path / "udl_summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)


def plot_comparison(results, colors, linestyles):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "UDL Project - Comprehensive Regularization Comparison", fontsize=16, fontweight="bold"
    )

    # Plot 1: Training Loss
    ax_train_loss = axes[0, 0]
    ax_train_loss.set_title("Training Loss Comparison", fontweight="bold", fontsize=14)
    for name, data in results.items():
        ax_train_loss.plot(
            data["train_losses"],
            label=name,
            linewidth=2.5,
            color=colors[name],
            linestyle=linestyles[name],
        )
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.legend()
    ax_train_loss.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax_val_loss = axes[0, 1]
    ax_val_loss.set_title("Validation Loss Comparison", fontweight="bold", fontsize=14)
    for name, data in results.items():
        ax_val_loss.plot(
            data["val_losses"],
            label=name,
            linewidth=2.5,
            color=colors[name],
            linestyle=linestyles[name],
        )
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Loss")
    ax_val_loss.legend()
    ax_val_loss.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    ax_train_acc = axes[1, 0]
    ax_train_acc.set_title("Training Accuracy Comparison", fontweight="bold", fontsize=14)
    for name, data in results.items():
        ax_train_acc.plot(
            data["train_accs"],
            label=name,
            linewidth=2.5,
            color=colors[name],
            linestyle=linestyles[name],
        )
    ax_train_acc.set_xlabel("Epoch")
    ax_train_acc.set_ylabel("Accuracy")
    ax_train_acc.legend()
    ax_train_acc.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    ax_val_acc = axes[1, 1]
    ax_val_acc.set_title("Validation Accuracy Comparison", fontweight="bold", fontsize=14)
    for name, data in results.items():
        ax_val_acc.plot(
            data["val_accs"],
            label=name,
            linewidth=2.5,
            color=colors[name],
            linestyle=linestyles[name],
        )
    ax_val_acc.set_xlabel("Epoch")
    ax_val_acc.set_ylabel("Accuracy")
    ax_val_acc.legend()
    ax_val_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overfitting_analysis(results, colors, linestyles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("UDL Project - Overfitting Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Overfitting gap evolution
    axes[0].set_title("Overfitting Gap Over Time", fontweight="bold")
    for name, data in results.items():
        overfitting_gap = np.array(data["train_accs"]) - np.array(data["val_accs"])
        axes[0].plot(
            overfitting_gap, label=name, linewidth=3, color=colors[name], linestyle=linestyles[name]
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train - Val Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="black", linestyle=":", alpha=0.5)

    # Plot 2: Final overfitting gaps
    axes[1].set_title("Final Overfitting Gaps", fontweight="bold")
    names = list(results.keys())
    gaps = [data["train_accs"][-1] - data["val_accs"][-1] for data in results.values()]
    bars = axes[1].bar(
        range(len(names)),
        gaps,
        color=[colors[name] for name in names],
        alpha=0.7,
        edgecolor="black",
    )
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Overfitting Gap")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, gap in zip(bars, gaps, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{gap:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 3: Final validation accuracies
    axes[2].set_title("Final Validation Accuracies", fontweight="bold")
    val_accs = [data["val_accs"][-1] for data in results.values()]
    bars = axes[2].bar(
        range(len(names)),
        val_accs,
        color=[colors[name] for name in names],
        alpha=0.7,
        edgecolor="black",
    )
    axes[2].set_xlabel("Method")
    axes[2].set_ylabel("Validation Accuracy")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, val_accs, strict=False):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_summary_dashboard(results, colors):
    fig = plt.figure(figsize=(12, 8))

    # Create a combined plot showing both training and validation curves
    plt.subplot(2, 2, 1)
    plt.title("Loss Curves", fontweight="bold")
    for name, data in results.items():
        plt.plot(data["train_losses"], color=colors[name], linestyle="-", alpha=0.7, linewidth=2)
        plt.plot(
            data["val_losses"], color=colors[name], linestyle="--", linewidth=2, label=f"{name}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.title("Accuracy Curves", fontweight="bold")
    for name, data in results.items():
        plt.plot(data["train_accs"], color=colors[name], linestyle="-", alpha=0.7, linewidth=2)
        plt.plot(data["val_accs"], color=colors[name], linestyle="--", linewidth=2, label=f"{name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary metrics
    plt.subplot(2, 2, 3)
    plt.title("Regularization Effectiveness", fontweight="bold")
    original_gap = (
        results["Original ResNet"]["train_accs"][-1] - results["Original ResNet"]["val_accs"][-1]
    )
    gap_reductions = []
    method_names = []

    for name, data in results.items():
        if name != "Original ResNet":
            gap = data["train_accs"][-1] - data["val_accs"][-1]
            reduction = (original_gap - gap) / original_gap * 100
            gap_reductions.append(reduction)
            method_names.append(name.replace(" ResNet", ""))

    bars = plt.bar(
        method_names, gap_reductions, color=["blue", "green"], alpha=0.7, edgecolor="black"
    )
    plt.ylabel("Gap Reduction (%)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for bar, reduction in zip(bars, gap_reductions, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{reduction:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.subplot(2, 2, 4)
    plt.title("Performance Summary", fontweight="bold")

    # Create a text summary
    plt.axis("off")
    summary_text = "REGULARIZATION COMPARISON SUMMARY\n\n"

    for name, data in results.items():
        train_acc = data["train_accs"][-1]
        val_acc = data["val_accs"][-1]
        gap = train_acc - val_acc

        if name == "Original ResNet":
            summary_text += f"{name}:\n"
            summary_text += f"  Validation Acc: {val_acc:.3f}\n"
            summary_text += f"  Overfitting Gap: {gap:.3f} (baseline)\n\n"
        else:
            val_improvement = (
                (val_acc - results["Original ResNet"]["val_accs"][-1])
                / results["Original ResNet"]["val_accs"][-1]
                * 100
            )
            gap_reduction = (original_gap - gap) / original_gap * 100

            summary_text += f"{name}:\n"
            summary_text += f"  Validation Acc: {val_acc:.3f} ({val_improvement:+.1f}%)\n"
            summary_text += f"  Overfitting Gap: {gap:.3f} ({gap_reduction:.1f}% reduction)\n\n"

    # Find best method based on the overfitting gap
    best_method = min(results.items(), key=lambda x: x[1]["train_accs"][-1] - x[1]["val_accs"][-1])[
        0
    ]

    summary_text += f"BEST REGULARIZATION:\n{best_method}"

    plt.text(
        0.05,
        0.95,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    return fig


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
