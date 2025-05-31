import matplotlib

from udl_project.plotter import plot
from udl_project.train_model import main as main_res
from udl_project.train_model_l2_explicit import main as main_l2
from udl_project.train_ensemble_model import main as main_ensemble

matplotlib.use("Agg")


def run_training_scripts():
    # Run all three training scripts in sequence
    print("RUNNING ALL TRAINING SCRIPTS")
    print("=" * 60)

    # These function should only accept the artifact path for now. In the future create a proper interface
    scripts = [
        (main_res, "Original ResNet"),
        (main_l2, "L2 Regularized ResNet"),
        (main_ensemble, "Ensemble ResNet"),
    ]

    for train_callable, name in scripts:
        print(f"\n  Running {name}...")
        train_callable()
        print(f"{name} completed successfully")

    print("\nAll training scripts completed!")


def main():
    print("UDL PROJECT - COMPREHENSIVE REGULARIZATION EXPERIMENT")
    print("=" * 80)

    # Run all training scripts
    run_training_scripts()

    # Run the plotting script
    print("\nRunning plotting script...")
    plot()
    print("Plotting completed successfully")


if __name__ == "__main__":
    main()
