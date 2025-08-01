import matplotlib as mpl

from udl_project.plotter import plot
from udl_project.training import config
from udl_project.training.abstract_trainer import Trainer as AbstractTrainer
from udl_project.training.ensemble_model_trainer import EnsembleModelTrainer
from udl_project.training.l2_regularized_model_trainer import L2RegularizedModelTrainer
from udl_project.training.resnet_model_trainer import ResNetModelTrainer

mpl.use("Agg")


def run_all_trainings():
    # Run all three training scripts in sequence
    print("RUNNING ALL TRAININGS")
    print("=" * 60)

    trainer: list[tuple[AbstractTrainer, str]] = [
        (ResNetModelTrainer(learning_rate=config.LEARNING_RATE, epochs=10), "Original ResNet"),
        (L2RegularizedModelTrainer(config.WEIGHT_DECAY, epochs=10), "L2 Regularized ResNet"),
        (EnsembleModelTrainer(config.NUMBER_OF_ENSEMBLE_MODELS, epochs=10), "Ensemble ResNet"),
    ]

    for trainer_instance, name in trainer:
        print(f"\n  Running {name}...")
        trainer_instance.train()
        print(f"{name} completed successfully")

    print("\nAll trainings completed!")


def main():
    print("UDL PROJECT - COMPREHENSIVE REGULARIZATION EXPERIMENT")
    print("=" * 80)

    # Run all training scripts
    run_all_trainings()

    # Run the plotting script
    print("\nRunning plotting script...")
    plot()
    print("Plotting completed successfully")


if __name__ == "__main__":
    main()
