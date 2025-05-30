import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess
import sys

def run_training_scripts():
    # Run all three training scripts in sequence
    print("RUNNING ALL TRAINING SCRIPTS")
    print("="*60)
    
    scripts = [
        ("udl_project/train_model.py", "Original ResNet"),
        ("udl_project/train_model_l2_explicit.py", "L2 Regularized ResNet"), 
        ("udl_project/train_ensemble_model.py", "Ensemble ResNet")
    ]
    
    for script, name in scripts:
        print(f"\n  Running {name}...")
        try:
            result = subprocess.run([sys.executable, script], check=True)
            print(f"{name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}:")
            print(e.stdout)
            print(e.stderr)
            return False
        except FileNotFoundError:
            print(f"Script {script} not found")
            return False
    
    print("\nAll training scripts completed!")
    return True

def main():
    print("UDL PROJECT - COMPREHENSIVE REGULARIZATION EXPERIMENT")
    print("="*80)
    
    # Run all training scripts
    if not run_training_scripts():
        print("Training failed. Please check errors above.")
        return
    
    # Run the plotting script
    print("\nRunning plotting script...")
    try:
        result = subprocess.run([sys.executable, "plot.py"], check=True)
        print("Plotting completed successfully")
    except subprocess.CalledProcessError as e:
        print("Error running plotting script")
        return
    except FileNotFoundError:
        print("plot.py not found")
        return

if __name__ == "__main__":
    main()