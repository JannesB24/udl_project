import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create a numpy array and compute its mean.")
    parser.add_argument("--values", nargs='+', type=float, required=True, help="List of numbers to process")
    args = parser.parse_args()
    arr = np.array(args.values)
    print(f"Array: {arr}")
    print(f"Mean: {np.mean(arr)}")

if __name__ == "__main__":
    main()