"""Cleanup script to remove deprecated use_path column from CSVs."""

import pandas as pd
import glob
import os


def remove_use_path():
    csv_paths = glob.glob("../../output/metrics_draft/*.csv")

    for path in csv_paths:
        df = pd.read_csv(path)
        if 'use_path' in df.columns:
            print(f"Removing use_path from {os.path.basename(path)}")
            df = df.drop(columns=['use_path'])
            df.to_csv(path, index=False)
            print(f"  ✅ Fixed")

    print("Done!")


if __name__ == '__main__':
    remove_use_path()