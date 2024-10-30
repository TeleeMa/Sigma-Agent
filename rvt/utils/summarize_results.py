import argparse
import pandas as pd
import numpy as np

def average_results(args):
    df = pd.read_csv(args.csv_path)
    sr = df['success rate'].values
    print(sr)
    df_mean = np.mean(sr)
    print(df_mean)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=False)

    args = parser.parse_args()

    average_results(args)


