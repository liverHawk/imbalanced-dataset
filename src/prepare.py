import os
import yaml
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from ipaddress import ip_address as ip
from sklearn.model_selection import train_test_split
import imblearn.over_sampling as imblearn_os
from dvclive import Live

from lib.util import CICIDS2017, BASE


def fast_process(df, type="normal"):
    if type == "normal":
        df = df.drop(CICIDS2017().get_delete_columns(), axis=1)
    elif type == "full":
        df = df.drop(['Flow ID','Src IP','Attempted Category'], axis=1)
        # Timestamp→秒
        df['Timestamp'] = (
            pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            .astype('int64') // 10**9
        )
        # IP文字列→整数
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ip.IPv4Address(x)))
    # 欠損／無限大落とし
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def min_label_count(df):
    """
    Returns the minimum count of labels in the DataFrame.
    """
    return df["Label"].value_counts().min()


def oversampling(df, method, method_params):
    min_count = min_label_count(df)
    if min_count + 1 < method_params["neighbors"]:
        raise ValueError(
            f"Minimum label count {min_count} is less than neighbors {method_params['neighbors']}"
        )

    if method == "SMOTE":
        os_method = imblearn_os.SMOTE(
            k_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    elif method == "ADASYN":
        os_method = imblearn_os.ADASYN(
            n_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    else:
        raise ValueError(f"Unsupported oversampling method: {method}")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_resampled, y_resampled = os_method.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


def data_process(input_path, params):
    print("start")
    files = glob(f"{input_path}/*.csv")
    dfs = [fast_process(pd.read_csv(f)) for f in tqdm(files)]
    df  = pd.concat(dfs, ignore_index=True)

    rename_dict = {
        k: v for k, v in zip(CICIDS2017().get_features_labels(), BASE().get_features_labels())
    }
    df = df.rename(columns=rename_dict)

    train_df, test_df = train_test_split(df, test_size=params["split"], random_state=42)
    
    train_df = oversampling(
        train_df,
        method=params["oversampling"]["method"],
        method_params=params["oversampling"]["method_params"]
    )

    return train_df, test_df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    print(params)
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    output_train = os.path.join("data", "prepared", "train.csv")
    output_test = os.path.join("data", "prepared", "test.csv")

    if len(sys.argv) != 2:
        print("Usage: python src/prepare.py <input_file>")
        sys.exit(1)

    input = sys.argv[1]
    if not os.path.exists(input):
        print(f"Input file {input} does not exist.")
        sys.exit(1)

    train_df, test_df = data_process(
        input_path=input,
        params=params
    )

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)


if __name__ == "__main__":
    main()
