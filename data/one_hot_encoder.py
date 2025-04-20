import pandas as pd

input_csv = "./phonology/data/historical_phonology.csv"

def one_hot_encode_column(df, column_name):
    df[column_name] = df[column_name].fillna("")

    # Get unique values excluding blanks
    unique_values = sorted(set(df[column_name]) - {""})
    value_map = {val: i for i, val in enumerate(unique_values)}

    def encode_one_hot(val):
        vec = [0] * len(value_map)
        if val in value_map:
            vec[value_map[val]] = 1
        return vec

    # Add compact one-hot vector column
    df[f"{column_name}_vector"] = df[column_name].apply(encode_one_hot)
    df = df.drop(columns=[column_name])
    return df


if __name__ == "__main__":
    df = pd.read_csv(input_csv)
    for clmn in df.columns[2:]:
        df = one_hot_encode_column(df, clmn)

    df.to_csv("phonology.csv", index=False, encoding="utf-8")
