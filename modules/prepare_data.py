import pandas as pd

def cut_september(df):
    return df[(df["timestamp"] < "2025-09-01") | (df["timestamp"] > "2025-10-15")].reset_index(drop=True)

def cut_august(df):
    return df[df["timestamp"] >= "2025-09-01"].reset_index(drop=True)

def cut_not_saturday(df):
    return df[df["timestamp"].dt.dayofweek == 5].reset_index(drop=True)

def cut_not_10am(df):
    return df[(df["timestamp"].dt.hour == 10) & (df["timestamp"].dt.minute == 30)].reset_index(drop=True)

def drop_status_6(df):
    return df.drop(columns=["status_6"]).reset_index(drop=True)