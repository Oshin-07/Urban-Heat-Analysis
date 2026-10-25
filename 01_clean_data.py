# Step 1: Data Cleaning
# Same steps as Practical 1 and Practical 3: read_csv, head, shape, info, describe, isnull, fillna, drop
# Run:  python 01_clean_data.py

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- read the three raw files
lst = pd.read_csv("data/raw/LST_Final_V3.csv")
no2 = pd.read_csv("data/raw/NO2_Final_V3.csv")
wind = pd.read_csv("data/raw/Wind_Final_V3.csv")

lst.head()
lst.shape
lst.info()
lst.describe()
no2.head()
wind.head()

# ---------------------------------------------------------------- check missing values and duplicates
print(lst.isnull().sum())
print(no2.isnull().sum())
print(wind.isnull().sum())
print("Duplicate dates:", lst["date"].duplicated().sum())

# fill missing numeric values with the mean (Practical 3) and drop unwanted column
lst.fillna(lst.mean(numeric_only=True), inplace=True)
no2.fillna(no2.mean(numeric_only=True), inplace=True)
wind.fillna(wind.mean(numeric_only=True), inplace=True)
no2.drop("unit", axis=1, inplace=True, errors="ignore")

# date column
lst["date"] = pd.to_datetime(lst["date"])
no2["date"] = pd.to_datetime(no2["date"])
wind["date"] = pd.to_datetime(wind["date"])

# ---------------------------------------------------------------- wide -> long (one row for every date, city and zone)
cities = ["Delhi", "Bangalore"]
zones = ["Urban", "Rural"]

parts = []
for city in cities:
    for zone in zones:
        site = city + "_" + zone
        part = pd.DataFrame({
            "date": lst["date"],
            "city": city,
            "zone": zone,
            "lst_c": lst[site],
            "no2_umol_m2": no2[site + "_NO2"],
            "wind_u": wind[site + "_u"],
            "wind_v": wind[site + "_v"],
            "wind_speed_file": wind[site + "_speed"],
        })
        parts.append(part)

df = pd.concat(parts, ignore_index=True)
df = df.sort_values(["city", "zone", "date"]).reset_index(drop=True)
print(df.shape)

# ---------------------------------------------------------------- outliers (cloud-contaminated LST values)
# deviation = LST - 15 day rolling median of the same site
df["lst_rollmed"] = df.groupby(["city", "zone"])["lst_c"].transform(
    lambda s: s.rolling(15, center=True, min_periods=8).median())
df["lst_dev"] = df["lst_c"] - df["lst_rollmed"]

# the paired site (urban <-> rural) of the same city on the same day
pair = df[["date", "city", "zone", "lst_dev"]].copy()
pair["zone"] = pair["zone"].map({"Urban": "Rural", "Rural": "Urban"})
pair = pair.rename(columns={"lst_dev": "pair_dev"})
df = df.merge(pair, on=["date", "city", "zone"], how="left")

# cold flag: 10 C below the rolling median
# hot flag : 10 C above the rolling median and the paired site did not also jump (a real heatwave shows at both)
cold = df["lst_dev"] < -10
hot = (df["lst_dev"] > 10) & (df["pair_dev"].fillna(0.0) <= 5)
df["lst_outlier"] = (cold | hot).astype(bool)
df = df.drop("pair_dev", axis=1)
df = df.sort_values(["date", "city", "zone"]).reset_index(drop=True)
print("Flagged outliers:", df["lst_outlier"].sum())

# ---------------------------------------------------------------- calendar columns
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["doy"] = df["date"].dt.dayofyear

season_of_month = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
df["season"] = df["month"].map(season_of_month)

# ---------------------------------------------------------------- final check and save
df.isnull().sum()
df.info()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/lst_long_clean.csv", index=False)
print("Saved data/processed/lst_long_clean.csv", df.shape)
