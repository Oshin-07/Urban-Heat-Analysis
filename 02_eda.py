# Step 2: Exploratory Data Analysis (EDA)
# Same steps as Practical 1 and Practical 3: head, shape, info, describe, isnull, value_counts,
# histplot, barplot, correlation heatmap (coolwarm), savefig
# Run:  python 02_eda.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/processed/lst_long_clean.csv", parse_dates=["date"])
os.makedirs("outputs/figures", exist_ok=True)

# ---------------------------------------------------------------- basic information
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df["city"].value_counts()
df["zone"].value_counts()
df["lst_outlier"].value_counts()

# remove the flagged LST values for the analysis
data = df[df["lst_outlier"] == False]

# ---------------------------------------------------------------- univariate analysis
sns.histplot(data["lst_c"], bins=30, kde=True)
plt.title("LST distribution")
plt.xlabel("LST (C)")
plt.savefig("outputs/figures/01_lst_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

sns.boxplot(x="city", y="lst_c", hue="zone", data=data)
plt.title("LST by city and zone")
plt.savefig("outputs/figures/02_lst_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- mean LST
print(data.groupby(["city", "zone"])["lst_c"].mean())
print(data.groupby("season")["lst_c"].mean())

sns.barplot(x="season", y="lst_c", hue="city", data=data,
            order=["Winter", "Summer", "Monsoon", "Post-monsoon"])
plt.title("Average LST by season")
plt.savefig("outputs/figures/03_lst_by_season.png", dpi=150, bbox_inches="tight")
plt.close()

sns.barplot(x="year", y="lst_c", hue="city", data=data)
plt.title("Average LST by year")
plt.savefig("outputs/figures/04_lst_by_year.png", dpi=150, bbox_inches="tight")
plt.close()

monthly = data.groupby(["month", "city", "zone"])["lst_c"].mean().reset_index()
sns.lineplot(x="month", y="lst_c", hue="city", style="zone", data=monthly)
plt.title("Monthly average LST")
plt.savefig("outputs/figures/05_monthly_lst.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- urban - rural difference
wide = data.pivot_table(index=["date", "city"], columns="zone", values="lst_c").dropna()
wide["difference"] = wide["Urban"] - wide["Rural"]
print(wide.groupby("city")["difference"].mean())

monthly_diff = wide.reset_index()
monthly_diff["month"] = monthly_diff["date"].dt.month
monthly_diff = monthly_diff.groupby(["month", "city"])["difference"].mean().reset_index()
sns.barplot(x="month", y="difference", hue="city", data=monthly_diff)
plt.title("Urban - Rural LST by month")
plt.savefig("outputs/figures/06_urban_rural_difference.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- multivariate analysis
plt.figure(figsize=(10, 8))
sns.heatmap(
    data[["lst_c", "no2_umol_m2", "wind_u", "wind_v", "month"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation heatmap")
plt.savefig("outputs/figures/07_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

sns.scatterplot(x="no2_umol_m2", y="lst_c", hue="city", data=data, alpha=0.4)
plt.title("LST vs NO2")
plt.savefig("outputs/figures/08_lst_vs_no2.png", dpi=150, bbox_inches="tight")
plt.close()

print("EDA figures saved in outputs/figures")
