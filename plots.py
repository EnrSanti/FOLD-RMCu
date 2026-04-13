import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# -----------------------------
# Speedup data
# -----------------------------
speedup_data = """
Test,AvgSerial,AvgParallel,AvgSpeedup
lifestyle,127.8544426169974,7.768122850000509,16.458859506449365
lifestyle,193.33127398000215,12.998802093003178,14.873006958392414
human,86.41573047600104,6.5826519089896465,13.127798897885938
lifestyle,130.12834161698993,14.09875119899516,9.229777856230584
sloan,111.16348404900054,6.010417236990179,18.49513597240174
diabetes,72.90129529699334,6.538914865988772,11.148836892827427
smoke_drink,1518.1449034160032,101.90025388100185,14.898342698820736
coverType,746.8898803560005,29.176174587002606,25.599308028843673
MNIST,350.7314994910121,18.579526299989084,18.877311177261724
crops,627.5174418980023,48.493697650003014,12.94018547372956
earthquake,1699.5535870289896,79.97622038700501,21.250736516490125
weather,3853.6085423990007,347.58794196099916,11.086715265949564
"""

df = pd.read_csv(StringIO(speedup_data))
df["Test"] = df["Test"].str.lower()

df_speedup = df.groupby("Test", as_index=False)["AvgSpeedup"].mean()

# -----------------------------
# Metadata
# -----------------------------
meta = [
    ("jannis", 54, "70k"),
    ("miniboone", 50, "105k"),
    ("human", 562, "9k"),
    ("lifestyle", 31, "360k"),
    ("sloan", 41, "80k"),
    ("diabetes", 21, "203k"),
    ("weather", 11, "320k"),
    ("smoke_drink", 23, "392k"),
    ("covertype", 54, "465k"),
    ("mnist", 784, "60k"),
    ("crops", 174, "260k"),
    ("earthquake", 29, "610k"),
]

df_meta = pd.DataFrame(meta, columns=["Test", "Features", "Examples"])

def parse_k(x):
    x = x.lower().replace("∼", "").strip()
    return float(x.replace("k", "")) * 1000 if "k" in x else float(x)

df_meta["Examples"] = df_meta["Examples"].apply(parse_k)

# -----------------------------
# Merge
# -----------------------------
df_all = pd.merge(df_speedup, df_meta, on="Test", how="inner")
df_all["Size"] = df_all["Features"] * df_all["Examples"]

# -----------------------------
# Plot helper
# -----------------------------
def scatter_with_labels(x, y, xlabel, title, logx=False):
    plt.figure()

    plt.scatter(x, y)

    for i, r in df_all.iterrows():
        plt.text(x.iloc[i], y.iloc[i], r["Test"], fontsize=8)

    if logx:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("Speedup")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.show()

# -----------------------------
# 1) Speedup vs Features (log x)
# -----------------------------
scatter_with_labels(
    df_all["Features"],
    df_all["AvgSpeedup"],
    "Number of features",
    "Speedup vs Features (log scale)",
    logx=True
)

# -----------------------------
# 2) Speedup vs Examples (log x)
# -----------------------------
scatter_with_labels(
    df_all["Examples"],
    df_all["AvgSpeedup"],
    "Number of examples",
    "Speedup vs Examples (log scale)",
    logx=True
)

# -----------------------------
# 3) Speedup vs Size (log x) ⭐ most important
# -----------------------------
scatter_with_labels(
    df_all["Size"],
    df_all["AvgSpeedup"],
    "Dataset size (features × examples)",
    "Speedup vs Dataset Size (log scale)",
    logx=True
)