from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ONLY_CURVES = None

def normalize_axis_metric(axis_metric):

    normalized = str(axis_metric).lower()

    aliases = {
        "dbm": "dbm",
        "power": "dbm",
        "snr": "snr_db",
        "snr_db": "snr_db",
        "ebn0": "ebn0_db",
        "ebn0_db": "ebn0_db",
    }

    return aliases.get(normalized, "dbm")

results_path = Path("Results")

csv_files = list(results_path.glob("*.csv"))

if not csv_files:

    print("No CSV files found in Results/")
    exit()

plt.figure(figsize=(10, 7))

axis_metric_global = None


for file in csv_files:

    df = pd.read_csv(file)

    file_label = file.stem

    x_values = df["x_value"]

    axis_metric = normalize_axis_metric(
        df["axis_metric"].iloc[0]
    )

    if axis_metric_global is None:
        axis_metric_global = axis_metric

    ber_columns = []

    for col in df.columns:

        if not col.endswith("_BER"):
            continue

        curve_name = col.replace("_BER", "")

        if ONLY_CURVES is not None:

            if curve_name not in ONLY_CURVES:
                continue

        ber_columns.append(col)
        
    for col in ber_columns:

        curve_name = col.replace("_BER", "")

        legend_label = f"{curve_name}_{file_label}"

        if "uncoded" in curve_name.lower():

            plt.semilogy(
                x_values,
                df[col],
                '--',
                linewidth=2,
                label=legend_label
            )

        else:

            plt.semilogy(
                x_values,
                df[col],
                marker='o',
                label=legend_label
            )

plt.grid(
    True,
    which='both',
    linestyle='--',
    alpha=0.5
)

if axis_metric_global == "dbm":

    plt.xlabel("P_rx [dBm]")

elif axis_metric_global == "ebn0_db":

    plt.xlabel("Eb/N0 [dB]")

else:

    plt.xlabel("SNR [dB]")


plt.ylabel("BER")
plt.title("BER TCH/FS AWGN")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()