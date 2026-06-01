from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ONLY_CURVES = None

# plot_one_file = "No" - строит ВСЕ файлы | plot_one_file = "Название" - строит только файл с таким названием
plot_one_file = "No"

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

results_path = Path("res_tch_hs")

csv_files = list(results_path.glob("*.csv"))

if plot_one_file != "No":
    for f in csv_files:
        if plot_one_file == f.stem:
            csv_files = [f]
            break

if not csv_files:

    print("No CSV files found in Results/")
    exit()

plt.figure(figsize=(10, 7))

axis_metric_global = "ebn0_db"

count = 0

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

    i = 0
    for col in ber_columns:

        curve_name = col.replace("_BER", "")

        # legend_label = f"{curve_name}_{file_label}"
        legend_label = ["Дифф., кодированные", "Дифф., весь пакет", "Дифф., некодированные", "a"
                         "MLSE, кодированные", "MLSE, весь пакет", "MLSE, некодированные", "b"]
        color = ['r', 'b', 'g', 'c']

        # if "uncoded" in curve_name.lower():
        if count == 0:

            plt.semilogy(
                x_values,
                df[col],  
                marker='o',
                markersize=10,
                linewidth=6,
                label=legend_label[i],
                c=color[i]
            )

        else:
        
            plt.semilogy(
                x_values,
                df[col],
                '--',
                marker='o',
                markersize=10,
                linewidth=6,
                label=legend_label[count + i],
                c=color[i]
            )

        i += 1

    count += 3

plt.grid(
    True,
    which='both',
    linestyle='--',
    alpha=0.5
)

if axis_metric_global == "dbm":

    plt.xlabel("P_rx [dBm]", fontsize=14)

elif axis_metric_global == "ebn0_db":

    plt.xlabel("Eb/N0 [dB]",  fontsize=16)

else:

    plt.xlabel("SNR [dB]")


plt.ylabel("BER", fontsize=16)
plt.title("BER TCH/FS AWGN", fontsize=18)
plt.legend(fontsize=13, handlelength=6, loc="upper right")
# plt.ylim(5e-4, 0)
# plt.xlim(0, 24)

plt.tight_layout()
plt.show()

print("nice")