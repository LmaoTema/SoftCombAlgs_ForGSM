from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

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

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

axis_metric_global = "ebn0_db"   # "ebn0_db"

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
        legend_label = ["Дифф., некодированные", "Дифф., весь пакет", "Дифф., некодированные",
                         "MLSE, некодированные", "MLSE, весь пакет", "MLSE, некодированные"]
        color = ['r', 'b', 'g', 'c']

        # if "uncoded" in curve_name.lower():
        if count == 0:

            ax.semilogy(
                x_values,
                df[col],  
                marker='o',
                markersize=10,
                linewidth=6,
                label=legend_label[i],
                c=color[i]
            )

        else:
        
            ax.semilogy(
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

ax.grid(
    True,
    which='both',
    linestyle='--',
    alpha=0.5
)

if axis_metric_global == "dbm":

    ax.set_xlabel("P_rx [dBm]", fontsize=14)

elif axis_metric_global == "ebn0_db":

    ax.set_xlabel("$E_b/N_0$ [дБ]",  fontsize=18)

else:

    ax.set_alphaxlabel("SNR [dB]")


ax.set_ylabel("BER", fontsize=18)
ax.set_title("TCH/FS АБГШ", fontsize=20)
ax.legend(fontsize=16, handlelength=2.5, loc="upper right")
ax.tick_params(axis='both', which='major', labelsize=18)

plt.ylim(1e-5, 1)
plt.xlim(0, 14)

plt.tight_layout()
plt.show()

print("nice")