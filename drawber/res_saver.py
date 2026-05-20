from pathlib import Path
import pandas as pd

def save_ber_results(res_coded, res_uncoded=None, simulation_params=None):

    results_dir = Path("Results")
    results_dir.mkdir(exist_ok=True)

    axis_metric = res_coded["axis_metric"]

    x_values = res_coded["x"]

    coded_results = res_coded["results"]

    channel_metrics = res_coded.get("channel_metrics", {})

    rows = []

    for i, x in enumerate(x_values):

        row = {
            "x_value": x,
            "axis_metric": axis_metric,
        }

        for block_name, metrics in coded_results.items():

            ber = metrics["BER"][i]
            fer = metrics["FER"][i]

            row[f"{block_name}_BER"] = ber
            row[f"{block_name}_FER"] = fer

        if res_uncoded is not None:

            uncoded_key = list(res_uncoded["results"].keys())[0]

            row["uncoded_BER"] = (
                res_uncoded["results"][uncoded_key]["BER"][i]
            )
        for metric_name, values in channel_metrics.items():

            if i < len(values):
                row[metric_name] = values[i]

        rows.append(row)

    df = pd.DataFrame(rows)

    file_name = simulation_params["file_name"]

    filename = f"{file_name}.csv"

    full_path = results_dir / filename

    df.to_csv(full_path, index=False)

    print(f"\nResults saved to: {full_path}\n")