import matplotlib.pyplot as plt
import numpy as np

def _normalize_axis_metric(axis_metric):
    normalized = "dbm" if axis_metric is None else str(axis_metric).lower()
    aliases = {
        "dbm": "dbm",
        "power": "dbm",
        "power_dbm": "dbm",
        "signal_power_dbm": "dbm",
        "prx": "dbm",
        "db": "snr_db",
        "snr": "snr_db",
        "snr_db": "snr_db",
        "ebn0": "ebn0_db",
        "ebn0_db": "ebn0_db",
    }
    return aliases.get(normalized, "dbm")

@staticmethod
def plot_ber(x_values, results, uncoded_results = None, channel_type = "Channel", axis_metric = "dbm"):

    plt.figure(figsize=(8, 5))

    for name, data in results.items():
        plt.semilogy(x_values, data["BER"], marker = 'o', label = f"{name} BER")

    if uncoded_results is not None:

        first_key = list(uncoded_results.keys())[0]
        uncoded_ber = uncoded_results[first_key]["BER"]
        plt.semilogy(x_values, uncoded_ber, 'k--', linewidth = 2, label = "Uncoded BER")
    
    plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    
    normalized_axis = _normalize_axis_metric(axis_metric)
    if normalized_axis == "dbm":    
        plt.xlabel("P_rx [dBm]")
        plt.title(f"BER vs received power for {channel_type}")
    elif normalized_axis == "ebn0_db":
        plt.xlabel("Eb/N0 [dB]")
        plt.title(f"BER vs Eb/N0 for {channel_type}")
    else:
        plt.xlabel("SNR [dB]")
        plt.title(f"BER vs SNR for {channel_type}")
    
    plt.ylabel('BER')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def parse_and_plot(filename = "BER", channel_type = "Channel", axis_metric = "dbm"):
    data = {}
    with open(filename + '.txt', "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(' | ')

            # Pr_x, осш dBm
            x_list = parts[1].split('=')
            # Достаем численное значение ОСШ
            x = float(x_list[1].split()[0])

            # class, P_error
            y_list = parts[2].split(' BER=')
            name, ber = y_list[0], float(y_list[1])
            
            if name not in data:
                data[name] = {}
            data[name][x] = ber

    # Ось x
    keys_data = list(data.keys())
    x_coords = list(data[keys_data[0]].keys())

    # Ось y и названия графиков
    results = {}
    for name in data:
        ber_list = []
        for x in x_coords:
            val = data[name][x]
            ber_list.append(val)
        
        results[name] = {"BER": ber_list}
    
    # Ищем, есть ли в словаре uncoded
    uncoded_data = None
    uncoded_key = None
    for name in results.keys():
        if "uncoded" in name:
            uncoded_key = name
            break 
    
    # Вырезаем из словаря uncoded. (Чтобы не дублировалось)
    if uncoded_key is not None:
        uncoded_res = results.pop(uncoded_key)  
        uncoded_data = {uncoded_key: uncoded_res}
    
    plot_ber(x_coords, results, uncoded_results=uncoded_data, 
             channel_type=channel_type, axis_metric=axis_metric)

if __name__ == "__main__":
    parse_and_plot(filename = "BER_with_uncoded", channel_type = "Channel", axis_metric = "dbm")