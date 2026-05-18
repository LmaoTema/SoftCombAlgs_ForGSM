import matplotlib.pyplot as plt


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
    