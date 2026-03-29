import matplotlib.pyplot as plt

def plot_ber(x_values, results, uncoded_results = None, channel_type = "Channel", sweep_mode = "snr"):

    plt.figure(figsize=(8,5))

    # 🔹 Кодированные (по блокам)
    for name, data in results.items():
        plt.semilogy(x_values, data["BER"], marker='o', label=f"{name} BER")

    # 🔹 Uncoded — одна линия
    if uncoded_results is not None:
        # берём первый (и по сути единственный) блок
        first_key = list(uncoded_results.keys())[0]
        uncoded_BER = uncoded_results[first_key]["BER"]
        plt.semilogy(x_values, uncoded_BER, 'k--', linewidth=2, label="Uncoded BER")
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
    if sweep_mode == "snr":
        plt.xlabel("h2 [dB]")
        plt.title(f"BER vs h2 for {channel_type}")
    else:
        plt.xlabel("P_rx [dBm]")
        plt.title(f"BER vs received power for {channel_type}")
    
    plt.ylabel('BER')
    plt.legend()
    plt.tight_layout()
    plt.show()