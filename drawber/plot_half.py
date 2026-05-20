import matplotlib.pyplot as plt
import numpy as np

def plot_ber_half(x_values, results, uncoded_ber_list, channel_type="Channel", title_suffix=""):

    plt.figure(figsize=(8, 5))
    
    colors = {'class1': '#1f77b4', 'class2': '#ff7f0e', 'full': '#2ca02c', 
              'header': '#d62728', 'data': '#9467bd'}

    for name, data in results.items():
        color = colors.get(name, None)
        label = f"{name} BER (coded)"
        plt.semilogy(x_values, data["BER"], marker='o', linestyle='-', 
                     color=color, label=label, linewidth=2)

    if uncoded_ber_list is not None:
        uncoded_ber = np.asarray(uncoded_ber_list)
        plt.semilogy(x_values, uncoded_ber, 'k--', linewidth=2, label="Uncoded BER")

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("RSSI [dBm]")
    plt.ylabel("BER")
    main_title = f"BER vs RSSI for {channel_type} "
    plt.title(main_title)
    plt.legend()
    plt.tight_layout()
    plt.show()