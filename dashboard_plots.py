# =============================================================================
# dashboard_plots.py - Grafik cizim fonksiyonlari
# =============================================================================

import numpy as np


def get_symbol_array(res):
    if "qam16_symbols" in res:
        return res["qam16_symbols"]
    if "qpsk_symbols" in res:
        return res["qpsk_symbols"]
    raise KeyError("Sonuc sozlugu 'qam16_symbols' veya 'qpsk_symbols' icermeli.")


def _style_constellation_ax(ax, title):
    ax.set_title(title)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_facecolor("black")
    ax.grid(True, color="gray", alpha=0.3)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal", adjustable="box")


def draw_bits_comparison(ax1, ax2, results):
    ax1.scatter(np.arange(len(results["bits"])), results["bits"], s=35)
    ax1.set_title("TX Bits")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)

    ax2.scatter(np.arange(len(results["rx_bits"])), results["rx_bits"], s=35)
    ax2.set_title("RX Bits")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)


def draw_symbols_comparison(ax1, ax2, results):
    norm = np.sqrt(10)
    s_tx = get_symbol_array(results)
    s_rx = results["rx_symbols"]
    eq_suffix = f" ({results.get('equalizer_type', 'None')})" if results.get("equalization_enabled") else ""

    ax1.scatter(np.real(s_tx) * norm, np.imag(s_tx) * norm, s=80, c="yellow")
    _style_constellation_ax(ax1, "TX 16-QAM")

    ax2.scatter(np.real(s_rx) * norm, np.imag(s_rx) * norm, s=80, c="yellow")
    _style_constellation_ax(ax2, f"RX 16-QAM{eq_suffix}")


def draw_dd_grid_comparison(fig, ax1, ax2, results):
    im1 = ax1.imshow(np.real(results["dd_grid"]), aspect="auto", cmap="coolwarm")
    ax1.set_title("TX Real(DD Grid)")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.real(results["Y_dd"]), aspect="auto", cmap="coolwarm")
    ax2.set_title("RX Real(Y_dd)")
    fig.colorbar(im2, ax=ax2)


def draw_tf_grid_comparison(fig, ax1, ax2, results):
    im1 = ax1.imshow(np.abs(results["X_tf"]), aspect="auto", cmap="viridis")
    ax1.set_title("TX |X_tf|")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.abs(results["Y_tf"]), aspect="auto", cmap="viridis")
    if results.get("equalization_enabled"):
        ax2.set_title(f"RX |Y_tf| ({results.get('equalizer_type', 'None')})")
    else:
        ax2.set_title("RX |Y_tf|")
    fig.colorbar(im2, ax=ax2)


def draw_block_comparison(fig, ax1, ax2, results):
    im1 = ax1.imshow(np.real(results["x_time_blocks"]), aspect="auto", cmap="coolwarm")
    ax1.set_title("TX Real(x_time_blocks)")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.real(results["rx_blocks"]), aspect="auto", cmap="coolwarm")
    ax2.set_title("RX Real(rx_blocks)")
    fig.colorbar(im2, ax=ax2)


def _annotate_cp_block(ax, cp_len, total_rows):
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Sample Index")

    if cp_len <= 0 or cp_len >= total_rows:
        return

    ax.axhline(y=cp_len - 0.5, color="black", linestyle="--", linewidth=1.5)

    label_style = dict(
        color="black",
        fontsize=11,
        fontweight="bold",
        va="center",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    cp_center = (cp_len - 1) / 2
    data_center = cp_len + (total_rows - cp_len - 1) / 2
    ax.text(-0.35, cp_center, "CP", **label_style)
    ax.text(-0.35, data_center, "Data", **label_style)


def draw_cp_block_comparison(fig, ax1, ax2, results):
    cp_len = results["blocks_with_cp"].shape[0] - results["x_time_blocks"].shape[0]
    total_rows = results["blocks_with_cp"].shape[0]

    im1 = ax1.imshow(np.real(results["blocks_with_cp"]), aspect="auto", cmap="coolwarm")
    ax1.set_title("TX Real(blocks_with_cp)")
    _annotate_cp_block(ax1, cp_len=cp_len, total_rows=total_rows)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.real(results["rx_blocks_with_cp"]), aspect="auto", cmap="coolwarm")
    ax2.set_title("RX Real(rx_blocks_with_cp)")
    _annotate_cp_block(ax2, cp_len=cp_len, total_rows=total_rows)
    fig.colorbar(im2, ax=ax2)


def draw_signal_comparison(ax1, ax2, results):
    ax1.plot(np.real(results["tx_signal"]), label="Real")
    ax1.plot(np.imag(results["tx_signal"]), label="Imag")
    ax1.set_title("TX Signal")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(np.real(results["rx_signal"]), label="Real")
    ax2.plot(np.imag(results["rx_signal"]), label="Imag")
    ax2.set_title("RX Signal")
    ax2.legend()
    ax2.grid(True)


def draw_ber_ser_sweep(ax, sweep_results):
    if sweep_results is None:
        ax.text(0.5, 0.5, "BER/SER vs SNR secenegiyle Run'a basin", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.semilogy(sweep_results["snr_db_list"], sweep_results["ber_list"], marker="o", label="BER")
    ax.semilogy(sweep_results["snr_db_list"], sweep_results["ser_list"], marker="s", label="SER")

    if sweep_results.get("equalization_enabled"):
        eq_label = f"EQ: {sweep_results.get('equalizer_type', 'None')}"
    else:
        eq_label = "EQ: Off"

    ax.set_title(f"16-QAM {sweep_results['channel_type']} Kanali | {eq_label}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Hata Orani")
    ax.grid(True, which="both")
    ax.legend()
