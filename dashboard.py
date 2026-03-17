# =============================================================================
# dashboard.py — Tkinter dashboard
# =============================================================================

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import dashboard_plots as plots


def launch_otfs_dashboard(simulation_func, bits_generator, snr_sweep_func):
    root = tk.Tk()
    root.title("OTFS TX-RX Comparison Dashboard")
    root.geometry("1500x920")

    channel_type_var = tk.StringVar(master=root, value="Ideal")
    snr_var = tk.StringVar(master=root, value="20")
    trials_var = tk.StringVar(master=root, value="50")

    current_bits = bits_generator()

    results = simulation_func(
        bits=current_bits,
        channel_type=channel_type_var.get(),
        snr_db=float(snr_var.get())
    )

    sweep_results = None

    def fmt(arr, precision=3):
        if isinstance(arr, (float, int, np.floating, np.integer)):
            return str(arr)
        return np.array2string(np.array(arr), precision=precision, suppress_small=False)

    def build_comparison_text(res):
        sym_tx = plots.get_symbol_array(res)
        return {
            "Bits Comparison": (
                "TX Bits\n" + "=" * 50 + "\n" + fmt(res["bits"]) +
                "\n\nRX Bits\n" + "=" * 50 + "\n" + fmt(res["rx_bits"])
            ),
            "Symbols Comparison": (
                "TX Symbols\n" + "=" * 50 + "\n" + fmt(sym_tx) +
                "\n\nRX Symbols\n" + "=" * 50 + "\n" + fmt(res["rx_symbols"])
            ),
            "DD Grid Comparison": (
                "TX DD Grid\n" + "=" * 50 + "\n" + fmt(res["dd_grid"]) +
                "\n\nRX Y_dd\n" + "=" * 50 + "\n" + fmt(res["Y_dd"])
            ),
            "TF Grid Comparison": (
                "TX X_tf\n" + "=" * 50 + "\n" + fmt(res["X_tf"]) +
                "\n\nRX Y_tf\n" + "=" * 50 + "\n" + fmt(res["Y_tf"])
            ),
            "Block Comparison": (
                "TX x_time_blocks\n" + "=" * 50 + "\n" + fmt(res["x_time_blocks"]) +
                "\n\nRX rx_blocks\n" + "=" * 50 + "\n" + fmt(res["rx_blocks"])
            ),
            "CP Block Comparison": (
                "TX blocks_with_cp\n" + "=" * 50 + "\n" + fmt(res["blocks_with_cp"]) +
                "\n\nRX rx_blocks_with_cp\n" + "=" * 50 + "\n" + fmt(res["rx_blocks_with_cp"])
            ),
            "Signal Comparison": (
                "TX tx_signal\n" + "=" * 50 + "\n" + fmt(res["tx_signal"]) +
                "\n\nRX rx_signal\n" + "=" * 50 + "\n" + fmt(res["rx_signal"])
            ),
            "Metrics": (
                f"Channel Type = {res['channel_type']}\n"
                f"SNR (dB) = {res['snr_db']}\n"
                f"BER = {res['ber']}\n"
                f"SER = {res['ser']}"
            ),
        }

    comparison_text = build_comparison_text(results)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # =====================================================
    # TAB 1 - MATHEMATICAL VIEW
    # =====================================================
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Mathematical View")

    left_frame = ttk.Frame(tab1, width=280)
    left_frame.pack(side="left", fill="y", padx=10, pady=10)

    right_frame = ttk.Frame(tab1)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    ttk.Label(left_frame, text="Select Comparison", font=("Arial", 12, "bold")).pack(pady=5)

    listbox = tk.Listbox(left_frame, font=("Consolas", 10), height=20)
    listbox.pack(fill="y", expand=False)

    for key in comparison_text.keys():
        listbox.insert(tk.END, key)

    text_area = ScrolledText(right_frame, font=("Consolas", 10), wrap=tk.WORD)
    text_area.pack(fill="both", expand=True)

    def update_text_view(event=None):
        sel = listbox.curselection()
        if not sel:
            return
        key = listbox.get(sel[0])
        text_area.config(state="normal")
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, f"{key}\n")
        text_area.insert(tk.END, "=" * 80 + "\n\n")
        text_area.insert(tk.END, comparison_text[key])
        text_area.config(state="disabled")

    listbox.bind("<<ListboxSelect>>", update_text_view)
    listbox.selection_set(0)
    update_text_view()

    # =====================================================
    # TAB 2 - VISUAL VIEW
    # =====================================================
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Visual View")

    top_bar = ttk.Frame(tab2)
    top_bar.pack(fill="x", padx=10, pady=10)

    ttk.Label(top_bar, text="Channel:", font=("Arial", 11, "bold")).pack(side="left", padx=5)

    channel_selector = ttk.Combobox(
        top_bar,
        values=["Ideal", "AWGN", "Multipath"],
        state="readonly",
        width=10,
        textvariable=channel_type_var
    )
    channel_selector.pack(side="left", padx=5)

    ttk.Label(top_bar, text="SNR (dB):", font=("Arial", 11, "bold")).pack(side="left", padx=5)
    snr_entry = ttk.Entry(top_bar, width=8, textvariable=snr_var)
    snr_entry.pack(side="left", padx=5)

    ttk.Label(top_bar, text="Sweep Trials:", font=("Arial", 11, "bold")).pack(side="left", padx=5)
    trials_entry = ttk.Entry(top_bar, width=8, textvariable=trials_var)
    trials_entry.pack(side="left", padx=5)

    ttk.Label(top_bar, text="Select Comparison:", font=("Arial", 12, "bold")).pack(side="left", padx=15)

    plot_selector = ttk.Combobox(
        top_bar,
        values=[
            "Bits Comparison",
            "Symbols Comparison",
            "DD Grid Comparison",
            "TF Grid Comparison",
            "Block Comparison",
            "CP Block Comparison",
            "Signal Comparison",
            "BER/SER vs SNR",
        ],
        state="readonly",
        width=28,
    )
    plot_selector.pack(side="left", padx=10)
    plot_selector.set("Bits Comparison")

    metrics_label = ttk.Label(
        top_bar,
        text=f"BER = {results['ber']:.6f} | SER = {results['ser']:.6f}",
        font=("Arial", 11, "bold")
    )
    metrics_label.pack(side="right", padx=10)

    run_button = ttk.Button(top_bar, text="Run")
    run_button.pack(side="right", padx=10)

    new_bits_button = ttk.Button(top_bar, text="New Random Bits")
    new_bits_button.pack(side="right", padx=10)

    plot_frame = ttk.Frame(tab2)
    plot_frame.pack(fill="both", expand=True)

    fig = plt.Figure(figsize=(12, 6), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_selected_plot(event=None):
        fig.clear()
        selected = plot_selector.get()

        if selected == "BER/SER vs SNR":
            ax = fig.add_subplot(111)
            plots.draw_ber_ser_sweep(ax, sweep_results)
            fig.tight_layout()
            canvas.draw()
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        if selected == "Bits Comparison":
            plots.draw_bits_comparison(ax1, ax2, results)
        elif selected == "Symbols Comparison":
            plots.draw_symbols_comparison(ax1, ax2, results)
        elif selected == "DD Grid Comparison":
            plots.draw_dd_grid_comparison(fig, ax1, ax2, results)
        elif selected == "TF Grid Comparison":
            plots.draw_tf_grid_comparison(fig, ax1, ax2, results)
        elif selected == "Block Comparison":
            plots.draw_block_comparison(fig, ax1, ax2, results)
        elif selected == "CP Block Comparison":
            plots.draw_cp_block_comparison(fig, ax1, ax2, results)
        elif selected == "Signal Comparison":
            plots.draw_signal_comparison(ax1, ax2, results)

        fig.tight_layout()
        canvas.draw()

    def refresh_views():
        nonlocal comparison_text
        comparison_text = build_comparison_text(results)
        metrics_label.config(text=f"BER = {results['ber']:.6f} | SER = {results['ser']:.6f}")
        update_text_view()
        draw_selected_plot()

    def run_action():
        nonlocal results, sweep_results

        try:
            snr_db = float(snr_var.get())
        except ValueError:
            snr_var.set("20")
            snr_db = 20.0

        # Her zaman tek koşu hesaplanır
        results = simulation_func(
            bits=current_bits,
            channel_type=channel_type_var.get(),
            snr_db=snr_db
        )

        # Sweep görünümündeyse ayrıca sweep hesaplanır
        if plot_selector.get() == "BER/SER vs SNR":
            try:
                trials = int(trials_var.get())
            except ValueError:
                trials_var.set("50")
                trials = 50

            sweep_results = snr_sweep_func(
                channel_type=channel_type_var.get(),
                snr_db_list=list(range(0, 21, 2)),
                trials_per_snr=trials
            )
        else:
            sweep_results = None

        refresh_views()

    def new_random_bits():
        nonlocal current_bits, results, sweep_results
        current_bits = bits_generator()

        try:
            snr_db = float(snr_var.get())
        except ValueError:
            snr_var.set("20")
            snr_db = 20.0

        results = simulation_func(
            bits=current_bits,
            channel_type=channel_type_var.get(),
            snr_db=snr_db
        )

        sweep_results = None
        refresh_views()

    run_button.config(command=run_action)
    new_bits_button.config(command=new_random_bits)

    plot_selector.bind("<<ComboboxSelected>>", draw_selected_plot)
    draw_selected_plot()

    root.mainloop()