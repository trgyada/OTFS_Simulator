import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def launch_otfs_dashboard(simulation_func, bits_generator):
    root = tk.Tk()
    root.title("OTFS TX-RX Comparison Dashboard")
    root.geometry("1500x920")

    channel_type_var = tk.StringVar(master=root, value="Ideal")
    snr_var = tk.StringVar(master=root, value="20")

    current_bits = bits_generator()

    results = simulation_func(
        bits=current_bits,
        channel_type=channel_type_var.get(),
        snr_db=float(snr_var.get())
    )

    def fmt(arr, precision=3):
        if isinstance(arr, (float, int, np.floating, np.integer)):
            return str(arr)
        return np.array2string(np.array(arr), precision=precision, suppress_small=False)

    def build_comparison_text(res):
        return {
            "Bits Comparison": (
                "TX Bits\n" + "=" * 50 + "\n" + fmt(res["bits"]) +
                "\n\nRX Bits\n" + "=" * 50 + "\n" + fmt(res["rx_bits"])
            ),
            "Symbols Comparison": (
                "TX Symbols\n" + "=" * 50 + "\n" + fmt(res["qpsk_symbols"]) +
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
        values=["Ideal", "AWGN"],
        state="readonly",
        width=10,
        textvariable=channel_type_var
    )
    channel_selector.pack(side="left", padx=5)

    ttk.Label(top_bar, text="SNR (dB):", font=("Arial", 11, "bold")).pack(side="left", padx=5)

    snr_entry = ttk.Entry(top_bar, width=8, textvariable=snr_var)
    snr_entry.pack(side="left", padx=5)

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

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        if selected == "Bits Comparison":
            x_tx = np.arange(len(results["bits"]))
            x_rx = np.arange(len(results["rx_bits"]))

            ax1.scatter(x_tx, results["bits"], s=35)
            ax1.set_title("TX Bits")
            ax1.set_ylim(-0.1, 1.1)
            ax1.grid(True)

            ax2.scatter(x_rx, results["rx_bits"], s=35)
            ax2.set_title("RX Bits")
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True)

        elif selected == "Symbols Comparison":
            s_tx = results["qpsk_symbols"]
            s_rx = results["rx_symbols"]

            ax1.scatter(np.real(s_tx) * np.sqrt(10), np.imag(s_tx) * np.sqrt(10), s=80, c="yellow")
            ax1.set_title("TX 16-QAM")
            ax1.set_xlabel("In-Phase")
            ax1.set_ylabel("Quadrature")
            ax1.set_facecolor("black")
            ax1.grid(True, color="gray", alpha=0.3)
            ax1.tick_params(colors="white")
            ax1.xaxis.label.set_color("white")
            ax1.yaxis.label.set_color("white")
            ax1.title.set_color("white")
            ax1.set_xlim(-3.5, 3.5)
            ax1.set_ylim(-3.5, 3.5)
            ax1.set_aspect("equal", adjustable="box")

            ax2.scatter(np.real(s_rx) * np.sqrt(10), np.imag(s_rx) * np.sqrt(10), s=80, c="yellow")
            ax2.set_title("RX 16-QAM")
            ax2.set_xlabel("In-Phase")
            ax2.set_ylabel("Quadrature")
            ax2.set_facecolor("black")
            ax2.grid(True, color="gray", alpha=0.3)
            ax2.tick_params(colors="white")
            ax2.xaxis.label.set_color("white")
            ax2.yaxis.label.set_color("white")
            ax2.title.set_color("white")
            ax2.set_xlim(-3.5, 3.5)
            ax2.set_ylim(-3.5, 3.5)
            ax2.set_aspect("equal", adjustable="box")

        elif selected == "DD Grid Comparison":
            im1 = ax1.imshow(np.real(results["dd_grid"]), aspect="auto", cmap="coolwarm")
            ax1.set_title("TX Real(DD Grid)")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(np.real(results["Y_dd"]), aspect="auto", cmap="coolwarm")
            ax2.set_title("RX Real(Y_dd)")
            fig.colorbar(im2, ax=ax2)

        elif selected == "TF Grid Comparison":
            im1 = ax1.imshow(np.abs(results["X_tf"]), aspect="auto", cmap="viridis")
            ax1.set_title("TX |X_tf|")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(np.abs(results["Y_tf"]), aspect="auto", cmap="viridis")
            ax2.set_title("RX |Y_tf|")
            fig.colorbar(im2, ax=ax2)

        elif selected == "Block Comparison":
            im1 = ax1.imshow(np.real(results["x_time_blocks"]), aspect="auto", cmap="coolwarm")
            ax1.set_title("TX Real(x_time_blocks)")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(np.real(results["rx_blocks"]), aspect="auto", cmap="coolwarm")
            ax2.set_title("RX Real(rx_blocks)")
            fig.colorbar(im2, ax=ax2)

        elif selected == "CP Block Comparison":
            im1 = ax1.imshow(np.real(results["blocks_with_cp"]), aspect="auto", cmap="coolwarm")
            ax1.set_title("TX Real(blocks_with_cp)")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(np.real(results["rx_blocks_with_cp"]), aspect="auto", cmap="coolwarm")
            ax2.set_title("RX Real(rx_blocks_with_cp)")
            fig.colorbar(im2, ax=ax2)

        elif selected == "Signal Comparison":
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

        fig.tight_layout()
        canvas.draw()

    def refresh_views():
        nonlocal comparison_text
        comparison_text = build_comparison_text(results)
        metrics_label.config(text=f"BER = {results['ber']:.6f} | SER = {results['ser']:.6f}")
        update_text_view()
        draw_selected_plot()

    def rerun_simulation():
        nonlocal results
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
        refresh_views()

    def new_random_bits():
        nonlocal current_bits, results
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
        refresh_views()

    run_button.config(command=rerun_simulation)
    new_bits_button.config(command=new_random_bits)

    plot_selector.bind("<<ComboboxSelected>>", draw_selected_plot)
    draw_selected_plot()

    root.mainloop()