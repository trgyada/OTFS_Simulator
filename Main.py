# =============================================================================
# Main.py — OTFS simülasyonu giriş noktası.
#
# Modül yapısı:
#   config.py          → Çerçeve parametreleri (M, N, cp_len, ...)
#   modulation.py      → 16-QAM modülasyon / demodülasyon
#   channel.py         → Kanal modelleri (Ideal, AWGN)
#   simulation.py      → TX-RX pipeline ve SNR sweep
#   dashboard_plots.py → Grafik çizim fonksiyonları
#   dashboard.py       → Tkinter tabanlı gösterge paneli
# =============================================================================
from simulation import run_otfs_simulation, generate_bits, run_snr_sweep
from dashboard import launch_otfs_dashboard


if __name__ == "__main__":
    launch_otfs_dashboard(run_otfs_simulation, generate_bits, run_snr_sweep)