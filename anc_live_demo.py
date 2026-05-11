"""
Adaptive Noise Cancellation — Live Demo
ECE 6th Semester Project

Modes:
  1. Offline demo  — generates synthetic signals, runs LMS ANC, plots results
  2. Live mic demo — captures mic input in real time, cancels 50/60 Hz hum

Usage:
  python anc_live_demo.py          → offline demo (default, no mic needed)
  python anc_live_demo.py --live   → real-time mic demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CORE LMS ADAPTIVE FILTER
# ─────────────────────────────────────────────────────────────

class LMSFilter:
    """
    Least Mean Squares (LMS) adaptive filter.

    Parameters
    ----------
    order : int   — number of filter taps (controls noise model complexity)
    mu    : float — step size / learning rate (0 < mu < 1)
    """
    def __init__(self, order=64, mu=0.005):
        self.order = order
        self.mu    = mu
        self.w     = np.zeros(order)          # filter weights
        self.x_buf = np.zeros(order)          # circular input buffer

    def update(self, x_n, d_n):
        """
        Process one sample.

        x_n : float — reference signal sample  (noise only)
        d_n : float — primary signal sample    (desired + noise)

        Returns e_n (float) — error = estimated clean signal
        """
        # Shift buffer and insert new sample
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x_n

        y_n = np.dot(self.w, self.x_buf)        # filter output (noise estimate)
        e_n = d_n - y_n                          # error signal ≈ clean signal
        self.w += 2 * self.mu * e_n * self.x_buf  # LMS weight update
        return e_n

    def process_block(self, reference, primary):
        """Process arrays of samples at once. Returns output array."""
        out = np.zeros(len(primary))
        for n in range(len(primary)):
            out[n] = self.update(reference[n], primary[n])
        return out


class NLMSFilter(LMSFilter):
    """
    Normalized LMS — divides step size by signal power.
    More stable than LMS across varying noise amplitudes.
    """
    def __init__(self, order=64, mu=0.5, eps=1e-6):
        super().__init__(order, mu)
        self.eps = eps

    def update(self, x_n, d_n):
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x_n
        y_n = np.dot(self.w, self.x_buf)
        e_n = d_n - y_n
        norm = np.dot(self.x_buf, self.x_buf) + self.eps
        self.w += (self.mu / norm) * e_n * self.x_buf   # normalized update
        return e_n


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────

def snr_db(clean, signal):
    """Signal-to-Noise Ratio in dB."""
    noise      = signal - clean
    sig_power  = np.mean(clean**2) + 1e-12
    noise_power= np.mean(noise**2) + 1e-12
    return 10 * np.log10(sig_power / noise_power)

def mse(clean, signal):
    return np.mean((clean - signal)**2)


# ─────────────────────────────────────────────────────────────
#  OFFLINE DEMO
# ─────────────────────────────────────────────────────────────

def run_offline_demo():
    print("\n" + "="*60)
    print("  ADAPTIVE NOISE CANCELLATION — Offline Demo")
    print("="*60)

    FS      = 8000        # sample rate (Hz)
    DURATION= 3.0         # seconds
    t       = np.arange(0, DURATION, 1/FS)

    # ── Generate signals ──────────────────────────────────────
    # Clean speech-like signal: mix of 300 Hz and 800 Hz tones
    clean = (0.4 * np.sin(2*np.pi*300*t) +
             0.2 * np.sin(2*np.pi*800*t))

    # Noise: 50 Hz power-line hum (very common in ECE lab setups)
    noise_freq = 50
    noise = 0.7 * np.sin(2*np.pi*noise_freq*t + 0.3)  # phase offset

    # Primary mic = speech + noise
    primary   = clean + noise + 0.02*np.random.randn(len(t))

    # Reference mic = noise + tiny decorrelation (real mic leakage)
    reference = (0.7 * np.sin(2*np.pi*noise_freq*t) +
                 0.05*np.random.randn(len(t)))

    print(f"\nSignal frequency   : 300 Hz + 800 Hz (simulated voice)")
    print(f"Noise frequency    : {noise_freq} Hz (power-line hum)")
    print(f"Sample rate        : {FS} Hz  |  Duration: {DURATION}s")
    print(f"SNR before ANC     : {snr_db(clean, primary):+.1f} dB")

    # ── Run LMS ──────────────────────────────────────────────
    lms  = LMSFilter(order=64, mu=0.005)
    nlms = NLMSFilter(order=64, mu=0.5)

    out_lms  = lms.process_block(reference, primary)
    out_nlms = nlms.process_block(reference, primary)

    print(f"SNR after LMS      : {snr_db(clean, out_lms):+.1f} dB")
    print(f"SNR after NLMS     : {snr_db(clean, out_nlms):+.1f} dB")
    print(f"\nLMS improvement    : {snr_db(clean, out_lms) - snr_db(clean, primary):+.1f} dB")
    print(f"NLMS improvement   : {snr_db(clean, out_nlms) - snr_db(clean, primary):+.1f} dB")

    # ── Compute convergence curve ─────────────────────────────
    lms2 = LMSFilter(order=64, mu=0.005)
    mse_curve = []
    for n in range(len(primary)):
        e = lms2.update(reference[n], primary[n])
        mse_curve.append(e**2)
    mse_smooth = np.convolve(mse_curve, np.ones(200)/200, mode='same')

    # ── Plot ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor='#0d1117')
    fig.suptitle('Adaptive Noise Cancellation — LMS Demo', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    ax_style = dict(facecolor='#161b22', tick_params=dict(colors='#8b949e'),
                    xlabel_color='#8b949e', ylabel_color='#8b949e')

    # Time window for clarity
    t_ms = t * 1000
    ZOOM = slice(0, 400)   # first 50 ms

    axes = fig.subplots(3, 2)
    fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.08, right=0.97,
                        top=0.93, bottom=0.07)

    def style_ax(ax, title, xlabel='Time (ms)', ylabel='Amplitude'):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5)

    # Row 1: time-domain waveforms
    axes[0,0].plot(t_ms[ZOOM], primary[ZOOM],   color='#f85149', lw=0.8, label='Primary (noisy)')
    axes[0,0].plot(t_ms[ZOOM], clean[ZOOM],     color='#3fb950', lw=0.8, alpha=0.6, label='Clean (ref)')
    axes[0,0].legend(fontsize=7, facecolor='#21262d', labelcolor='white', loc='upper right')
    style_ax(axes[0,0], 'Primary input d(n) — noisy signal')

    axes[0,1].plot(t_ms[ZOOM], reference[ZOOM], color='#ff9f0a', lw=0.8)
    style_ax(axes[0,1], 'Reference input x(n) — noise only')

    # Row 2: output comparison
    axes[1,0].plot(t_ms[ZOOM], out_lms[ZOOM],  color='#58a6ff', lw=0.8, label='LMS output')
    axes[1,0].plot(t_ms[ZOOM], clean[ZOOM],    color='#3fb950', lw=0.8, alpha=0.5, label='Clean signal')
    axes[1,0].legend(fontsize=7, facecolor='#21262d', labelcolor='white', loc='upper right')
    style_ax(axes[1,0], 'LMS output e(n) — recovered signal')

    axes[1,1].plot(t_ms[ZOOM], out_nlms[ZOOM], color='#bc8cff', lw=0.8, label='NLMS output')
    axes[1,1].plot(t_ms[ZOOM], clean[ZOOM],    color='#3fb950', lw=0.8, alpha=0.5, label='Clean signal')
    axes[1,1].legend(fontsize=7, facecolor='#21262d', labelcolor='white', loc='upper right')
    style_ax(axes[1,1], 'NLMS output — faster convergence')

    # Row 3: frequency domain + convergence
    freqs = np.fft.rfftfreq(len(primary), 1/FS)
    def spectrum(sig):
        return 20*np.log10(np.abs(np.fft.rfft(sig)) / len(sig) + 1e-12)

    axes[2,0].plot(freqs, spectrum(primary),  color='#f85149', lw=0.8, label='Noisy input', alpha=0.8)
    axes[2,0].plot(freqs, spectrum(out_lms),  color='#58a6ff', lw=0.8, label='LMS output')
    axes[2,0].axvline(noise_freq, color='#ff9f0a', lw=1, ls='--', alpha=0.7, label=f'{noise_freq}Hz noise')
    axes[2,0].set_xlim(0, 1500)
    axes[2,0].legend(fontsize=7, facecolor='#21262d', labelcolor='white')
    style_ax(axes[2,0], 'Frequency spectrum — noise suppression', 'Frequency (Hz)', 'dB')

    axes[2,1].plot(np.arange(len(mse_smooth))/FS*1000, mse_smooth, 
                   color='#3fb950', lw=0.8)
    axes[2,1].axhline(y=mse_smooth[-100:].mean(), color='#ff9f0a', lw=1, ls='--', 
                      label=f'Converged MSE')
    axes[2,1].legend(fontsize=7, facecolor='#21262d', labelcolor='white')
    style_ax(axes[2,1], 'LMS convergence (MSE vs time)', 'Time (ms)', 'MSE')

    plt.savefig('anc_demo_results.png', 
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print("\n✓ Plot saved → anc_demo_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  LIVE MIC DEMO
# ─────────────────────────────────────────────────────────────

def run_live_demo():
    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice: pip install sounddevice --break-system-packages")
        return

    FS         = 16000
    BLOCK_SIZE = 512       # samples per callback
    NOISE_FREQ = 50        # Hz — target hum to cancel
    ORDER      = 64
    MU         = 0.01

    lms_filter = LMSFilter(order=ORDER, mu=MU)

    # Phase accumulator for reference signal generation
    # (In a real setup this would come from a second mic near the noise source)
    phase = [0.0]

    # Buffers for the live plot
    BUF_LEN  = FS * 3   # 3 seconds of history
    buf_in   = np.zeros(BUF_LEN)
    buf_out  = np.zeros(BUF_LEN)

    print("\n" + "="*60)
    print("  ANC LIVE DEMO — Real-time 50 Hz hum cancellation")
    print("="*60)
    print(f"Sample rate : {FS} Hz  |  Block size: {BLOCK_SIZE}")
    print(f"Filter order: {ORDER}  |  Step size (μ): {MU}")
    print("Press Ctrl+C to stop.\n")

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        primary = indata[:, 0].copy()

        # Synthesized reference: pure sine at noise frequency
        t_block = (np.arange(frames) + phase[0]) / FS
        reference = np.sin(2 * np.pi * NOISE_FREQ * t_block)
        phase[0] += frames

        output = np.array([lms_filter.update(reference[n], primary[n])
                           for n in range(frames)])
        outdata[:, 0] = output

        # Update circular buffers
        buf_in[:-frames]  = buf_in[frames:]
        buf_in[-frames:]  = primary
        buf_out[:-frames] = buf_out[frames:]
        buf_out[-frames:] = output

    # ── Animated live plot ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0d1117')
    fig.suptitle('ANC Live Demo — 50 Hz Hum Cancellation', 
                 color='white', fontsize=13, fontweight='bold')

    t_axis = np.arange(BUF_LEN) / FS * 1000   # ms

    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5)
        ax.set_xlim(0, BUF_LEN/FS*1000)
        ax.set_ylim(-1.2, 1.2)

    line_in,  = ax1.plot(t_axis, buf_in,  color='#f85149', lw=0.6)
    line_out, = ax2.plot(t_axis, buf_out, color='#58a6ff', lw=0.6)
    ax1.set_title('Microphone input (noisy)',       color='white', fontsize=9)
    ax2.set_title('ANC output (hum cancelled)',     color='white', fontsize=9)
    ax1.set_ylabel('Amplitude', color='#8b949e', fontsize=8)
    ax2.set_ylabel('Amplitude', color='#8b949e', fontsize=8)
    ax2.set_xlabel('Time (ms)', color='#8b949e', fontsize=8)

    def update_plot(frame):
        line_in.set_ydata(buf_in)
        line_out.set_ydata(buf_out)
        return line_in, line_out

    ani = animation.FuncAnimation(fig, update_plot, interval=80, blit=True)

    try:
        with sd.Stream(samplerate=FS, blocksize=BLOCK_SIZE,
                       dtype='float32', channels=1,
                       callback=audio_callback):
            plt.show()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as ex:
        print(f"Audio device error: {ex}")
        print("Run without --live for the offline demo.")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANC Live Demo")
    parser.add_argument("--live", action="store_true",
                        help="Run real-time mic demo (requires microphone)")
    args = parser.parse_args()

    if args.live:
        run_live_demo()
    else:
        run_offline_demo()
