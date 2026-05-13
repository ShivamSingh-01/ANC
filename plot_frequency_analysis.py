"""
plot_frequency_analysis.py
--------------------------
Generates Figure 1: Comparison of noise spectra before and after ANC.

This script extends the existing ANC project (ShivamSingh-01/ANC) by adding
a frequency-domain (FFT) visualization that matches Section 0.15 of the report.
It does NOT modify or remove any existing code — it only adds new functionality.

The signal setup mirrors offline_demo.py (same FS, hum frequencies, filter chain)
but adds a 600 Hz noise tone (as mentioned in the report) so the FFT plot shows
a clear peak before ANC and its suppression after filtering.

Usage:
    python plot_frequency_analysis.py

Output:
    frequency_comparison.png  — Figure 1 for the report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Inline lightweight LMS filter (no dependency on filters.py needed to run
# standalone, but keeps compatibility — import from filters.py if available)
# ---------------------------------------------------------------------------
try:
    from filters import NLMSFilter
    USE_PROJECT_FILTER = True
except ImportError:
    USE_PROJECT_FILTER = False


class _StandaloneLMS:
    """Minimal LMS for standalone use when filters.py is not on the path."""
    def __init__(self, order=64, mu=0.005):
        self.order = order
        self.mu = mu
        self.w = np.zeros(order)

    def process_block(self, ref: np.ndarray, primary: np.ndarray) -> np.ndarray:
        N = len(primary)
        output = np.zeros(N)
        buf = np.zeros(self.order)
        for n in range(N):
            buf = np.roll(buf, 1)
            buf[0] = ref[n]
            y = float(np.dot(self.w, buf))
            e = primary[n] - y
            norm = float(np.dot(buf, buf)) + 1e-8
            self.w += (self.mu / norm) * e * buf
            output[n] = e
        return output


# ---------------------------------------------------------------------------
# Signal generation — mirrors offline_demo.py but adds 600 Hz tonal noise
# ---------------------------------------------------------------------------
FS = 8000          # sample rate (Hz)
DURATION = 3.0     # seconds
NFFT = 8192        # FFT size for smooth frequency axis

t = np.arange(0, DURATION, 1.0 / FS)

# Clean speech-like signal (300 Hz + 800 Hz tones, as in offline_demo.py)
clean = (
    0.40 * np.sin(2 * np.pi * 300 * t) +
    0.20 * np.sin(2 * np.pi * 800 * t)
)

# Noise: 50 Hz hum + 150 Hz harmonic (project default) + 600 Hz tonal noise
# (600 Hz is specifically mentioned in Section 0.15 as the example frequency)
hum = (
    0.70 * np.sin(2 * np.pi * 50 * t + 0.30) +
    0.20 * np.sin(2 * np.pi * 150 * t + 0.10)
)
tonal_600 = 0.55 * np.sin(2 * np.pi * 600 * t + 0.5)   # prominent 600 Hz tone
thermal = 0.01 * np.random.default_rng(42).standard_normal(len(t))

primary = clean + hum + tonal_600 + thermal   # what the primary mic picks up

# Reference signal (synthesised hum + 600 Hz) — what the reference mic picks up
ref_raw = (
    np.sin(2 * np.pi * 50 * t) +
    0.30 * np.sin(2 * np.pi * 150 * t) +
    np.sin(2 * np.pi * 600 * t)
)
reference = ref_raw / (np.max(np.abs(ref_raw)) + 1e-12)

# ---------------------------------------------------------------------------
# Run adaptive filter
# ---------------------------------------------------------------------------
if USE_PROJECT_FILTER:
    filt = NLMSFilter(order=64, mu=0.5)
else:
    filt = _StandaloneLMS(order=64, mu=0.005)

error_signal = filt.process_block(reference, primary)   # residual after ANC

# ---------------------------------------------------------------------------
# Compute FFT magnitude spectra
# ---------------------------------------------------------------------------
window = np.hanning(len(primary))

def compute_spectrum(sig, nfft=NFFT, fs=FS):
    """Return (frequencies in Hz, magnitude spectrum in dB)."""
    w = np.hanning(len(sig))
    S = np.fft.rfft(sig * w, n=nfft)
    mag = np.abs(S)
    mag_db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag_db

freqs_in, mag_in = compute_spectrum(primary)
freqs_out, mag_out = compute_spectrum(error_signal)

# Frequency axis limit for display
F_MAX = 1200   # Hz — covers all relevant peaks

mask = freqs_in <= F_MAX

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 6))
fig.suptitle(
    "Figure 1: Noise Spectra Before and After ANC\n"
    r"$|X(f)|$ — Primary Microphone vs Residual Error Signal",
    fontsize=13, fontweight='bold', y=1.01
)

gs = gridspec.GridSpec(2, 1, hspace=0.55)

# ── Top panel: Before ANC ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(freqs_in[mask], mag_in[mask], color='#d62728', linewidth=1.4,
         label='Before ANC (primary mic)')
ax1.set_title('Before ANC — Primary Microphone Signal', fontsize=11)
ax1.set_ylabel('Magnitude (dB)', fontsize=10)
ax1.set_xlim(0, F_MAX)
ax1.set_ylim(-80, 5)
ax1.grid(True, alpha=0.35)

# Annotate key peaks
for hz, label in [(50, '50 Hz'), (150, '150 Hz'), (300, '300 Hz'),
                  (600, '600 Hz\n(tonal noise)'), (800, '800 Hz')]:
    idx = np.argmin(np.abs(freqs_in - hz))
    y_val = mag_in[idx]
    if y_val > -60:
        ax1.annotate(
            label,
            xy=(hz, y_val),
            xytext=(hz + 20, y_val + 8),
            fontsize=7.5,
            arrowprops=dict(arrowstyle='->', lw=0.8, color='#444'),
            color='#222'
        )

ax1.legend(fontsize=9, loc='upper right')

# ── Bottom panel: After ANC ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(freqs_out[mask], mag_out[mask], color='#1f77b4', linewidth=1.4,
         label='After ANC (residual error signal)')
ax2.set_title('After ANC — Residual Error Signal', fontsize=11)
ax2.set_ylabel('Magnitude (dB)', fontsize=10)
ax2.set_xlabel('Frequency (Hz)', fontsize=10)
ax2.set_xlim(0, F_MAX)
ax2.set_ylim(-80, 5)
ax2.grid(True, alpha=0.35)

# Annotate suppressed and retained peaks
idx_600 = np.argmin(np.abs(freqs_out - 600))
ax2.annotate(
    '600 Hz\n(suppressed)',
    xy=(600, mag_out[idx_600]),
    xytext=(630, mag_out[idx_600] + 18),
    fontsize=7.5,
    arrowprops=dict(arrowstyle='->', lw=0.8, color='#444'),
    color='#1a7abf'
)

ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
out_path = "frequency_comparison.png"
fig.savefig(out_path, dpi=180, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()