import numpy as np
import matplotlib.pyplot as plt

from filters import LMSFilter, NLMSFilter, RLSFilter

from signal_utils import (
    snr_db,
    mse_metric,
    detect_hum_frequencies,
    save_audio
)


def generate_test_signals(fs=8000, duration=3.0):

    t = np.arange(0, duration, 1.0 / fs)

    clean = (
        0.40 * np.sin(2 * np.pi * 300 * t) +
        0.20 * np.sin(2 * np.pi * 800 * t)
    )

    hum = (
        0.70 * np.sin(2 * np.pi * 50 * t + 0.30) +
        0.20 * np.sin(2 * np.pi * 150 * t + 0.10)
    )

    thermal = 0.01 * np.random.randn(len(t))

    primary = clean + hum + thermal

    ref_raw = (
        np.sin(2 * np.pi * 50 * t) +
        0.30 * np.sin(2 * np.pi * 150 * t)
    )

    reference = ref_raw / (np.max(np.abs(ref_raw)) + 1e-12)

    return t, clean, primary, reference


def run_all_filters(reference, primary, clean):

    filters = {
        'LMS': LMSFilter(order=64, mu=0.005),
        'NLMS': NLMSFilter(order=64, mu=0.5),
        'RLS': RLSFilter(order=16,
    lam=0.999,
    delta=100.0)
    }

    results = {}

    for name, filt in filters.items():

        output = filt.process_block(reference, primary)

        results[name] = {
            'output': output,
            'snr': snr_db(clean, output),
            'mse': mse_metric(clean, output),
        }

    return results


def run_offline_demo():

    FS = 8000

    t, clean, primary, reference = generate_test_signals(fs=FS)

    detected = detect_hum_frequencies(primary, FS)

    print("\nDetected Hum Frequencies:")
    print(detected)

    input_snr = snr_db(clean, primary)

    print(f"\nInput SNR: {input_snr:.2f} dB")

    results = run_all_filters(reference, primary, clean)

    for name, res in results.items():

        print(f"\n{name}")

        print(f"SNR : {res['snr']:.2f} dB")

        print(f"MSE : {res['mse']:.6f}")

        save_audio(res['output'], FS, prefix=name.lower())

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)

    plt.title("Primary Noisy Signal")

    plt.plot(primary[:1000])

    idx = 2

    for name, res in results.items():

        plt.subplot(4, 1, idx)

        plt.title(f"{name} Output")

        plt.plot(res['output'][:1000])

        idx += 1

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_offline_demo()