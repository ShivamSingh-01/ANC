import tkinter as tk
from tkinter import ttk

import numpy as np

import sounddevice as sd

import soundfile as sf

from datetime import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure

from filters import LMSFilter, NLMSFilter, RLSFilter

from signal_utils import detect_hum_frequencies


FS = 16000

BLOCK_SIZE = 512


class ANCProcessor:

    def __init__(self, filter_type='LMS'):

        self.filter_type = filter_type

        self.detected_freqs = []

        self.phase = 0

        self.set_filter(filter_type)

    def set_filter(self, filter_type):

        self.filter_type = filter_type

        if filter_type == 'LMS':
            self.filter = LMSFilter(order=64, mu=0.005)

        elif filter_type == 'NLMS':
            self.filter = NLMSFilter(order=64, mu=0.5)

        else:
            self.filter = RLSFilter(order=16,
    lam=0.999,
    delta=100.0)

    def generate_reference(self, freqs, n):

        if not freqs:
            return np.zeros(n)

        t = np.arange(n)

        ref = np.zeros(n)

        for f in freqs:
            ref += np.sin(
                2 * np.pi * f * t / FS + self.phase
            )

        self.phase += 0.1

        return ref / (np.max(np.abs(ref)) + 1e-12)

    def process_block(self, primary):

        self.detected_freqs = detect_hum_frequencies(
            primary,
            FS
        )

        reference = self.generate_reference(
            self.detected_freqs,
            len(primary)
        )

        output = np.zeros_like(primary)

        for i in range(len(primary)):

            output[i] = self.filter.update(
                reference[i],
                primary[i]
            )

        return output


class LiveANCApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Adaptive Noise Cancellation")

        self.processor = ANCProcessor()

        self.running = False

        self.recorded = []

        self.build_ui()

    def build_ui(self):

        top = tk.Frame(self.root)

        top.pack()

        self.filter_var = tk.StringVar(value='LMS')

        combo = ttk.Combobox(
            top,
            textvariable=self.filter_var,
            values=['LMS', 'NLMS', 'RLS']
        )

        combo.pack(side='left')

        combo.bind(
            "<<ComboboxSelected>>",
            self.change_filter
        )

        tk.Button(
            top,
            text="START",
            command=self.start_audio
        ).pack(side='left')

        tk.Button(
            top,
            text="STOP",
            command=self.stop_audio
        ).pack(side='left')

        self.fig = Figure(figsize=(8, 6))

        self.ax1 = self.fig.add_subplot(211)

        self.ax2 = self.fig.add_subplot(212)

        self.line1, = self.ax1.plot(np.zeros(BLOCK_SIZE))

        self.line2, = self.ax2.plot(np.zeros(BLOCK_SIZE))

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.root
        )

        self.canvas.get_tk_widget().pack()

    def change_filter(self, event):

        self.processor.set_filter(
            self.filter_var.get()
        )

    def audio_callback(
        self,
        indata,
        outdata,
        frames,
        time,
        status
    ):

        primary = indata[:, 0]

        output = self.processor.process_block(primary)

        outdata[:, 0] = output.astype(np.float32)

        self.recorded.append(output.copy())

        self.line1.set_data(
            np.arange(len(primary)),
            output
        )

        self.line2.set_data(
            np.arange(len(output)),
            output
        )
        self.ax1.set_xlim(0, len(primary))

        self.ax2.set_xlim(0, len(output))

        self.canvas.draw()

    def start_audio(self):

        self.running = True

        self.stream = sd.Stream(
            samplerate=FS,
            blocksize=BLOCK_SIZE,
            channels=1,
            callback=self.audio_callback
        )

        self.stream.start()

    def stop_audio(self):

        self.running = False

        self.stream.stop()

        self.stream.close()

        audio = np.concatenate(self.recorded)

        filename = datetime.now().strftime(
            "anc_output_%Y%m%d_%H%M%S.wav"
        )

        sf.write(filename, audio, FS)

        print(f"Saved: {filename}")


def run_live_demo():

    root = tk.Tk()

    app = LiveANCApp(root)

    root.mainloop()


if __name__ == "__main__":
    run_live_demo()