import numpy as np
import scipy.signal as signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.evaluate import run_full_evaluation

class SequenceTester:
    def __init__(self, fs=4000, duration=0.5):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.sequences = []
        self._generate_catalog()

    def _generate_catalog(self):
        """Generate distinct test sequences with reference scale metadata."""

        def get_mod(syms, sps, constellation):
            sig_len = len(self.t)
            n_syms = len(syms)
            train = np.zeros(sig_len, dtype=complex)
            indices = np.arange(0, n_syms * sps, sps)
            indices = indices[indices < sig_len]
            train[indices] = constellation[syms[:len(indices)]]
            pulse = np.ones(sps)
            baseband = np.convolve(train, pulse, mode='full')[:sig_len]
            return baseband

        mod_schemes = [
            ('BPSK', [1, -1]),
            ('QPSK', [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]),
            ('16QAM', [x + 1j * y for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]])
        ]
        rates = [50, 200]
        carriers = [500, 1200]

        for name, const in mod_schemes:
            const = np.array(const) / np.max(np.abs(const))
            for rate in rates:
                for fc in carriers:
                    sps = int(self.fs / rate)
                    ideal_win = sps
                    num_syms = int(self.fs * self.duration / sps) + 10
                    rand_syms = np.random.randint(0, len(const), num_syms)
                    bb = get_mod(rand_syms, sps, const)
                    pb = np.real(bb * np.exp(1j * 2 * np.pi * fc * self.t))
                    self.sequences.append((f"{name} (Rate={rate}, Fc={fc})", pb, ideal_win * 8, ideal_win * 2))

        m_t = np.sin(2 * np.pi * 5 * self.t)
        self.sequences.append(("AM (Depth=0.5)", (1 + 0.5 * m_t) * np.cos(2 * np.pi * 1000 * self.t), 128, 64))
        self.sequences.append(("AM (Depth=1.0)", (1 + 1.0 * m_t) * np.cos(2 * np.pi * 1000 * self.t), 128, 64))

        phi1 = 2 * np.pi * 50 * np.cumsum(m_t) / self.fs
        self.sequences.append(("FM (Dev=50Hz)", np.cos(2 * np.pi * 1000 * self.t + phi1), 64, 64))
        self.sequences.append(("Chirp Linear (0-2k)", signal.chirp(self.t, 0, 1.0, 2000, method='linear'), 128, 32))
        self.sequences.append(("Chirp Quad (0-500)", signal.chirp(self.t, 0, 1.0, 500, method='quadratic'), 256, 64))

        impulse_sig = np.zeros_like(self.t)
        impulse_sig[np.linspace(0, len(self.t) - 1, 10).astype(int)] = 10.0
        self.sequences.append(("Dirac Train (10 pulses)", impulse_sig, 32, 8))

        step_sig = np.sign(np.sin(2 * np.pi * 2 * self.t))
        self.sequences.append(("Square Wave (2Hz)", step_sig, 64, 32))

        burst = np.zeros_like(self.t)
        mask = (self.t > 0.2) & (self.t < 0.3)
        burst[mask] = np.sin(2 * np.pi * 800 * self.t[mask])
        mask = (self.t > 0.6) & (self.t < 0.7)
        burst[mask] = np.sin(2 * np.pi * 800 * self.t[mask])
        mask = (self.t > 1.1) & (self.t < 1.2)
        burst[mask] = np.sin(2 * np.pi * 800 * self.t[mask])
        self.sequences.append(("Tone Bursts (800Hz)", burst, 128, 32))

    def run_all(self):
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic')
        os.makedirs(out_dir, exist_ok=True)
        
        # We define a standard FFT setup for synthetic tests
        n_fft_univ = int(self.fs * 0.128)  # about 128ms context
        hop_size = 16                      # dense grid for good reconstruction
        beta = 0.4
        
        for name, sig, ideal_w, ideal_hop in self.sequences:
            try:
                run_full_evaluation(sig, self.fs, name, out_dir, n_fft=n_fft_univ, hop_size=hop_size, beta=beta)
            except Exception as e:
                import traceback
                print(f"Error in {name}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    tester = SequenceTester(fs=4000, duration=1.5)
    tester.run_all()
