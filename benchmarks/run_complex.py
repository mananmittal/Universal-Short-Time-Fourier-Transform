import numpy as np
import scipy.signal as signal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.evaluate import run_full_evaluation

class ComplexTester:
    def __init__(self, fs=4000, duration=15.0):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.sequences = []
        self._generate_catalog()

    def _generate_catalog(self):
        """Generates 20 distinct, extreme-conflict test sequences over a long duration."""
        
        # Helper for random diracs
        def random_diracs(n_pulses, amplitude=10.0):
            d = np.zeros_like(self.t)
            idx = np.random.choice(len(self.t), n_pulses, replace=False)
            d[idx] = amplitude
            return d
            
        # 1. Continuous Sine + Random Diracs
        sine1 = np.sin(2 * np.pi * 50 * self.t)
        diracs1 = random_diracs(150, amplitude=5.0)
        self.sequences.append(("Continuous Sine + Random Diracs", sine1 + diracs1))

        # 2. Parallel Sweeps + AM Radar Pulses
        chirp_up = signal.chirp(self.t, 100, self.duration, 800)
        chirp_down = signal.chirp(self.t, 800, self.duration, 100)
        am_pulses = np.zeros_like(self.t)
        for i in range(100):
            mask = (self.t > i*0.15) & (self.t < i*0.15 + 0.02)
            am_pulses[mask] = np.sin(2 * np.pi * 1500 * self.t[mask])
        self.sequences.append(("Parallel Sweeps + AM Radar Pulses", chirp_up + chirp_down + 2*am_pulses))

        # 3. Deep FM + Mechanical Clicks
        fm_mod = np.sin(2 * np.pi * 2 * self.t)
        deep_fm = np.sin(2 * np.pi * 400 * self.t + 100 * np.cumsum(fm_mod) / self.fs)
        clicks = random_diracs(200, amplitude=8.0)
        self.sequences.append(("Deep FM + Mechanical Clicks", deep_fm + clicks))

        # 4. The Heisenberg Grid
        tone1 = np.sin(2 * np.pi * 500 * self.t)
        tone2 = np.sin(2 * np.pi * 510 * self.t)
        imp1 = np.zeros_like(self.t); imp1[int(self.fs*5.0)] = 20
        imp2 = np.zeros_like(self.t); imp2[int(self.fs*5.0) + int(self.fs*0.005)] = 20
        imp3 = np.zeros_like(self.t); imp3[int(self.fs*10.0)] = 20
        imp4 = np.zeros_like(self.t); imp4[int(self.fs*10.0) + int(self.fs*0.005)] = 20
        self.sequences.append(("The Heisenberg Grid", tone1 + tone2 + imp1 + imp2 + imp3 + imp4))

        # 5. Steady Harmonics + Noise Bursts
        harmonics = sum(np.sin(2 * np.pi * f * self.t) for f in [200, 400, 600])
        noise_bursts = np.zeros_like(self.t)
        for i in range(50):
            mask = (self.t > i*0.3 + 0.1) & (self.t < i*0.3 + 0.11)
            noise_bursts[mask] = np.random.randn(np.sum(mask)) * 3
        self.sequences.append(("Steady Harmonics + Noise Bursts", harmonics + noise_bursts))

        # 6. Rapid FSK + Slow Vibrato
        fsk = np.zeros_like(self.t)
        fsk_freqs = [300, 1200]
        for i in range(len(self.t)):
            freq = fsk_freqs[(i // int(self.fs * 0.05)) % 2]
            fsk[i] = np.sin(2 * np.pi * freq * self.t[i])
        slow_vib = np.sin(2 * np.pi * 700 * self.t + 10 * np.cumsum(np.sin(2 * np.pi * 1 * self.t)) / self.fs)
        self.sequences.append(("Rapid FSK + Slow Vibrato", fsk + slow_vib))

        # 7. Glockenspiel Synthesis
        glock = np.zeros_like(self.t)
        for strike_t in np.linspace(0.2, self.duration - 1.0, 15):
            t_decay = self.t[self.t >= strike_t] - strike_t
            glock[self.t >= strike_t] += np.exp(-3 * t_decay) * np.sin(2 * np.pi * 1500 * t_decay)
            glock[int(strike_t * self.fs)] += 10.0 
        self.sequences.append(("Glockenspiel Synthesis", glock))

        # 8. Amplitude Modulated Chirp
        slow_chirp = signal.chirp(self.t, 200, self.duration, 1000)
        fast_am = np.sign(np.sin(2 * np.pi * 20 * self.t))
        fast_am[fast_am < 0] = 0
        self.sequences.append(("Amplitude Modulated Chirp", slow_chirp * fast_am))

        # 9. Stepped Glissando (Staircase)
        step_gliss = np.zeros_like(self.t)
        base_f = 300
        for i in range(len(self.t)):
            freq = base_f + 50 * int(self.t[i] * 5) # Jumps every 0.2s
            step_gliss[i] = np.sin(2 * np.pi * freq * self.t[i])
        self.sequences.append(("Stepped Glissando (Staircase)", step_gliss))

        # 10. BPSK + Continuous Carrier
        carrier = np.sin(2 * np.pi * 400 * self.t)
        bpsk = np.zeros_like(self.t)
        phase = 0
        for i in range(len(self.t)):
            if i % int(self.fs * 0.1) == 0:
                phase += np.pi * np.random.choice([0, 1])
            bpsk[i] = np.sin(2 * np.pi * 900 * self.t[i] + phase)
        self.sequences.append(("BPSK + Continuous Carrier", carrier + bpsk))

        # 11. Narrowband Noise + Broadband Spikes
        noise = np.random.randn(len(self.t))
        b, a = signal.butter(4, [490 / (self.fs/2), 510 / (self.fs/2)], btype='band')
        nb_noise = signal.filtfilt(b, a, noise) * 10
        spikes = random_diracs(100, 15.0)
        self.sequences.append(("Narrowband Noise + Broadband Spikes", nb_noise + spikes))

        # 12. Log Sweep + Droplets
        log_sweep = signal.chirp(self.t, 50, self.duration, 1000, method='logarithmic')
        droplets = np.zeros_like(self.t)
        for d_t in np.linspace(0.1, self.duration - 0.5, 30):
            mask = (self.t > d_t) & (self.t < d_t + 0.05)
            if np.any(mask):
                t_local = self.t[mask] - d_t
                droplets[mask] = np.sin(2 * np.pi * (1500 - 10000 * t_local) * t_local) * np.exp(-20 * t_local)
        self.sequences.append(("Log Sweep + Droplets", log_sweep + 2 * droplets))

        # 13. Multi-Tone Burst Train
        chord = np.sin(2 * np.pi * 400 * self.t) + np.sin(2 * np.pi * 500 * self.t) + np.sin(2 * np.pi * 600 * self.t)
        burst_train = np.zeros_like(self.t)
        for i in range(150):
            mask = (self.t > i*0.1) & (self.t < i*0.1 + 0.02)
            burst_train[mask] = chord[mask]
        self.sequences.append(("Multi-Tone Burst Train", burst_train))

        # 14. Doppler Pass-by + Clutter
        v = 30  # m/s
        f0 = 1000
        c = 343
        distance = np.sqrt(10**2 + (v * (self.t - self.duration/2))**2)
        f_doppler = f0 * c / (c + v * (v * (self.t - self.duration/2)) / distance)
        doppler = np.sin(2 * np.pi * np.cumsum(f_doppler) / self.fs)
        clutter = random_diracs(300, 2.0)
        self.sequences.append(("Doppler Pass-by + Clutter", doppler + clutter))

        # 15. Spectral Stutter (Chopper)
        pure_tone = np.sin(2 * np.pi * 600 * self.t)
        chopper = np.sign(np.sin(2 * np.pi * 20 * self.t))
        chopper[chopper < 0] = 0
        self.sequences.append(("Spectral Stutter (Chopper)", pure_tone * chopper))

        # 16. Extreme Bandwidth Gap
        sub_bass = np.sin(2 * np.pi * 15 * self.t) * 2.0
        fast_pulse = np.zeros_like(self.t)
        for i in range(300):
            mask = (self.t > i*0.05) & (self.t < i*0.05 + 0.005)
            fast_pulse[mask] = np.sin(2 * np.pi * 1500 * self.t[mask])
        self.sequences.append(("Extreme Bandwidth Gap", sub_bass + fast_pulse))

        # 17. Swept Harmonic Stack + Pulses
        f_base = np.linspace(100, 300, len(self.t))
        stack = np.zeros_like(self.t)
        for h in range(1, 6):
            stack += np.sin(2 * np.pi * h * np.cumsum(f_base) / self.fs)
        pulses = random_diracs(150, 8.0)
        self.sequences.append(("Swept Harmonic Stack + Pulses", stack + pulses))

        # 18. Dual Antagonistic Chirps
        slow_c = signal.chirp(self.t, 200, self.duration, 400)
        fast_c = np.zeros_like(self.t)
        for i in range(int(self.duration / 0.3)):
            mask = (self.t >= i*0.3) & (self.t < (i+1)*0.3)
            if np.any(mask):
                fast_c[mask] = signal.chirp(self.t[mask], 1000, self.t[mask][-1], 1800)
        self.sequences.append(("Dual Antagonistic Chirps", slow_c + fast_c))

        # 19. Triangle FM + Square AM
        tri_mod = signal.sawtooth(2 * np.pi * 2 * self.t, 0.5)
        tri_fm = np.sin(2 * np.pi * 800 * self.t + 200 * np.cumsum(tri_mod) / self.fs)
        sq_am = signal.square(2 * np.pi * 8 * self.t)
        sq_am[sq_am < 0] = 0
        self.sequences.append(("Triangle FM + Square AM", tri_fm * sq_am))

        # 20. The Absolute Chaos
        chaos = np.sin(2 * np.pi * 300 * self.t) + np.sin(2 * np.pi * 320 * self.t) # Two close tones
        chaos += signal.chirp(self.t, 1500, self.duration, 500) # Descending chirp
        chaos += random_diracs(80, 15.0) # Impulses
        self.sequences.append(("The Absolute Chaos", chaos))

    def run_all(self):
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'complex')
        os.makedirs(out_dir, exist_ok=True)
        
        # We define a standard FFT setup for synthetic tests
        n_fft_univ = int(self.fs * 0.512)  # 128ms context
        hop_size = 128                      # dense grid for good reconstruction
        beta = 1
        
        for name, sig in self.sequences:
            try:
                run_full_evaluation(sig, self.fs, name, out_dir, n_fft=n_fft_univ, hop_size=hop_size, beta=beta)
            except Exception as e:
                import traceback
                print(f"Error in {name}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    tester = ComplexTester(fs=4000, duration=15.0)
    tester.run_all()
