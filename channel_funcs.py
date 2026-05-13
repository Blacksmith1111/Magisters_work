import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter, rcosfilter
from scipy import signal as sig
from scipy.signal import fftconvolve
from numba import njit


def apply_fixed_lpf(signal, cutoff_hz, fs, N = 401, plt_en = 0, sps_2 = 20):
    taps = sig.firwin(N, cutoff_hz, window=('kaiser', 14), fs=fs)
    taps *= sps_2
    if plt_en:
        spectrum_plot(taps, fs, title = 'LPF frequency characteristics', plt_en = plt_en)
    return fftconvolve(signal, taps, mode="full")

def spectrum_plot(signal: np.ndarray, Fs: float, title: str, plt_en: bool = 0) -> None:
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / Fs))

    if plt_en:
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, 20 * np.log10(np.abs(spectrum) + 1e-15))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.axvline(x = Fs / 2, color = 'red')
        plt.axvline(x = -Fs / 2, color = 'red')
        plt.grid(True)
        plt.title(title)
        plt.show()


def get_ideal_constellation(mod_order: int) -> np.ndarray:
    if mod_order == 64:
        levels = np.arange(-7, 8, 2)
        X, Y = np.meshgrid(levels, levels)
        return X.flatten() + 1j * Y.flatten()
        
    elif mod_order == 32:
        levels = np.arange(-5, 6, 2)
        X, Y = np.meshgrid(levels, levels)
        points = X.flatten() + 1j * Y.flatten()
        corners_mask = (np.abs(points.real) == 5) & (np.abs(points.imag) == 5)
        return points[~corners_mask]
        
    return None

def constellation_plot(modulated_signal: np.ndarray, mod_order: int, title: str, show_decision_boundaries: bool = True, save_file:str = 'None') -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(modulated_signal.real, modulated_signal.imag, s=5, alpha=0.5, label='Received Signal')
    
    reference_points = get_ideal_constellation(mod_order)
    if reference_points is not None:
        plt.scatter(reference_points.real, reference_points.imag, 
        s=30, color='red', marker='o', edgecolors='black', label=f'Ideal {mod_order}-QAM')
        
    if show_decision_boundaries:
        if mod_order == 64:
            boundaries = np.arange(-6, 7, 2)
        elif mod_order == 32:
            boundaries = np.arange(-4, 5, 2)
        else:
            boundaries = []

        for b in boundaries:
            plt.axvline(b, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            plt.axhline(b, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    
    plt.grid(False)
    if title is not None:
        plt.title(title)
        
    plt.legend(loc='upper right')
    
    axis_limit = 9 if mod_order == 64 else 7
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)
    if save_file is not 'None': 
        plt.savefig(save_file)
    plt.close('all')
    #plt.show()


def upsample(signal: np.ndarray, sps: int) -> np.ndarray:
    signal_upsampled = np.zeros(len(signal) * sps, dtype=complex)
    signal_upsampled[::sps] = signal
    return signal_upsampled

def downsample(signal: np.ndarray, sps: int) -> np.ndarray:
    return signal[::sps]

def pulse_shaping(
    upsampled_signal: np.ndarray,
    rolloff: float,
    filter_span: int,
    sps: int,
    Ts: float,
    Fs: float,
    normaliztion: str = "L2",
    plt_en:bool = 0
) -> np.ndarray:
    filter_len = filter_span * sps# + 1
    time_stamps, h = rrcosfilter(filter_len, alpha=rolloff, Ts=Ts, Fs=Fs)


    if normaliztion == "L2":
        h = h / np.sqrt(np.sum(np.abs(h)**2))
    else:
        h = h / np.sum(h)

    shaped_signal = fftconvolve(upsampled_signal, h, mode="full") #np.convolve(upsampled_signal, h, mode="full")
    if plt_en:
        plt.figure(1)
        plt.title('RRC filter impulse response')
        plt.stem(time_stamps, h, label = 'Impulse response')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(2)
        plt.title('Shaped signal real')
        plt.plot(shaped_signal.real[:-1], marker = 'o')
        plt.grid()
        plt.show()

        plt.figure(3)
        plt.title('Shaped signal imag')
        plt.plot(shaped_signal.imag[:-1], marker = 'o')
        plt.grid()
        plt.show()
        plt.close('all')


    return shaped_signal


def ber_calc(initial_bits: np.ndarray, final_bits: np.ndarray) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)


def INL(full_scale: np.ndarray, lsb_amplitude: float, plt_en: bool = 0) -> np.ndarray:
    
    inl_vals = lsb_amplitude * np.sin(2 * np.pi * (full_scale - full_scale[0]) / (len(full_scale) - 1))

    if plt_en:
        plt.figure(figsize=(8, 3))
        plt.plot(full_scale, inl_vals)
        plt.xlabel("DAC Input Code")
        plt.ylabel("INL (LSB)")
        plt.title(f"INL Profile (Max = {lsb_amplitude} LSB)")
        plt.grid(True)
        plt.show()

    return inl_vals


def quantizer(signal: np.ndarray, resolution: int, gain: float, inl_en: float = 0):

    left_border, right_border = int(-(2**resolution) / 2), int(2**resolution / 2 - 1)
    scaled_signal = signal * gain
    i_quantized = np.clip(np.round((scaled_signal.real)).astype(np.int32), left_border, right_border)
    q_quantized = np.clip(np.round((scaled_signal.imag)).astype(np.int32), left_border, right_border)
    
    if inl_en > 0:

        full_scale = np.arange(left_border, right_border + 1, 1)
        inl_array = INL(full_scale, lsb_amplitude = inl_en, plt_en=0)
        i_indices = i_quantized - left_border
        q_indices = q_quantized - left_border
        
        i_with_inl = i_quantized.astype(np.float64) + inl_array[i_indices]
        q_with_inl = q_quantized.astype(np.float64) + inl_array[q_indices]
        
        return i_with_inl + 1j * q_with_inl

    return i_quantized + 1j * q_quantized

def ADC(signal: np.ndarray, resolution: int, gain: float):
    left_border, right_border = int(-(2**resolution) / 2), int(2**resolution / 2 - 1)
    scaled_signal = signal * gain
    i_quantized, q_quantized = np.clip(np.round((scaled_signal.real)).astype(np.int32), left_border, right_border), np.clip(np.round((scaled_signal.imag)).astype(np.int32), left_border, right_border)
    return i_quantized + 1j * q_quantized


def upconversion(baseband_signal: np.ndarray, Fc: float, Fs: float, plt_en : bool = 1) -> np.ndarray:
    t = np.arange(len(baseband_signal)) / Fs
    passband_signal = baseband_signal * np.exp(2j * np.pi * Fc * t)
    if plt_en:
        spectrum_plot(baseband_signal, Fs = Fs, title = 'Baseband signal spectrum', plt_en = 1)
        spectrum_plot(passband_signal, Fs = Fs, title = 'Passband signal spectrum', plt_en = 1)
    return passband_signal


def downconversion(passband_signal: np.ndarray, Fc: float, Fs: float, plt_en: bool = 1) -> np.ndarray:
    t = np.arange(len(passband_signal)) / Fs
    baseband_signal = passband_signal * np.exp(-2j * np.pi * Fc * t)
    if plt_en:
        spectrum_plot(baseband_signal, Fs = Fs, title = 'Downconverted baseband signal spectrum', plt_en = plt_en)

    return baseband_signal


def qam_constellation_rms_calc(mod_order):
    if mod_order == 64:
        return np.sqrt(42.0)
    elif mod_order == 32:
        return np.sqrt(20.0)
    else:
        raise ValueError("Unsupported modulation order")

def normalize_to_ones(objects, targets):
    norm_val = max(np.max(np.abs(objects)), np.max(np.abs(targets)))
    objects = objects / norm_val
    targets = targets / norm_val
    return objects, targets, norm_val


def denormalize_from_ones(objects_n, targets_n, norm_val):
    return objects_n * norm_val, targets_n * norm_val


class Modulator32QAM:
    def __init__(self):
        self.mapping_table = {
            (0,0,0,0,0): -3+5j, (0,0,0,0,1): -1+5j, (0,0,0,1,1):  1+5j, (0,0,0,1,0):  3+5j,
            (0,0,1,0,0): -5+3j, (0,0,1,0,1): -3+3j, (0,0,1,1,1): -1+3j, (0,0,1,1,0):  1+3j, (0,1,1,1,0):  3+3j, (0,1,1,0,0):  5+3j,
            (0,1,0,0,0): -5+1j, (0,1,0,0,1): -3+1j, (0,1,0,1,1): -1+1j, (0,1,0,1,0):  1+1j, (0,1,1,1,1):  3+1j, (0,1,1,0,1):  5+1j,
            (1,1,0,0,0): -5-1j, (1,1,0,0,1): -3-1j, (1,1,0,1,1): -1-1j, (1,1,0,1,0):  1-1j, (1,1,1,1,1):  3-1j, (1,1,1,0,1):  5-1j,
            (1,0,1,0,0): -5-3j, (1,0,1,0,1): -3-3j, (1,0,1,1,1): -1-3j, (1,0,1,1,0):  1-3j, (1,1,1,1,0):  3-3j, (1,1,1,0,0):  5-3j,
            (1,0,0,0,0): -3-5j, (1,0,0,0,1): -1-5j, (1,0,0,1,1):  1-5j, (1,0,0,1,0):  3-5j
        }

        self.bits_tuples = list(self.mapping_table.keys())
        self.constellation = np.array(list(self.mapping_table.values()))
        
        self.avg_power = np.mean(np.abs(self.constellation)**2)

    def modulate(self, bits):

        reshaped_bits = np.reshape(bits, (-1, 5))
        symbols = np.zeros(len(reshaped_bits), dtype=complex)

        for i, bit_group in enumerate(reshaped_bits):
            symbols[i] = self.mapping_table[tuple(bit_group)]

        return symbols

    def demodulate(self, symbols, type = 'hard'):
        symbols_complex = np.array(symbols)
        points = self.constellation

        distances = np.abs(symbols_complex[:, np.newaxis] - points[np.newaxis, :])
        min_indices = np.argmin(distances, axis=1)

        demodulated_bits = np.array([self.bits_tuples[idx] for idx in min_indices]).flatten()

        return demodulated_bits


@njit(cache = True)
def _nlms_core_fast(rx_symbols, constellation, initial_gain, num_taps, mu):
    N = len(rx_symbols)
    w = np.zeros(num_taps, dtype=np.complex128)
    w[num_taps // 2] = initial_gain + 0j
    
    out_symbols = np.zeros(N, dtype=np.complex128)
    buffer = np.zeros(num_taps, dtype=np.complex128)
    
    for i in range(N):
        for j in range(num_taps - 1, 0, -1):
            buffer[j] = buffer[j - 1]
        buffer[0] = rx_symbols[i]
        
        y = 0j
        for j in range(num_taps):
            y += w[j] * buffer[j]
        out_symbols[i] = y
        
        if i >= num_taps:
            distances = np.abs(constellation - y)
            best_idx = np.argmin(distances)
            d = constellation[best_idx]
            
            e = d - y
            
            buffer_power = 0.0
            for j in range(num_taps):
                buffer_power += np.abs(buffer[j])**2
                
            step = mu / (buffer_power + 1e-8)
            
            for j in range(num_taps):
                w[j] = w[j] + step * e * np.conj(buffer[j])
                
    return out_symbols


def dd_lms_equalizer(rx_symbols, qam_obj, num_taps=21, mu=0.05):
    constellation = np.array(qam_obj.constellation, dtype=np.complex128)

    ideal_rms = np.sqrt(np.mean(np.abs(constellation)**2))
    current_rms = np.sqrt(np.mean(np.abs(rx_symbols)**2))
    initial_gain = ideal_rms / current_rms if current_rms > 0 else 1.0

    out_symbols = _nlms_core_fast(
        np.array(rx_symbols, dtype=np.complex128), 
        constellation, 
        initial_gain, 
        num_taps, 
        mu
    )
    
    delay = num_taps // 2
    out_aligned = np.roll(out_symbols, -delay)
    
    return out_aligned


