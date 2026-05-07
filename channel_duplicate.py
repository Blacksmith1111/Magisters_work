import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal as sig
from scipy.signal import fftconvolve
from numba import njit

# ==========================================
# НИЗКОУРОВНЕВЫЕ БЫСТРЫЕ ЯДРА (NUMBA C-CODE)
# ==========================================
@njit(cache=True, nogil=True)
def fast_awgn(signal, snr_dB):
    snr_linear = 10.0 ** (snr_dB / 10.0)
    power = 0.0
    for i in range(len(signal)):
        power += signal[i].real**2 + signal[i].imag**2
    power /= len(signal)
    noise_std = np.sqrt((power / snr_linear) / 2.0)
    out = np.zeros(len(signal), dtype=np.complex128)
    for i in range(len(signal)):
        out[i] = signal[i] + (np.random.normal(0.0, noise_std) + 1j * np.random.normal(0.0, noise_std))
    return out

@njit(cache=True, nogil=True)
def fast_downconversion(passband_signal, Fc, Fs):
    N = len(passband_signal)
    out = np.zeros(N, dtype=np.complex128)
    factor = -2j * np.pi * Fc / Fs
    for i in range(N):
        out[i] = passband_signal[i] * np.exp(factor * i)
    return out

@njit(cache=True, nogil=True)
def fast_quantizer_core(signal, resolution, gain, inl_en, inl_array):
    N = len(signal)
    out = np.zeros(N, dtype=np.complex128)
    left_border = -(1 << (resolution - 1))
    right_border = (1 << (resolution - 1)) - 1
    
    for i in range(N):
        val = signal[i] * gain
        r = round(val.real)
        im = round(val.imag)
        if r < left_border: r = left_border
        elif r > right_border: r = right_border
        if im < left_border: im = left_border
        elif im > right_border: im = right_border
        
        if inl_en > 0:
            r += inl_array[int(r - left_border)]
            im += inl_array[int(im - left_border)]
            
        out[i] = r + 1j * im
    return out

@njit(cache=True, nogil=True)
def fast_demod_core(symbols, constellation, bits_array, bits_per_symbol):
    N = len(symbols)
    M = len(constellation)
    out = np.zeros(N * bits_per_symbol, dtype=np.int8)
    for i in range(N):
        min_dist = 1e9
        best_idx = 0
        for j in range(M):
            d_real = symbols[i].real - constellation[j].real
            d_imag = symbols[i].imag - constellation[j].imag
            dist = d_real**2 + d_imag**2
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        for b in range(bits_per_symbol):
            out[i * bits_per_symbol + b] = bits_array[best_idx, b]
    return out

@njit(cache=True, nogil=True)
def _nlms_core_fast(rx_symbols, constellation, initial_gain, num_taps, mu):
    N = len(rx_symbols)
    w = np.zeros(num_taps, dtype=np.complex128)
    delay = num_taps // 2
    w[delay] = initial_gain + 0j
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
            min_dist = 1e9
            best_idx = 0
            for k in range(len(constellation)):
                d_real = y.real - constellation[k].real
                d_imag = y.imag - constellation[k].imag
                dist = d_real**2 + d_imag**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = k
            d = constellation[best_idx]
            e = d - y
            buffer_power = 0.0
            for j in range(num_taps):
                buffer_power += buffer[j].real**2 + buffer[j].imag**2
            step = mu / (buffer_power + 1e-8)
            for j in range(num_taps):
                w[j] = w[j] + step * e * np.conj(buffer[j])
    return out_symbols

# ==========================================
# ВЫСОКОУРОВНЕВЫЕ ФУНКЦИИ И КЛАССЫ
# ==========================================
def INL(full_scale: np.ndarray, lsb_amplitude: float, plt_en: bool = 0) -> np.ndarray:
    inl_vals = lsb_amplitude * np.sin(2 * np.pi * (full_scale - full_scale[0]) / len(full_scale))
    if plt_en:
        plt.figure(figsize=(10, 4))
        plt.plot(full_scale, inl_vals)
        plt.grid(True)
        plt.title(f'INL Transfer characteristic, max amplitude = {lsb_amplitude}')
        plt.xlabel('Input level [LSB]')
        plt.ylabel('Deviation [LSB]')
        plt.show()
    return inl_vals

def quantizer(signal: np.ndarray, resolution: int, gain: float, inl_en: float = 0):
    left_border, right_border = int(-(2**resolution) / 2), int(2**resolution / 2 - 1)
    if inl_en > 0:
        full_scale = np.arange(left_border, right_border + 1, 1)
        inl_array = INL(full_scale, lsb_amplitude=inl_en, plt_en=0)
    else:
        inl_array = np.zeros(1, dtype=np.float64)
    return fast_quantizer_core(signal, resolution, gain, inl_en, inl_array)

def rms_calc(signal: np.ndarray) -> float:
    return np.sqrt(np.mean(np.abs(signal)**2))

def ber_calc(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return np.sum(tx_bits != rx_bits) / len(tx_bits)

def nmse_calc(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    return np.sum(np.abs(tx_symbols - rx_symbols)**2) / np.sum(np.abs(tx_symbols)**2)

def dd_lms_equalizer(rx_symbols, qam_obj, num_taps=31, mu=0.05):
    constellation = np.array(qam_obj.constellation, dtype=np.complex128)
    ideal_rms = np.sqrt(np.mean(np.abs(constellation)**2))
    current_rms = np.sqrt(np.mean(np.abs(rx_symbols)**2))
    initial_gain = ideal_rms / current_rms if current_rms > 0 else 1.0
    out_symbols = _nlms_core_fast(np.array(rx_symbols, dtype=np.complex128), constellation, initial_gain, num_taps, mu)
    delay = num_taps // 2
    return np.roll(out_symbols, -delay)

def apply_fixed_lpf(signal, cutoff_hz, fs, N=401):
    taps = sig.firwin(N, cutoff_hz, window=('kaiser', 14), fs=fs)
    return fftconvolve(signal, taps, mode="full")

def upsample(signal: np.ndarray, sps: int) -> np.ndarray:
    upsampled = np.zeros(len(signal) * sps, dtype=np.complex128)
    upsampled[::sps] = signal
    return upsampled

def downsample(signal: np.ndarray, sps: int) -> np.ndarray:
    return signal[::sps]

def pulse_shaping(signal: np.ndarray, rolloff: float, span: int, sps: int, Ts: float, Fs: float, plt_en: bool = 0) -> np.ndarray:
    t, h = rrcosfilter(span * sps, rolloff, Ts, Fs)
    return fftconvolve(signal, h, mode="full")

def upconversion(baseband_signal: np.ndarray, Fc: float, Fs: float, plt_en: bool = 0) -> np.ndarray:
    t = np.arange(len(baseband_signal)) / Fs
    return np.real(baseband_signal * np.exp(2j * np.pi * Fc * t))

def get_snr_from_ber(target_ber: float, snr_array: list, ber_array: list) -> float:
    snr_arr, ber_arr = np.array(snr_array), np.array(ber_array)
    valid_indices = np.where(ber_arr > 0)[0]
    if len(valid_indices) < 2: return np.nan
    valid_snr, valid_ber = snr_arr[valid_indices], ber_arr[valid_indices]
    log_ber, log_target = np.log10(valid_ber), np.log10(target_ber)
    log_ber_asc, snr_asc = log_ber[::-1], valid_snr[::-1]
    return float(np.interp(log_target, log_ber_asc, snr_asc))

class Modulator32QAM:
    def __init__(self):
        self.mapping_table = {
            (0,0,0,0,0): -3+5j, (0,0,0,0,1): -1+5j, (0,0,0,1,0): 1+5j, (0,0,0,1,1): 3+5j,
            (0,0,1,0,0): -5+3j, (0,0,1,0,1): -3+3j, (0,0,1,1,0): -1+3j, (0,0,1,1,1): 1+3j,
            (0,1,0,0,0): 3+3j, (0,1,0,0,1): 5+3j, (0,1,0,1,0): -5+1j, (0,1,0,1,1): -3+1j,
            (0,1,1,0,0): -1+1j, (0,1,1,0,1): 1+1j, (0,1,1,1,0): 3+1j, (0,1,1,1,1): 5+1j,
            (1,0,0,0,0): -5-1j, (1,0,0,0,1): -3-1j, (1,0,0,1,0): -1-1j, (1,0,0,1,1): 1-1j,
            (1,0,1,0,0): 3-1j, (1,0,1,0,1): 5-1j, (1,0,1,1,0): -5-3j, (1,0,1,1,1): -3-3j,
            (1,1,0,0,0): -1-3j, (1,1,0,0,1): 1-3j, (1,1,0,1,0): 3-3j, (1,1,0,1,1): 5-3j,
            (1,1,1,0,0): -3-5j, (1,1,1,0,1): -1-5j, (1,1,1,1,0): 1-5j, (1,1,1,1,1): 3-5j
        }
        self.bits_tuples = list(self.mapping_table.keys())
        self.constellation = np.array(list(self.mapping_table.values()), dtype=np.complex128)
        self.bits_array = np.array(self.bits_tuples, dtype=np.int8)

    def modulate(self, bits):
        reshaped_bits = np.reshape(bits, (-1, 5))
        symbols = np.zeros(len(reshaped_bits), dtype=np.complex128)
        for i, bit_group in enumerate(reshaped_bits):
            symbols[i] = self.mapping_table[tuple(bit_group)]
        return symbols

    def demodulate(self, symbols, type='hard'):
        return fast_demod_core(np.array(symbols, dtype=np.complex128), self.constellation, self.bits_array, 5)