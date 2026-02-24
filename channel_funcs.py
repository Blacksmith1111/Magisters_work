import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal as sig


def spectrum_plot(signal: np.ndarray, Fs: float, title: str) -> None:
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / Fs))
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, 20 * np.log10(np.abs(spectrum) + 1e-15))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.title(title)
    plt.show()

def constellation_plot(modulated_signal: np.ndarray, mod_order: int) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(modulated_signal.real, modulated_signal.imag, s=5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.title(f'Constellation {mod_order} QAM')
    plt.show()

def upsample(signal: np.ndarray, sps: int) -> np.ndarray:
    signal_upsampled = np.zeros(len(signal) * sps, dtype=complex)
    signal_upsampled[::sps] = signal
    return signal_upsampled

def downsample(signal: np.ndarray, sps: int) -> np.ndarray:
    return signal[::sps]

def pulse_shaping(upsampled_signal: np.ndarray, rolloff: float, filter_span: int, 
                  sps: int, Ts: float, Fs: float, normaliztion: str = 'L2') -> np.ndarray:
    filter_len = filter_span * sps + 1
    time_stamps, h = rrcosfilter(filter_len, alpha=rolloff, Ts=Ts, Fs=Fs)

    if normaliztion == 'L2':
        h = h / np.sqrt(np.sum(h**2))
    else:
        h = h / np.sum(h)

    return np.convolve(upsampled_signal, h, mode='full')

def ber_calc(initial_bits: np.ndarray, final_bits: np.ndarray) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)


def INL(full_scale: np.ndarray) -> np.ndarray:
    inl_vals = 0.8 * np.sin(2 * np.pi * (full_scale - full_scale[0]) / (len(full_scale)))
    plt.figure(figsize=(8, 3))
    plt.plot(inl_vals)
    plt.title('INL values')
    plt.grid(True)
    plt.show()
    return inl_vals

def quantizer(signal: np.ndarray, resolution: int):
    dac_rng = np.arange(-2**resolution / 2, 2**resolution / 2, 1)
    scaling_factor_dac = max(np.max(np.abs(signal.real)), np.max(np.abs(signal.imag)))
    
    signal_normalized = signal * scaling_factor_dac
    
    # Searching fot the nearest level
    i_indices = np.argmin(np.abs(signal_normalized.real[:, None] - dac_rng[None, :]), axis=1)
    q_indices = np.argmin(np.abs(signal_normalized.imag[:, None] - dac_rng[None, :]), axis=1)
    
    i_quantized = dac_rng[i_indices]
    q_quantized = dac_rng[q_indices]
    
    # INL adding
    inl_vals = INL(dac_rng)
    i_quantized += inl_vals[i_indices]
    q_quantized += inl_vals[q_indices]
    
    return i_quantized + 1j * q_quantized, scaling_factor_dac

def ADC(signal: np.ndarray, adc_bits: int, dac_bits: int):
    scaling_factor_adc = 2**(adc_bits - dac_bits)
    return signal * scaling_factor_adc, scaling_factor_adc

def upconversion(baseband_signal: np.ndarray, Fc: float, Fs: float) -> np.ndarray:
    t = np.arange(len(baseband_signal)) / Fs
    return baseband_signal * np.exp(2j * np.pi * Fc * t)

def downconversion(passband_signal: np.ndarray, Fc: float, Fs: float, B: float, filterEn: bool = True) -> np.ndarray:
    t = np.arange(len(passband_signal)) / Fs
    mixed = passband_signal * np.exp(-2j * np.pi * Fc * t)
    
    if filterEn:
        lpf_len = 65
        fc_lpf = B / 2 * 1.1
        lpf_h = sig.firwin(lpf_len, fc_lpf / (Fs / 2), pass_zero='lowpass')

        spectrum_plot(lpf_h, Fs, title='Test LPF')
        mixed = np.convolve(mixed, lpf_h, mode='full')
        spectrum_plot(mixed, Fs, title='After LPF test')
        
    return mixed