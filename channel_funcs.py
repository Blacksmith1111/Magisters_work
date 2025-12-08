import numpy as np
import commpy.modulation as mod
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter


def constellation_plot(modulated_signal: complex) -> None:
    x = modulated_signal.real
    y = modulated_signal.imag

    plt.figure(1)
    plt.scatter(x, y)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.show()

def upsample(signal:complex, sps:int) -> complex:
    I = signal.real
    Q = signal.imag
    symbols = I + 1j * Q
    symb_num = len(signal)
    signal_upsampled = np.zeros(symb_num * sps, dtype = complex)
    signal_upsampled[::sps] = symbols
    return signal_upsampled

def downsample(signal: complex, sps: int) -> complex:
    return signal[::sps]

def pulse_shaping(upsampled_signal: complex, rolloff: float, filter_span: int, sps: int, Ts: float, Fs: float, normaliztion:str = 'L2') -> complex:
    filter_len = filter_span * sps * 2 + 1
    time_stamps, h = rrcosfilter(filter_len, alpha = rolloff, Ts=Ts, Fs=Fs)
    if normaliztion == 'L2':
        h = h / np.sqrt(np.sum(h**2))  # L2 normalization
    else:
        h = h / np.sum(h)  # L1 normalization
    shaped_signal = np.convolve(upsampled_signal, h, mode = 'full')

    return shaped_signal

def ber_calc(initial_bits, final_bits) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)

def quantizer(signal, resolution: int, S_max: float):
    ### DAC full-scale range
    DAC_rng = np.arange(-2**resolution / 2, 2**resolution / 2, 1)
    ### Scaling factor
    scaling_factor_dac = max(np.max(np.abs(signal.real)), np.max(np.abs(signal.imag)))
    signal_normalized = signal * scaling_factor_dac
    I_norm, Q_norm = signal_normalized.real, signal_normalized.imag
    DAC_indicies_I = np.argmin(np.abs(I_norm[:, None] - DAC_rng[None, :]), axis = 1)
    DAC_indicies_Q = np.argmin(np.abs(Q_norm[:, None] - DAC_rng[None, :]), axis = 1)
    I_quantized = DAC_rng[DAC_indicies_I]
    Q_quantized = DAC_rng[DAC_indicies_Q]
    return I_quantized + 1j * Q_quantized, scaling_factor_dac

def ADC(signal, adc_bits, dac_bits):
    scaling_factor_adc = 2**(adc_bits - dac_bits)
    scaled_signal = signal * scaling_factor_adc
    return scaled_signal, scaling_factor_adc

def upconversion(baseband_signal, Fc, Fs):
    t = np.arange(len(baseband_signal)) / Fs
    
    passband_signal_complex = baseband_signal * np.exp(2 * np.pi * Fc * t * 1j)
    passband_signal_real = passband_signal_complex
    return passband_signal_real

def downconversion(passband_signal, Fc, Fs):
    t = np.arange(len(passband_signal)) / Fs
    mixed_signal_complex = passband_signal * np.exp(-2 * np.pi * Fc * t * 1j)
    
    return mixed_signal_complex 