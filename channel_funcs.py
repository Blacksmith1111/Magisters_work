import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal as sig


def signal_plot(signal: complex, title:str) -> None:
    plt.figure(1)
    plt.plot(signal)
    plt.title(title)
    plt.grid()
    plt.show()

def spectrum_plot(signal: complex, Fs: float, title: str) -> None:
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / Fs))
    
    plt.figure(30)
    plt.plot(freqs, 20 * np.log10(np.abs(spectrum)))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.title(title)
    plt.show()
    
def constellation_plot(modulated_signal: complex, mod_order: int) -> None:
    x = modulated_signal.real
    y = modulated_signal.imag

    plt.figure(1)
    plt.scatter(x, y)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.title(f'Constellation {mod_order} QAM')
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
    filter_len = filter_span * sps + 1 # in samples 
    print(f'Filter len in samples: {filter_len}')
    time_stamps, h = rrcosfilter(filter_len, alpha = rolloff, Ts=Ts, Fs=Fs)

    if normaliztion == 'L2':
        h = h / np.sqrt(np.sum(h**2))  # L2 normalization
    else:
        h = h / np.sum(h)  # L1 normalization

    shaped_signal = np.convolve(upsampled_signal, h, mode = 'full')
    return shaped_signal

def ber_calc(initial_bits, final_bits) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)

def quantizer(signal: complex, resolution: int):
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
    ### INL adding
    INL_vals = INL(DAC_rng)
    I_quantized += INL_vals[DAC_indicies_I]
    Q_quantized += INL_vals[DAC_indicies_Q]
    quantized_signal = I_quantized + 1j * Q_quantized
    return quantized_signal, scaling_factor_dac

def ADC(signal, adc_bits, dac_bits):
    scaling_factor_adc = 2**(adc_bits - dac_bits)
    scaled_signal = signal * scaling_factor_adc
    return scaled_signal, scaling_factor_adc

def upconversion(baseband_signal, Fc, Fs):
    t = np.arange(len(baseband_signal)) / Fs
    passband_signal_complex = baseband_signal * np.exp(2 * np.pi * Fc * t * 1j)
    passband_signal_real = passband_signal_complex
    return passband_signal_real

def downconversion(passband_signal, Fc, Fs, B, filterEn = True):
    t = np.arange(len(passband_signal)) / Fs
    mixed_signal_complex = passband_signal * np.exp(-2 * np.pi * Fc * t * 1j)
    if filterEn:
        ### Filter applying
        LPF_len = 65
        Fc_lpf = B / 2 * 1.1
        nyquist_rate = Fs / 2.0
        normalized_cutoff = Fc_lpf / nyquist_rate
        lpf_h = sig.firwin(LPF_len, normalized_cutoff, pass_zero='lowpass')
        spectrum_plot(lpf_h, Fs, title = 'Test LPF')
        mixed_signal_complex = np.convolve(mixed_signal_complex, lpf_h, mode = 'full')
        spectrum_plot(mixed_signal_complex, Fs, title = 'After LPF test')

    return mixed_signal_complex 

def INL(full_scale):
    INL_vals = 0.8 * np.sin(2 * np.pi * (full_scale - full_scale[0]) / (len(full_scale)))  # INL in LSB
    plt.figure()
    plt.title('INL values')
    plt.plot(INL_vals)
    plt.grid()
    plt.show()
    return INL_vals