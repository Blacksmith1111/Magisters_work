import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter, rcosfilter
from scipy import signal as sig


def rc_filter(signal, filter_span, sps, Fs, rolloff, Ts, plt_en = 1, normalization = 'L2'):
    filter_len = filter_span * sps# + 1
    time_stamps, h = rcosfilter(filter_len, alpha=rolloff, Ts=Ts, Fs=Fs)

    if normalization == "L2":
        h = h / np.sqrt(np.sum(np.abs(h)**2))
    else:
        h = h / np.sum(h)
    
    if plt_en:
        plt.figure(1)
        plt.title('RC filter impulse response')
        plt.stem(time_stamps, h, label = 'Impulse response')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close('all')

        spectrum_plot(h, Fs, 'RC filter, h spectrum', plt_en = 1)
    

    return np.convolve(signal, h, mode="full")
    
def apply_fixed_lpf(signal, cutoff_hz, fs, N=401, plt_en = 0):
    taps = sig.firwin(N, cutoff_hz, window=('kaiser', 14), fs=fs)
    taps = taps / np.sqrt(np.sum(np.abs(taps)**2))
    if plt_en:
        spectrum_plot(taps, fs, title = 'LPF frequency characteristics', plt_en = plt_en)
    return np.convolve(signal, taps, mode="full")

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


def constellation_plot(modulated_signal: np.ndarray, mod_order: int, title) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(modulated_signal.real, modulated_signal.imag, s=5)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(True)
    plt.title(f"Constellation {mod_order} QAM")
    if title is not None:
        plt.savefig(title)
    plt.show()


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

    if plt_en:
        plt.figure(1)
        plt.title('RRC filter impulse response')
        plt.stem(time_stamps, h, label = 'Impulse response')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close('all')

    if normaliztion == "L2":
        #h = h / np.sqrt(np.sum(h**2))
        h = h / np.sqrt(np.sum(np.abs(h)**2))
    else:
        h = h / np.sum(h)

    return np.convolve(upsampled_signal, h, mode="full")


def ber_calc(initial_bits: np.ndarray, final_bits: np.ndarray) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)


def INL(full_scale: np.ndarray, lsb_amplitude: float, plt_en: bool = 0) -> np.ndarray:
    # 1.42
    inl_vals = lsb_amplitude * np.sin(2 * np.pi * (full_scale - full_scale[0]) / len(full_scale))

    if plt_en:
        plt.figure(figsize=(8, 3))
        plt.plot(full_scale, inl_vals)
        plt.xlabel("DAC Input Code")
        plt.ylabel("INL (LSB)")
        plt.title(f"INL Profile (Max = {lsb_amplitude} LSB)")
        plt.grid(True)
        plt.show()

    return inl_vals


def quantizer(signal: np.ndarray, resolution: int, gain: float, inl_en: bool = 0):

    left_border, right_border = int(-(2**resolution) / 2), int(2**resolution / 2 - 1)
    scaled_signal = signal * gain
    i_quantized = np.clip(np.round((scaled_signal.real)).astype(np.int32), left_border, right_border)
    q_quantized = np.clip(np.round((scaled_signal.imag)).astype(np.int32), left_border, right_border)
    
    if inl_en:
        full_scale = np.arange(left_border, right_border + 1, 1)
        inl_array = INL(full_scale, lsb_amplitude=2, plt_en=0)
        i_indices = i_quantized - left_border
        q_indices = q_quantized - left_border
        
        i_with_inl = i_quantized.astype(np.float64) + inl_array[i_indices]
        q_with_inl = q_quantized.astype(np.float64) + inl_array[q_indices]
        
        return i_with_inl + 1j * q_with_inl

    return i_quantized + 1j * q_quantized


def ADC1(signal: np.ndarray, adc_bits: int, dac_bits: int):
    scaling_factor_adc = 2 ** (adc_bits - dac_bits)
    return signal * scaling_factor_adc, scaling_factor_adc


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
    axis_vals_num = np.sqrt(mod_order)
    vals = np.arange(-2 * axis_vals_num / 2 + 1, 2 * axis_vals_num / 2 + 1, 2)
    rms = np.sqrt(np.mean(vals**2) * 2)
    return rms
