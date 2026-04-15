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
    
def apply_fixed_lpf(signal, cutoff_hz, fs, N=401):
    # Прямое создание фильтра без автоматического подбора порядка
    taps = sig.firwin(N, cutoff_hz, window=('kaiser', 14), fs=fs)
    return np.convolve(signal, taps, mode="full")

def apply_lowpass_filter(signal, f_pass, f_stop, fs, attenuation_db=60):
    """
    Создает ФНЧ с заданным подавлением и переходной полосой.
    f_pass: частота, до которой коэффициент передачи = 1 (Гц)
    f_stop: частота, с которой начинается подавление (Гц)
    fs: частота дискретизации (Гц)
    attenuation_db: требуемое подавление в полосе заграждения (дБ)
    """
    # 1. Ширина переходной полосы в нормированных единицах (0..1, где 1 - частота Найквиста)
    nyq = fs / 2
    width = (f_stop - f_pass) / nyq
    
    # 2. Определение порядка фильтра (N) и параметра бета для окна Кайзера
    # Позволяет получить заданное подавление при заданной ширине перехода
    N, beta = sig.kaiserord(attenuation_db, width)
    
    # Делаем N нечетным для удобства (симметричный FIR фильтр)
    if N % 2 == 0:
        N += 1
        
    # 3. Расчет коэффициентов (taps) фильтра
    # Частота среза берется как середина переходной полосы
    cutoff = (f_pass + f_stop) / 2
    taps = sig.firwin(N, cutoff, window=('kaiser', beta), fs=fs)
    spectrum_plot(taps, fs, title = "LPF for upsampled", plt_en = 1)
    filtered = np.convolve(signal, taps, mode="full")
    spectrum_plot(filtered, fs, title = "Upsampled shaped after LPF", plt_en = 1)
    
    return filtered

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


def constellation_plot(modulated_signal: np.ndarray, mod_order: int) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(modulated_signal.real, modulated_signal.imag, s=5)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(True)
    plt.title(f"Constellation {mod_order} QAM")
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
    inl_vals = (
        1.42
        * lsb_amplitude
        * np.sin(2 * np.pi * (full_scale - full_scale[0]) / len(full_scale))
    )

    if plt_en:
        plt.figure(figsize=(8, 3))
        plt.plot(full_scale, inl_vals)
        plt.xlabel("DAC Input Code")
        plt.ylabel("INL (LSB)")
        plt.title(f"INL Profile (Max = {lsb_amplitude} LSB)")
        plt.grid(True)
        plt.show()

    return inl_vals


def quantizer(
    signal: np.ndarray, resolution: int, INL_en: int = 1, lsb_amplitude: float = 2.0
):
    dac_rng = np.arange(-(2**resolution) / 2, 2**resolution / 2, 1)
    scaling_factor_dac = max(np.max(np.abs(signal.real)), np.max(np.abs(signal.imag)))

    signal_normalized = signal * scaling_factor_dac

    # Searching fot the nearest level
    i_indices = np.argmin(
        np.abs(signal_normalized.real[:, None] - dac_rng[None, :]), axis=1
    )
    q_indices = np.argmin(
        np.abs(signal_normalized.imag[:, None] - dac_rng[None, :]), axis=1
    )

    i_quantized = dac_rng[i_indices]
    q_quantized = dac_rng[q_indices]

    # INL adding
    if INL_en:
        inl_vals = INL(dac_rng, lsb_amplitude=lsb_amplitude)
        i_quantized += inl_vals[i_indices]
        q_quantized += inl_vals[q_indices]

    return i_quantized + 1j * q_quantized, scaling_factor_dac


def ADC1(signal: np.ndarray, adc_bits: int, dac_bits: int):
    scaling_factor_adc = 2 ** (adc_bits - dac_bits)
    return signal * scaling_factor_adc, scaling_factor_adc


def ADC(signal: np.ndarray, adc_bits: int, dac_bits: int):
    scaling_factor_adc = 2 ** (adc_bits - dac_bits)
    scaled_signal = signal * scaling_factor_adc

    quantized_signal = np.round(scaled_signal)

    limit = 2 ** (adc_bits - 1)
    final_signal = np.clip(quantized_signal, -limit, limit - 1)

    return final_signal, scaling_factor_adc


def upconversion(baseband_signal: np.ndarray, Fc: float, Fs: float, plt_en : bool = 1) -> np.ndarray:
    t = np.arange(len(baseband_signal)) / Fs
    passband_signal = baseband_signal * np.exp(2j * np.pi * Fc * t)
    if plt_en:
        spectrum_plot(baseband_signal, Fs = Fs, title = 'Baseband signal spectrum', plt_en = 1)
        spectrum_plot(passband_signal, Fs = Fs, title = 'Passband signal spectrum', plt_en = 1)
    return passband_signal


def downconversion(
    passband_signal: np.ndarray, Fc: float, Fs: float, B: float, filterEn: bool = True
) -> np.ndarray:
    t = np.arange(len(passband_signal)) / Fs
    mixed = passband_signal * np.exp(-2j * np.pi * Fc * t)

    if filterEn:
        lpf_len = 65
        fc_lpf = B / 2 * 1.1
        lpf_h = sig.firwin(lpf_len, fc_lpf / (Fs / 2), pass_zero="lowpass")

        spectrum_plot(lpf_h, Fs, title="Test LPF")
        mixed = np.convolve(mixed, lpf_h, mode="full")
        spectrum_plot(mixed, Fs, title="After LPF test")

    return mixed


def qam_constellation_rms_calc(mod_order):
    axis_vals_num = np.sqrt(mod_order)
    vals = np.arange(-2 * axis_vals_num / 2 + 1, 2 * axis_vals_num / 2 + 1, 2)
    rms = np.sqrt(np.mean(vals**2) * 2)
    return rms
