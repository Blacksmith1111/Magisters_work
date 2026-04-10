import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn
import channel_funcs as cf
from dataclasses import dataclass


@dataclass
class SimConfig:
    """Хранит все параметры симуляции в одном месте"""

    bits_num: int = 1_000_002
    mod_order: int = 64
    fs: float = 10e3
    sps: int = 10
    rolloff: float = 0.25
    filter_span: int = 10
    dac_bits: int = 5
    adc_bits: int = 8

    # Флаги
    en_adc: int = 1
    en_quantizer: int = 1
    en_up_down_conv: int = 1
    en_noise: int = 0
    inl_en: int = 1
    lsb_ampl: float = 2.0

    @property
    def fc(self) -> float:
        return self.fs / 4

    @property
    def ts(self) -> float:
        return 1 / (self.fs / self.sps)


def run_transmitter(bits: np.ndarray, qam: mod.QAMModem, cfg: SimConfig):
    """Формирование базового сигнала (TX)"""
    symbol_signal = qam.modulate(bits)
    up_signal = cf.upsample(symbol_signal, cfg.sps)

    shaped_signal = cf.pulse_shaping(
        up_signal,
        rolloff=cfg.rolloff,
        filter_span=cfg.filter_span,
        sps=cfg.sps,
        Fs=cfg.fs,
        Ts=cfg.ts,
        normaliztion="L2",
    )
    return symbol_signal, up_signal, shaped_signal


def apply_hardware_and_channel(shaped_signal: np.ndarray, snr_db: int, cfg: SimConfig):
    """Имитация железа (ЦАП) и эфира (Перенос частот, Шум)"""
    # 1. DAC / Quantizer
    if cfg.en_quantizer:
        quantized_signal, _ = cf.quantizer(
            shaped_signal,
            resolution=cfg.dac_bits,
            INL_en=cfg.inl_en,
            lsb_amplitude=cfg.lsb_ampl,
        )
    else:
        quantized_signal = shaped_signal

    # 2. RF & Channel
    if cfg.en_up_down_conv:
        # Upconversion
        passband_signal = cf.upconversion(quantized_signal, Fs=cfg.fs, Fc=cfg.fc)

        # AWGN
        noisy_signal = (
            awgn(passband_signal, snr_dB=snr_db) if cfg.en_noise else passband_signal
        )

        # Downconversion
        baseband_signal = cf.downconversion(noisy_signal, Fs=cfg.fs, Fc=cfg.fc, B=1200)
    else:
        # Если переноса нет, добавляем шум напрямую к baseband (если включен)
        baseband_signal = (
            awgn(quantized_signal, snr_dB=snr_db) if cfg.en_noise else quantized_signal
        )

    return baseband_signal


def run_receiver(baseband_signal: np.ndarray, up_signal: np.ndarray, cfg: SimConfig):
    """Приемник: АЦП, фильтрация, синхронизация и децимация"""
    # 1. ADC
    if cfg.en_adc:
        scaled_signal, _ = cf.ADC(
            baseband_signal, adc_bits=cfg.adc_bits, dac_bits=cfg.dac_bits
        )
    else:
        scaled_signal = baseband_signal

    # 2. Matched Filter
    recovered_signal = cf.pulse_shaping(
        scaled_signal,
        rolloff=cfg.rolloff,
        filter_span=cfg.filter_span,
        sps=cfg.sps,
        Fs=cfg.fs,
        Ts=cfg.ts,
        normaliztion="L2",
    )

    # 3. Time Synchronization
    correlation = sig.correlate(up_signal, recovered_signal, mode="full")
    lags = sig.correlation_lags(len(up_signal), len(recovered_signal), mode="full")
    actual_delay = lags[np.argmax(correlation)]

    rec_sync = recovered_signal[-actual_delay : -actual_delay + len(up_signal)]

    # 4. Downsampling & Rescaling
    downsampled = cf.downsample(rec_sync, cfg.sps)

    constellation_rms = cf.qam_constellation_rms_calc(cfg.mod_order)
    downsampled_norm = (
        downsampled / np.sqrt(np.mean(np.abs(downsampled) ** 2)) * constellation_rms
    )

    return downsampled_norm


def calculate_metrics(
    downsampled_norm: np.ndarray,
    symbol_signal: np.ndarray,
    bits: np.ndarray,
    qam: mod.QAMModem,
    bits_num: int,
):
    """Расчет ошибок NMSE и BER"""
    nmse = 10 * np.log10(
        np.sum(np.abs(downsampled_norm - symbol_signal) ** 2)
        / np.sum(np.abs(symbol_signal) ** 2)
    )

    demod_bits = qam.demodulate(downsampled_norm, "hard")[:bits_num]
    ber = cf.ber_calc(bits, demod_bits)

    return nmse, ber


def main():
    cfg = SimConfig()
    SNR_vals, BER_vals = [], []

    # Transmitter init (делаем один раз, так как данные от SNR не зависят)
    np.random.seed(100)
    bits = np.random.randint(0, 2, cfg.bits_num)
    qam = mod.QAMModem(cfg.mod_order)

    symbol_signal, up_signal, shaped_signal = run_transmitter(bits, qam, cfg)

    # Цикл по SNR
    for snr in range(15, 30, 1):
        SNR_vals.append(snr)

        # Прохождение канала (включая ЦАП и АЦП)
        baseband_signal = apply_hardware_and_channel(shaped_signal, snr, cfg)

        # Приемник
        downsampled_norm = run_receiver(baseband_signal, up_signal, cfg)

        # Метрики
        nmse, ber = calculate_metrics(
            downsampled_norm, symbol_signal, bits, qam, cfg.bits_num
        )
        BER_vals.append(ber)

        print("-" * 30)
        print(f"SNR:  {snr} dB")
        print(f"NMSE: {nmse:.2f} dB")
        print(f"BER:  {ber:.9f} ")
        print("-" * 30)

    # Построение графиков
    plt.figure(figsize=(8, 5))
    plt.plot(SNR_vals, BER_vals, marker="o", linestyle="-", color="b")
    plt.yscale(
        "log"
    )  # Часто графики BER/SNR удобнее смотреть в логарифмическом масштабе
    plt.grid(True, which="both", ls="--")
    plt.title(f"BER vs SNR (INL: {bool(cfg.inl_en)}, 64-QAM)")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.savefig("With_INL.png")
    plt.show()


if __name__ == "__main__":
    main()
