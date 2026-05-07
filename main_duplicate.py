import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
import channel_duplicate as cf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Закомментировано до момента, когда ты решишь вставить нейросеть
# from mlp_model import MLP_model, inference
# from efficient_kan_model import inference_kan, KAN_model

def time_syncronization(tx_signal, rx_signal, time_delay=0):
    corr = np.correlate(np.abs(rx_signal), np.abs(tx_signal), mode='valid')
    delay = np.argmax(corr)
    return rx_signal[delay + time_delay : delay + len(tx_signal) + time_delay]

def generate_tx_base(bits_num, mod_order, sps, rolloff, filter_span, fs, ts, debug_check=0):
    bits = np.random.randint(0, 2, bits_num)
    if mod_order == 64:
        qam = mod.QAMModem(64)
        symbol_signal = qam.modulate(bits)
    else:
        qam = cf.Modulator32QAM()
        # Для 32-QAM нужно чтобы количество бит делилось на 5
        bits = bits[:len(bits) - (len(bits) % 5)] 
        symbol_signal = qam.modulate(bits)
        
    up_signal = cf.upsample(symbol_signal, sps=sps)
    shaped_signal = cf.pulse_shaping(up_signal, rolloff, filter_span, sps, ts, fs * sps, plt_en=debug_check)
    return bits, qam, symbol_signal, up_signal, shaped_signal

def simulate_channel_and_rx(bits, qam, shaped_signal_pure, up_signal, symbol_signal, 
                            snr_arr, inl_en, dac_gain, adc_gain, sps, sps_2, fs, rolloff,
                            filter_span, ts, mod_order, noise_en=1, debug_check=0, data_save=0):
    bers = np.zeros_like(snr_arr, dtype=np.float64)
    nmse_final_arr = np.zeros_like(bers)

    if dac_gain != 0:
        current_shaped = cf.quantizer(shaped_signal_pure, resolution=5, gain=dac_gain, inl_en=inl_en) 
    else:
        current_shaped = shaped_signal_pure

    shaped_upsampled = cf.upsample(current_shaped, sps=sps_2)
    shaped_upsampled_filtered = cf.apply_fixed_lpf(shaped_upsampled, fs / 2 * (1 + rolloff) * 1.5, fs * sps_2 * sps)
    passband_signal_clean = cf.upconversion(shaped_upsampled_filtered, Fc=fs * sps_2, Fs=fs * sps_2 * sps, plt_en=0)

    def rx_worker(i, snr):
        if noise_en: passband_signal = cf.fast_awgn(passband_signal_clean, snr)
        else: passband_signal = passband_signal_clean
            
        baseband_signal = cf.fast_downconversion(passband_signal, Fc=fs * sps_2, Fs=fs * sps_2 * sps)
        baseband_after_lpf = time_syncronization(shaped_upsampled, baseband_signal, time_delay=200)
        downsampled = cf.downsample(baseband_after_lpf, sps_2)

        if adc_gain != 0:
            current_rx_max = np.max([np.abs(downsampled.real), np.abs(downsampled.imag)])
            dynamic_adc_gain = 115.0 / current_rx_max
            downsampled = cf.fast_quantizer_core(downsampled, resolution=8, gain=dynamic_adc_gain, inl_en=0, inl_array=np.zeros(1))

        downsampled_shaped = cf.pulse_shaping(downsampled, rolloff, filter_span, sps, ts, fs * sps, plt_en=0)
        recovered = time_syncronization(up_signal, downsampled_shaped, time_delay=128)
        downsampled = cf.downsample(recovered, sps)

        final_symbols = cf.dd_lms_equalizer(downsampled, qam, num_taps=31, mu=0.05)
        final_nmse = cf.nmse_calc(symbol_signal, final_symbols)

        if mod_order == 32: demodulated_bits = qam.demodulate(final_symbols)
        else: demodulated_bits = qam.demodulate(final_symbols, "hard")
            
        # Отрезаем первые 20000 бит для сходимости фильтров
        ber = cf.ber_calc(bits[20000:-5000], demodulated_bits[20000:-5000])
        return i, ber, final_nmse

    num_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(rx_worker, i, snr_arr[i]): i for i in range(len(snr_arr))}
        for future in as_completed(futures): # Без tqdm внутри worker'а чтобы не засорять консоль
            i, ber, final_nmse = future.result()
            bers[i] = ber
            nmse_final_arr[i] = final_nmse
            
    return bers, nmse_final_arr, None, dac_gain, adc_gain

def sweep_clip_ratios(clip_ratios, bits, qam, shaped_signal, up_signal, symbol_signal, 
                      snr_arr, inl_val, adc_gain, sps, sps_2, fs, rolloff, 
                      filter_span, ts, mod_order, base_snr_ideal, fec_limit=3.84e-3):
    
    penalties, valid_crs = [], []
    rms_amp = cf.rms_calc(shaped_signal)
    
    def worker(cr):
        current_dac_gain = 15.0 / (rms_amp * cr)
        bers, _, _, _, _ = simulate_channel_and_rx(
            bits, qam, shaped_signal, up_signal, symbol_signal, 
            snr_arr, inl_en=inl_val, dac_gain=current_dac_gain, adc_gain=adc_gain, 
            sps=sps, sps_2=sps_2, fs=fs, rolloff=rolloff, filter_span=filter_span,
            ts=ts, mod_order=mod_order, debug_check=0, noise_en=1, data_save=0
        )
        snr_distorted = cf.get_snr_from_ber(fec_limit, snr_arr, bers)
        if not np.isnan(snr_distorted): return cr, (snr_distorted - base_snr_ideal)
        return cr, np.nan

    num_workers = os.cpu_count() or 4
    print(f"\n--- Перебор Clip Ratio: {mod_order}-QAM, INL={inl_val} LSB (Потоков: {num_workers}) ---")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, cr): cr for cr in clip_ratios}
        for future in tqdm(as_completed(futures), total=len(clip_ratios), desc="Sweeping Gain"):
            cr, penalty = future.result()
            valid_crs.append(cr)
            penalties.append(penalty)
            
    sorted_results = sorted(zip(valid_crs, penalties))
    valid_crs, penalties = zip(*sorted_results)
    return np.array(valid_crs), np.array(penalties)

def main():
    BITS_NUM = 6_00_000 # Оптимально для скорости и точности. Если нужен идеальный график, ставь 1_000_000
    F_SYM = 10e3 
    FS = 10e3 
    SPS = 2
    SPS_2 = 20 
    TS = 1 / F_SYM
    ROLLOFF = 0.2 
    FILTER_SPAN = 64 
    ADC_GAIN = 4.03 
    INL_VAL = 2 # Ставим 2 LSB для калибровки
    FEC_LIMIT = 3.84e-3
    snr_arr = np.arange(14, 33, 1)

    print("=== 1. РАСЧЕТ ИДЕАЛЬНЫХ БАЗОВЫХ ЛИНИЙ (Baseline) ===")
    
    bits_64, qam_64, sym_64, up_64, shaped_64 = generate_tx_base(
        BITS_NUM, 64, SPS, ROLLOFF, FILTER_SPAN, FS, TS)
    ideal_dac_gain_64 = 15.0 / (cf.rms_calc(shaped_64) * 2.6)
    bers_ideal_64, _, _, _, _ = simulate_channel_and_rx(
        bits_64, qam_64, shaped_64, up_64, sym_64, snr_arr, 0, ideal_dac_gain_64, ADC_GAIN, 
        SPS, SPS_2, FS, ROLLOFF, FILTER_SPAN, TS, 64)
    snr_base_64 = cf.get_snr_from_ber(FEC_LIMIT, snr_arr, bers_ideal_64)

    bits_32, qam_32, sym_32, up_32, shaped_32 = generate_tx_base(
        BITS_NUM, 32, SPS, ROLLOFF, FILTER_SPAN, FS, TS)
    ideal_dac_gain_32 = 15.0 / (cf.rms_calc(shaped_32) * 2.6) 
    bers_ideal_32, _, _, _, _ = simulate_channel_and_rx(
        bits_32, qam_32, shaped_32, up_32, sym_32, snr_arr, 0, ideal_dac_gain_32, ADC_GAIN, 
        SPS, SPS_2, FS, ROLLOFF, FILTER_SPAN, TS, 32)
    snr_base_32 = cf.get_snr_from_ber(FEC_LIMIT, snr_arr, bers_ideal_32)

    print(f"Базовый SNR (FEC={FEC_LIMIT}): 64-QAM = {snr_base_64:.2f} дБ | 32-QAM = {snr_base_32:.2f} дБ")

    print("\n=== 2. ПОИСК ОПТИМАЛЬНОЙ РАБОЧЕЙ ТОЧКИ (U-Curve) ===")
    clip_ratios_to_test = np.linspace(1.5, 3.5, 12) # Перебираем 12 значений

    cr_64, pen_64 = sweep_clip_ratios(
        clip_ratios_to_test, bits_64, qam_64, shaped_64, up_64, sym_64, 
        snr_arr, INL_VAL, ADC_GAIN, SPS, SPS_2, FS, ROLLOFF, FILTER_SPAN, TS, 64, snr_base_64, FEC_LIMIT)

    cr_32, pen_32 = sweep_clip_ratios(
        clip_ratios_to_test, bits_32, qam_32, shaped_32, up_32, sym_32, 
        snr_arr, INL_VAL, ADC_GAIN, SPS, SPS_2, FS, ROLLOFF, FILTER_SPAN, TS, 32, snr_base_32, FEC_LIMIT)

    print("\n=== 3. ПОСТРОЕНИЕ ГРАФИКА ===")
    plt.figure(figsize=(10, 6))
    plt.plot(cr_64, pen_64, marker='o', color='red', label=f'64-QAM Penalty (INL={INL_VAL})')
    plt.plot(cr_32, pen_32, marker='o', color='blue', label=f'32-QAM Penalty (INL={INL_VAL})')
    
    min_pen_64, best_cr_64 = np.nanmin(pen_64), cr_64[np.nanargmin(pen_64)]
    min_pen_32, best_cr_32 = np.nanmin(pen_32), cr_32[np.nanargmin(pen_32)]
    
    plt.axhline(min_pen_64, color='red', linestyle='--', alpha=0.5)
    plt.axhline(min_pen_32, color='blue', linestyle='--', alpha=0.5)
    plt.text(best_cr_64, min_pen_64 + 0.1, f"Min: {min_pen_64:.2f}dB\nCR: {best_cr_64:.2f}", color='red', ha='center')
    plt.text(best_cr_32, min_pen_32 + 0.1, f"Min: {min_pen_32:.2f}dB\nCR: {best_cr_32:.2f}", color='blue', ha='center')

    plt.title(f'Поиск оптимального Input Back-Off (INL = {INL_VAL} LSB)')
    plt.xlabel('Clip Ratio (Меньше = Громче, Больше = Тише)')
    plt.ylabel(f'OSNR Penalty @ BER={FEC_LIMIT} (dB)')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'U_Curve_Optimization_INL_{INL_VAL}.png')
    plt.show()

if __name__ == "__main__":
    main()