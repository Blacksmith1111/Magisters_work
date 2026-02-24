import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn

import channel_funcs as cf

def main():
    # --- 1. Parameters ---
    BITS_NUM = 60_000
    MOD_ORDER = 64
    FS = 10e3
    FC = FS / 4
    SPS = 10
    F_SYM = FS / SPS
    TS = 1 / F_SYM
    ROLLOFF = 0.25
    FILTER_SPAN = 10
    DAC_BITS = 6
    ADC_BITS = 8
    SNR_DB = 15
    
    # Flags
    EN_ADC = 1
    EN_QUANTIZER = 1
    EN_UP_DOWN_CONV = 1

    # --- 2. Transmitter (TX) ---
    bits = np.random.randint(0, 2, BITS_NUM)
    qam = mod.QAMModem(MOD_ORDER)
    symbol_signal = qam.modulate(bits)
    
    up_signal = cf.upsample(symbol_signal, SPS)
    
    shaped_signal = cf.pulse_shaping(
        up_signal, rolloff=ROLLOFF, filter_span=FILTER_SPAN,
        sps=SPS, Fs=FS, Ts=TS, normaliztion='L2')
    
    cf.spectrum_plot(shaped_signal, FS, title='Original baseband signal spectrum')

    # DAC / Quantizer
    if EN_QUANTIZER:
        quantized_signal, scale_dac = cf.quantizer(shaped_signal, resolution=DAC_BITS)
    else:
        quantized_signal = shaped_signal
        scale_dac = 1.0

    # --- 3. Channel & RF ---
    if EN_UP_DOWN_CONV:
        # Upconversion
        passband_signal = cf.upconversion(quantized_signal, Fs=FS, Fc=FC)
        cf.spectrum_plot(passband_signal, FS, title='Passband signal spectrum')
        
        # AWGN
        noisy_signal = awgn(passband_signal, snr_dB=SNR_DB)
        
        # Downconversion
        baseband_signal = cf.downconversion(noisy_signal, Fs=FS, Fc=FC, B=1200)
        cf.spectrum_plot(baseband_signal, FS, title='Baseband (shifted back) signal spectrum')
    else:
        baseband_signal = shaped_signal

    # --- 4. Receiver (RX) ---
    # ADC
    if EN_ADC:
        scaled_signal, scale_adc = cf.ADC(baseband_signal, adc_bits=ADC_BITS, dac_bits=DAC_BITS)
    else:
        scaled_signal = baseband_signal
        scale_adc = 1.0

    # Matched Filter
    recovered_signal = cf.pulse_shaping(
        scaled_signal, rolloff=ROLLOFF, filter_span=FILTER_SPAN, 
        sps=SPS, Fs=FS, Ts=TS, normaliztion='L2'
    )

    # Time Synchronization
    correlation = sig.correlate(up_signal, recovered_signal, mode='full')
    lags = sig.correlation_lags(len(up_signal), len(recovered_signal), mode='full')
    actual_delay = lags[np.argmax(correlation)]
    print(f"Calculated delay (in samples): {actual_delay}")

    rec_sync = recovered_signal[-actual_delay : -actual_delay + len(up_signal)]
    
    # Downsampling & Rescaling
    downsampled = cf.downsample(rec_sync, SPS)
    downsampled /= (scale_dac * scale_adc)
    
    # --- 5. Metrics & Results ---
    # NMSE
    nmse = 10 * np.log10(np.sum(np.abs(downsampled - symbol_signal)**2) / np.sum(np.abs(symbol_signal)**2))
    
    # Demapping
    demod_bits = qam.demodulate(downsampled, 'hard') 
    ber = cf.ber_calc(bits, demod_bits)
    
    print("-" * 30)
    print(f'NMSE: {nmse:.2f} dB')
    print(f'BER:  {ber * 100:.2f} %')
    print("-" * 30)

    # Plotting final results
    plt.figure(figsize=(10, 4))
    plt.stem(rec_sync.real[:100])
    plt.title("Shifted recovered signal (fragment)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()