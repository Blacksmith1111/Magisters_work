import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn
import channel_funcs as cf
from tqdm import tqdm


def time_syncronization(base_signal, delayed_signal):
    correlation = sig.correlate(base_signal, delayed_signal, mode = "full")
    lags = sig.correlation_lags(len(base_signal), len(delayed_signal), mode = "full")
    actual_delay = lags[np.argmax(correlation)]
    print(f"Calculated delay (in samples): {actual_delay}")
    recovered_signal = delayed_signal[-actual_delay: -actual_delay + len(base_signal)]
    return recovered_signal

def nmse_calc(signal_1, signal_2):
    nmse = 10 * np.log10(np.sum(np.abs(signal_1 - signal_2) ** 2) / np.sum(np.abs(signal_1) ** 2))
    return nmse

def nmse_calc_absolute(signal_1, signal_2):
    nmse = np.sum(np.abs(signal_1 - signal_2) ** 2) / np.sum(np.abs(signal_1) ** 2)
    return nmse

def rms_calc(signal):
    return np.sqrt(np.mean(np.abs(signal)**2))

def constellation_normalization(signal, mod_order):
    '''
    Normalizes the signal to rms = constellation rms
    '''
    constellation_rms = cf.qam_constellation_rms_calc(mod_order)
    signal = signal / rms_calc(signal) *  constellation_rms
    return signal

def normalize_energy(s):
    return s / np.sqrt(np.sum(np.abs(s)**2))

def compare_2_signals(signal_1, signal_2, title):
    mid = len(signal_1) // 2
    plt.figure(100)
    plt.stem(signal_1.real[mid : mid + 100], linefmt='r-', label = 'Initial, real part')
    plt.stem(signal_2.real[mid : mid + 100], linefmt='g-', label = 'Processed, real part')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

    plt.figure(101)
    plt.stem(signal_1.imag[mid : mid + 100], linefmt='r-', label = 'Initial, imaginary part')
    plt.stem(signal_2.imag[mid : mid + 100], linefmt='g-', label = 'Processed, imaginary part')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

def pulse_shaping_check(shaped_signal, up_signal, ROLLOFF, FILTER_SPAN, SPS, FS, TS):
    shaped_back_signal = cf.pulse_shaping(
        shaped_signal,
        rolloff=ROLLOFF,
        filter_span=FILTER_SPAN,
        sps=SPS,
        Fs=FS * SPS,
        Ts=TS,
        normaliztion="L2",
        plt_en = 1
    )
    recovered = time_syncronization(up_signal, shaped_back_signal) 
    downsampled = cf.downsample(recovered, SPS)
    recovered = cf.upsample(downsampled, SPS)
    shaped_energy, recovered_energy = np.sum(np.abs(shaped_signal)** 2), np.sum(np.abs(recovered) ** 2)
    print(f'Signals energies: {shaped_energy}; {recovered_energy}')
    nmse = nmse_calc(up_signal, recovered) 
    
    print(f'NMSE = {nmse} dB')
    plt.figure(100)
    plt.plot(up_signal.real[:100], color  = 'red', label = 'Up signal real part')
    plt.plot(recovered.real[:100], color = 'green', label = 'Shaped back signal real part')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(101)
    plt.plot(up_signal.imag[:100], color  = 'red', label = 'Up signal imag part')
    plt.plot(recovered.imag[:100], color = 'green', label = 'Shaped back signal imag part')
    plt.legend()
    plt.grid()
    plt.show()

def recover_signal(initial_signal, processed_signal, SPS):
    recovered = time_syncronization(initial_signal, processed_signal)
    downsampled = cf.downsample(recovered, SPS)
    recovered = cf.upsample(downsampled, SPS)
    return recovered, downsampled


def main():
    # Parameters
    BITS_NUM = 1_000_002
    MOD_ORDER = 64
    F_SYM = 10e3 #FS / SPS
    FS = 10e3 # for SPS = 1 !!!
    FC = FS / 4
    SPS = 4
    #F_SYM = FS / SPS
    TS = 1 / F_SYM
    ROLLOFF = 0.125
    FILTER_SPAN = 64
    DAC_BITS = 6
    ADC_BITS = 8
    SNR_DB = 15

    # Flags
    EN_ADC = 1
    EN_QUANTIZER = 1
    EN_UP_DOWN_CONV = 1
    EN_NOISE = 0

    # Transmitter (TX)
    np.random.seed(100)
    bits = np.random.randint(0, 2, BITS_NUM)
    qam = mod.QAMModem(MOD_ORDER)
    
    
    ######### 1 Symbols creating
    test_freq = 200 # Гц
    n_test = np.arange(BITS_NUM // 6) # количество символов
    # Создаем "медленную" синусоиду вместо случайных символов
    #symbol_signal = np.exp(1j * 2 * np.pi * test_freq / F_SYM * n_test)
    symbol_signal = qam.modulate(bits)
    ### Symbols spectrum check
    cf.spectrum_plot(symbol_signal, Fs = FS, title = 'Signal spectrum, 1 SPS', plt_en = 1)
    ###



    ######### 2 Upsampling to 4 SPS
    up_signal = cf.upsample(symbol_signal, SPS)
    ### Upsampled signal spectrum check
    cf.spectrum_plot(up_signal, Fs = SPS * FS, title = 'Signal spectrum, 4 SPS', plt_en = 1)
    ###
    
    

    ######### 3 TX Pulse shaping
    shaped_signal = cf.pulse_shaping(
        up_signal,
        rolloff=ROLLOFF,
        filter_span=FILTER_SPAN,
        sps=SPS,
        Fs=FS * SPS,
        Ts=TS,
        normaliztion="L2",
        plt_en = 1
    )

    ### Shaped signal spectrum check
    cf.spectrum_plot(shaped_signal, Fs = SPS * FS, title = 'Spectrum after the pulse shaping', plt_en = 1)
    ###

    ### TX Pulse shaping block check, NMSE = -67 dB
    pulse_shaping_check(shaped_signal, up_signal, ROLLOFF, FILTER_SPAN, SPS, FS, TS)
    ###



    ######### DAC with the distortions
    #gains_dac = np.linspace(0.1, 10, 50)
    gains_dac = np.arange(2.928, 2.93, 0.01)
    gains_adc = np.arange(14,15,1)
    bers = np.zeros_like(gains_adc)
    nmse_final_arr = np.zeros_like(gains_adc)
    shaped_signal_pure = shaped_signal.copy()
    
    for i in tqdm(range((len(gains_adc)))):
        #shaped_signal = cf.quantizer(shaped_signal, resolution = 5, gain = gains[i]) ### Add the quantizer
        #shaped_upsampled = cf.upsample(shaped_signal, sps = 10)
        current_shaped = cf.quantizer(shaped_signal_pure, resolution = 5, gain = 2.928) 
        #current_shaped = shaped_signal_pure
        if np.sum(np.abs(current_shaped)) == 0:
            bers[i] = 0.5
            continue
            
        SPS_2 = 10
        ### Upsampling to 40 SPS
        shaped_upsampled = cf.upsample(current_shaped, sps = SPS_2)
        ###
        ### Shaped signal spectrum check
        cf.spectrum_plot(shaped_upsampled, Fs = SPS_2* SPS * FS, title = 'Spectrum after the pulse shaping and upsampling, 40 SPS', plt_en = 0)
        ###
        
        ### Add a LPF to shaped_upsampled
        shaped_upsampled_filtered = cf.apply_fixed_lpf(shaped_upsampled, FS / 2 * (1 + ROLLOFF) * 1.5, FS * SPS_2 * SPS)
        #shaped_upsampled_filtered = shaped_upsampled
        ### Shaped upsampled and filtered through LPF signal spectrum check
        cf.spectrum_plot(shaped_upsampled_filtered, Fs = SPS_2 * SPS * FS, title = 'Spectrum after the pulse shaping and upsampling to 40 SPS, then adding a LPF', plt_en = 0)
        ###




        ######## Upconversion
        passband_signal = cf.upconversion(shaped_upsampled_filtered, Fc = FS * SPS_2, Fs = FS * SPS_2 * SPS, plt_en = 0)
        ######## Downconversion
        baseband_signal = cf.downconversion(passband_signal, Fc = FS * SPS_2, Fs = FS * SPS_2 * SPS, plt_en = 0)




        ######### ADC with distortions
        ### Correlation between the shaped_upsampled and shaped_upsampled_filtered
        after_lpf = time_syncronization(shaped_upsampled, baseband_signal)
        downsampled = cf.downsample(after_lpf, SPS_2)
        recovered = cf.upsample(downsampled, SPS_2)

        ### ADC quantizer
        adc_quantized = cf.quantizer(after_lpf, resolution = 8, gain = gains_adc[i])
        if np.sum(np.abs(adc_quantized)) == 0:
            bers[i] = 0.5
            continue
        downsampled = cf.downsample(adc_quantized, SPS_2)

        ### Shaped upsampled to 40 SPS signal and signal after the LPF normalization
        shaped_up_energy, recovered_energy = np.sum(np.abs(shaped_upsampled)** 2), np.sum(np.abs(recovered) ** 2)
        print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')
        shaped_upsampled, recovered = normalize_energy(shaped_upsampled), normalize_energy(recovered)
        shaped_up_energy, recovered_energy = np.sum(np.abs(shaped_upsampled)** 2), np.sum(np.abs(recovered) ** 2)
        print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')
        
        nmse = nmse_calc(shaped_upsampled, recovered)
        nmse_abs = nmse_calc_absolute(shaped_upsampled, recovered)
        print(f'NMSE = {nmse} dB')
        print(f'NMSE absolute = {nmse_abs}')

        ### 
        title = 'Shaped, upsampled to 40 SPS, before LPF signal; Recovered signal on 40 SPS after LPF with time syncro'
        compare_2_signals(shaped_upsampled, recovered, title)
        ###
        #########



        ######## Matched filter to the downsampled to 4 SPS signal
        downsampled_shaped = cf.pulse_shaping(downsampled, ROLLOFF, FILTER_SPAN, SPS, TS, FS * SPS, plt_en = 0)
        
        ### Correlation between the up_signal and downsampled matched filtered
        recovered = time_syncronization(up_signal, downsampled_shaped)
        downsampled = cf.downsample(recovered, SPS)
        recovered = cf.upsample(downsampled, SPS)
        recovered, up_signal = normalize_energy(recovered), normalize_energy(up_signal)
        recovered_energy, up_signal_energy = np.sum(np.abs(recovered)** 2), np.sum(np.abs(up_signal)** 2)
        print(f'Signals energies: {recovered_energy}; {up_signal_energy}')

        title = 'Initial upsampled on 4 SPS signal; Recovered signal after the matched filtering on 4 SPS with time syncro'
        compare_2_signals(up_signal, recovered, title)

        nmse = nmse_calc(up_signal, recovered)
        print(f'NMSE = {nmse} dB')
        


        ######## Downsampling to SPS = 1
        final_symbols = cf.downsample(recovered, SPS)
        final_symbols = constellation_normalization(final_symbols, MOD_ORDER)
        final_symbols_rms, symbol_signal_rms =  rms_calc(final_symbols), rms_calc(symbol_signal) 
        nmse = nmse_calc(symbol_signal, final_symbols)
        nmse_final_arr[i] = nmse
        print(f'NMSE = {nmse}')
        



        ####### Demapping
        demodulated_bits = qam.demodulate(final_symbols, "hard")
        ber = cf.ber_calc(bits, demodulated_bits)
        print(f'BER = {ber}')
        bers[i] = ber
    
    ### Ber (gain)
    plt.figure(1)
    plt.plot(gains_adc, bers)
    plt.ylabel('BER')
    plt.xlabel('Gain')
    plt.title(f'BER(Gain) in ADC, {MOD_ORDER} QAM')
    plt.grid()
    #plt.savefig('adc_ber_gain.png')
    plt.show()

    plt.figure(2)
    plt.plot(gains_adc, nmse_final_arr)
    plt.ylabel('Final NMSE between the symbols')
    plt.xlabel('Gain')
    plt.title(f'NMSE(Gain) in ADC, {MOD_ORDER} QAM')
    plt.grid()
    #plt.savefig('adc_nmse_gain.png')
    plt.show()
    ###
    cf.spectrum_plot(shaped_signal, FS, title="Original baseband signal spectrum")
    # DAC / Quantizer
    if EN_QUANTIZER:
        quantized_signal, scale_dac = cf.quantizer(shaped_signal, resolution=DAC_BITS)
    else:
        quantized_signal = shaped_signal
        scale_dac = 1.0

    # Channel and RF
    if EN_UP_DOWN_CONV:
        # Upconversion
        passband_signal = cf.upconversion(quantized_signal, Fs=FS, Fc=FC)
        cf.spectrum_plot(passband_signal, FS, title="Passband signal spectrum")

        # AWGN
        if EN_NOISE:
            noisy_signal = awgn(passband_signal, snr_dB=SNR_DB)
        else:
            noisy_signal = passband_signal

        # Downconversion
        baseband_signal = cf.downconversion(noisy_signal, Fs=FS, Fc=FC, B=1200)
        cf.spectrum_plot(
            baseband_signal, FS, title="Baseband (shifted back) signal spectrum"
        )
    else:
        baseband_signal = shaped_signal

    # Receiver (RX) ---
    # ADC
    if EN_ADC:
        scaled_signal, scale_adc = cf.ADC(
            baseband_signal, adc_bits=ADC_BITS, dac_bits=DAC_BITS
        )
    else:
        scaled_signal = baseband_signal
        scale_adc = 1.0

    # Matched Filter
    recovered_signal = cf.pulse_shaping(
        scaled_signal,
        rolloff=ROLLOFF,
        filter_span=FILTER_SPAN,
        sps=SPS,
        Fs=FS,
        Ts=TS,
        normaliztion="L2",
    )

    # Time Synchronization
    correlation = sig.correlate(up_signal, recovered_signal, mode="full")
    lags = sig.correlation_lags(len(up_signal), len(recovered_signal), mode="full")
    actual_delay = lags[np.argmax(correlation)]
    print(f"Calculated delay (in samples): {actual_delay}")

    rec_sync = recovered_signal[-actual_delay : -actual_delay + len(up_signal)]

    # Downsampling & Rescaling
    downsampled = cf.downsample(rec_sync, SPS)
    downsampled /= scale_dac * scale_adc

    # Metrics and results
    # NMSE
    nmse = 10 * np.log10(
        np.sum(np.abs(downsampled - symbol_signal) ** 2)
        / np.sum(np.abs(symbol_signal) ** 2)
    )

    # Demapping
    demod_bits = qam.demodulate(downsampled, "hard")
    ber = cf.ber_calc(bits, demod_bits)

    print("-" * 30)
    print(f"NMSE: {nmse:.2f} dB")
    print(f"BER:  {ber * 100:.2f} %")
    print("-" * 30)

    # Final results plot
    plt.figure(figsize=(10, 4))
    plt.stem(rec_sync.real[:100])
    plt.title("Shifted recovered signal (fragment)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
