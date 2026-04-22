import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn
import channel_funcs as cf
from tqdm import tqdm
from model import MLP_model


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

    nmse = nmse_calc(signal_1, signal_2)
    print(f'NMSE = {nmse} dB')
    
    mid = len(signal_1) // 2
    plt.figure(100)
    plt.stem(signal_1.real[mid : mid + 100], linefmt='r-', label = f'Initial, real part, nmse = {nmse}')
    plt.stem(signal_2.real[mid : mid + 100], linefmt='g-', label = f'Processed, real part, nmse = {nmse}')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

    plt.figure(101)
    plt.stem(signal_1.imag[mid : mid + 100], linefmt='r-', label = f'Initial, imaginary part, nmse = {nmse}')
    plt.stem(signal_2.imag[mid : mid + 100], linefmt='g-', label = f'Processed, imaginary part, nmse = {nmse}')
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
    title = 'Signal 4 SPS before the pulse shaping; Signal 4 SPS after the matched filtering'
    compare_2_signals(up_signal, recovered, title)

def generate_tx_base(bits_num, mod_order, sps, rolloff, filter_span, fs, ts, debug_check = 1, data_save = 0):
    np.random.seed(100)
    bits = np.random.randint(0, 2, bits_num)
    qam = mod.QAMModem(mod_order)
    
    symbol_signal = qam.modulate(bits)
    up_signal = cf.upsample(symbol_signal, sps)
    if debug_check:
        cf.spectrum_plot(symbol_signal, Fs = fs, title = 'Initial symbol signal spectrum', plt_en = 1)
        cf.spectrum_plot(up_signal, Fs = sps * fs, title = 'Initial upsampled to 4 SPS symbol signal spectrum', plt_en = 1)
    
    shaped_signal = cf.pulse_shaping(
        up_signal,
        rolloff=rolloff,
        filter_span=filter_span,
        sps=sps,
        Fs=fs * sps,
        Ts=ts,
        normaliztion="L2",
        plt_en=0
    )

    if data_save:
        normalized_shaped = normalize_energy(shaped_signal)
        np.save('model_targets_64_qam.npy', normalized_shaped)
    ### Shaped signal spectrum check
    cf.spectrum_plot(shaped_signal, Fs = sps * fs, title = 'Spectrum after the pulse shaping', plt_en = debug_check)
    ###
    if debug_check:
        ### TX Pulse shaping block check, NMSE = -67 dB
        pulse_shaping_check(shaped_signal, up_signal, rolloff, filter_span, sps, fs, ts)
        ###
    return bits, qam, symbol_signal, up_signal, shaped_signal

def simulate_channel_and_rx(bits, qam, shaped_signal_pure, up_signal, symbol_signal, 
                            snr_arr, inl_en, dac_gain, adc_gain, sps, sps_2, fs, rolloff,
                            filter_span, ts, mod_order, noise_en = 1, debug_check = 1, data_save = 0):
    
    bers = np.zeros_like(snr_arr, dtype=np.float64)
    nmse_final_arr = np.zeros_like(bers)
    
    for i in tqdm(range(len(snr_arr)), desc = f"Simulating INL_EN = {inl_en}"):
        ######### DAC with the distortions
        current_shaped = cf.quantizer(shaped_signal_pure, resolution = 5, gain=dac_gain, inl_en = inl_en) 
        
        '''if np.sum(np.abs(current_shaped)) == 0:
            bers[i] = 0.5
            continue'''
        # TODO add a normalization when saving the data
        if data_save:
            print('Data saving mode')
            ### ADC quantizer
            model_objects = cf.quantizer(current_shaped, resolution = 8, gain = adc_gain)
            model_objects = normalize_energy(model_objects)
            np.save('model_objects_64_qam.npy', model_objects)

        ### Upsampling to 40 SPS
        shaped_upsampled = cf.upsample(current_shaped, sps=sps_2)
        ### Shaped signal spectrum check
        if debug_check:
            cf.spectrum_plot(shaped_upsampled, Fs = sps_2 * sps * fs, title = f'Spectrum after the pulse shaping and upsampling, {sps * sps_2} SPS', plt_en = 1)
        ###
        
        ### Add a LPF to shaped_upsampled
        shaped_upsampled_filtered = cf.apply_fixed_lpf(shaped_upsampled, fs / 2 * (1 + rolloff) * 1.5, fs * sps_2 * sps)
        ### Shaped upsampled and filtered through LPF signal spectrum check
        if debug_check:
            cf.spectrum_plot(shaped_upsampled_filtered, Fs = sps_2 * sps * fs, title = 'Spectrum after the pulse shaping and upsampling to 40 SPS, then adding a LPF', plt_en = 1)
        ###

        ######## Upconversion
        passband_signal = cf.upconversion(shaped_upsampled_filtered, Fc = fs * sps_2, Fs=fs * sps_2 * sps, plt_en = debug_check)
        if noise_en:
            passband_signal = awgn(passband_signal, snr_dB=snr_arr[i])
        
        ######## Downconversion
        baseband_signal = cf.downconversion(passband_signal, Fc = fs * sps_2, Fs = fs * sps_2 * sps, plt_en = debug_check)

        ######### ADC with distortions
        baseband_after_lpf = time_syncronization(shaped_upsampled, baseband_signal)
        downsampled = cf.downsample(baseband_after_lpf, sps_2)
        recovered = cf.upsample(downsampled, sps_2)

        if debug_check:
            ### Shaped upsampled to 40 SPS signal and signal on 40 SPS after the LPF nmse check 
            print('Shaped upsampled to 40 SPS signal and  40 SPS signal after the LPF nmse calculation')
            shaped_up_energy, recovered_energy = np.sum(np.abs(shaped_upsampled)** 2), np.sum(np.abs(recovered) ** 2)
            print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')
            shaped_upsampled, recovered = normalize_energy(shaped_upsampled), normalize_energy(recovered)
            shaped_up_energy, recovered_energy = np.sum(np.abs(shaped_upsampled)** 2), np.sum(np.abs(recovered) ** 2)
            print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')

            title = f'Shaped, upsampled to {sps * sps_2} SPS, before LPF signal; Recovered signal on {sps * sps_2} SPS after LPF with time syncronization'
            compare_2_signals(shaped_upsampled, recovered, title)

        ### ADC quantizer
        downsampled = cf.quantizer(downsampled, resolution = 8, gain = adc_gain)

        ######## Matched filter
        downsampled_shaped = cf.pulse_shaping(downsampled, rolloff, filter_span, sps, ts, fs * sps, plt_en=0)
        
        ### Correlation and downsampling
        recovered = time_syncronization(up_signal, downsampled_shaped)
        downsampled = cf.downsample(recovered, sps)
        if debug_check:
            recovered = cf.upsample(downsampled, sps)
            print(f'Initial upsampled on {sps} SPS signal and Recovered signal after the matched filtering on {sps} SPS nmse calculation')
            up_energy, recovered_energy = np.sum(np.abs(up_signal)** 2), np.sum(np.abs(recovered) ** 2)
            print(f'Signals energies: {up_energy}; {recovered_energy}')
            up_signal, recovered = normalize_energy(up_signal), normalize_energy(recovered)
            up_energy, recovered_energy = np.sum(np.abs(up_signal)** 2), np.sum(np.abs(recovered) ** 2)
            print(f'Signals energies: {up_energy}; {recovered_energy}')
            title = f'Initial upsampled on {sps} SPS signal; Recovered signal after the matched filtering on {sps} SPS with time syncronization'
            compare_2_signals(up_signal, recovered, title)

        ######## Getting symbols back on SPS = 1
        final_symbols = constellation_normalization(downsampled, mod_order)
        
        final_nmse = nmse_calc(symbol_signal, final_symbols)
        nmse_final_arr[i] = final_nmse

        ####### Demapping
        demodulated_bits = qam.demodulate(final_symbols, "hard")
        ber = cf.ber_calc(bits, demodulated_bits)
        bers[i] = ber
        
    return bers, nmse_final_arr

def main():
    # Parameters
    BITS_NUM = 360_000 #6_000_000 #1_000_002
    MOD_ORDER = 64
    F_SYM = 10e3 #FS / SPS
    FS = 10e3 # for SPS = 1 !!!
    SPS = 4
    SPS_2 = 10
    TS = 1 / F_SYM
    ROLLOFF = 0.125
    FILTER_SPAN = 64
    DAC_GAIN = 2.928
    ADC_GAIN = 14
    DATA_SAVE = 1


    bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER, sps = SPS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = 0, data_save = 1)

    snr_arr = np.arange(10, 11, 1)
    shaped_signal_pure = shaped_signal.copy()
    ### Simulation with INL
    bers_with_inl, nmse_final_arr_with_inl = simulate_channel_and_rx(bits, qam, shaped_signal_pure, up_signal, symbol_signal, 
        snr_arr, inl_en = 1, dac_gain = DAC_GAIN, adc_gain = ADC_GAIN, sps = SPS, sps_2 = SPS_2, fs = FS,
        rolloff = ROLLOFF, filter_span = FILTER_SPAN,
        ts = TS, mod_order = MOD_ORDER, debug_check = 0, noise_en = 1, data_save = DATA_SAVE)
    
    ### Simulation without INL
    bers_no_inl, nmse_final_arr_no_inl = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
        snr_arr, inl_en = 0, dac_gain = DAC_GAIN, adc_gain = ADC_GAIN, sps = SPS, sps_2 = SPS_2, fs = FS,
        rolloff = ROLLOFF, filter_span = FILTER_SPAN,
        ts = TS, mod_order = MOD_ORDER, debug_check = 1, noise_en = 1)
    
    
    plt.figure(10)
    plt.plot(snr_arr, bers_no_inl, marker = 'o', color = 'red', label = f'{MOD_ORDER} QAM, without INL')
    plt.plot(snr_arr, bers_with_inl, marker = 'o', color = 'blue', label = f'{MOD_ORDER} QAM, with INL')
    plt.legend()
    plt.ylabel('BER')
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.title(f'BER(SNR), {MOD_ORDER} QAM')
    plt.grid()
    plt.savefig('BER(SNR).png')
    plt.show()

    plt.figure(11)
    plt.plot(snr_arr, nmse_final_arr_no_inl, marker = 'o', color = 'purple', label = f'{MOD_ORDER} QAM, without INL')
    plt.plot(snr_arr, nmse_final_arr_with_inl, marker = 'o', color = 'orange', label = f'{MOD_ORDER} QAM, with INL')
    plt.legend()
    plt.ylabel('NSME')
    plt.xlabel('SNR')
    plt.title(f'NMSE(SNR), {MOD_ORDER} QAM')
    plt.grid()
    plt.savefig('NMSE(SNR).png')
    plt.show()




if __name__ == "__main__":
    main()
