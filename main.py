import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn
import channel_funcs as cf
from tqdm import tqdm
from mlp_model import MLP_model, inference
from efficient_kan_model import inference_kan, KAN_model
from concurrent.futures import ProcessPoolExecutor, as_completed


DEVICE = 'cuda:0'
model_mlp = MLP_model.to(DEVICE)
model_kan = KAN_model.to(DEVICE)
### 64 QAM weights
WEIGHTS_FILE_64_QAM_MLP_2_LSB = 'qam_64_mlp_2_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_KAN_2_LSB = 'qam_64_kan_2_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_MLP_4_LSB = 'qam_64_mlp_4_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_KAN_4_LSB = 'qam_64_kan_4_lsb_weights.pt'
### 32 QAM weights
WEIGHTS_FILE_32_QAM_MLP_2_LSB = 'qam_32_mlp_2_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_MLP_4_LSB = 'qam_32_mlp_4_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_KAN_2_LSB = 'qam_32_kan_2_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_KAN_4_LSB = 'qam_32_kan_4_LSB_weights.pt'

SIMULATION_WEIGHTS = {64:{'MLP':{2 : WEIGHTS_FILE_64_QAM_MLP_2_LSB, 4 : WEIGHTS_FILE_64_QAM_MLP_4_LSB}, 
                          'KAN':{2 : WEIGHTS_FILE_64_QAM_KAN_2_LSB, 4 : WEIGHTS_FILE_64_QAM_KAN_4_LSB}},
                        32:{'MLP':{2 : WEIGHTS_FILE_32_QAM_MLP_2_LSB, 4 : WEIGHTS_FILE_32_QAM_MLP_4_LSB}, 
                          'KAN':{2 : WEIGHTS_FILE_32_QAM_KAN_2_LSB, 4 : WEIGHTS_FILE_32_QAM_KAN_4_LSB}}}


def time_syncronization(base_signal, delayed_signal, time_delay = None):
    if time_delay is None:
        correlation = sig.correlate(base_signal, delayed_signal, mode = "full")
        lags = sig.correlation_lags(len(base_signal), len(delayed_signal), mode = "full")
        actual_delay = lags[np.argmax(correlation)]
        print(f"Calculated delay (in samples): {actual_delay}")
    else:
        actual_delay = -time_delay
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
    #print(constellation_rms / rms_calc(signal))
    signal = signal / rms_calc(signal) *  constellation_rms
    return signal

def normalize_energy(s):
    return s / np.sqrt(np.sum(np.abs(s)**2))

def energy_calc(s):
    return np.sum(np.abs(s)**2)

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
    recovered = time_syncronization(up_signal, shaped_back_signal, time_delay = None) 
    downsampled = cf.downsample(recovered, SPS)
    recovered = cf.upsample(downsampled, SPS)
    shaped_energy, recovered_energy = np.sum(np.abs(shaped_signal)** 2), np.sum(np.abs(recovered) ** 2)
    print(f'Signals energies: {shaped_energy}; {recovered_energy}')
    nmse = nmse_calc(up_signal, recovered) 
    print(f'NMSE = {nmse} dB')
    title = 'Signal 4 SPS before the pulse shaping; Signal 4 SPS after the matched filtering'
    compare_2_signals(up_signal, recovered, title)

def generate_tx_base(bits_num, mod_order, sps, rolloff, filter_span, fs, ts, debug_check = 1,
                      data_save = 0, model_apply = 0, inl_val = 2, seed = 100):
    np.random.seed(seed)
    bits = np.random.randint(0, 2, bits_num)
    qam = mod.QAMModem(mod_order) if mod_order == 64 else cf.Modulator32QAM()
    
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
        plt_en = 0
    )

    global_rms = rms_calc(shaped_signal)
    if model_apply:
        print('Using the model')
        ### Model applying
        mean_val = np.mean(shaped_signal)
        shaped_signal -= mean_val
        #shaped_rms = rms_calc(shaped_signal)
        shaped_normalized = shaped_signal / global_rms
        if model_apply == 1:
            prediction = inference(shaped_normalized,
                batch_size = len(shaped_normalized),
                model = model_mlp,
                device = DEVICE,
                weights_file = SIMULATION_WEIGHTS[mod_order]['MLP'][inl_val]) 
            print(SIMULATION_WEIGHTS[mod_order]['MLP'][inl_val])
        else:
            current_max = np.max(np.abs(shaped_normalized))
            prediction = inference_kan(
                shaped_normalized, 
                batch_size=8192, 
                model = model_kan, 
                device = DEVICE, 
                weights_file = SIMULATION_WEIGHTS[mod_order]['KAN'][inl_val],
                max_val = current_max
            )
            print(SIMULATION_WEIGHTS[mod_order]['KAN'][inl_val])
        prediction *= global_rms#shaped_rms # Switched back to the initial rms
        de_centered_signal = prediction + mean_val
        shaped_signal = de_centered_signal[:, 0] + 1j * de_centered_signal[:, 1]
    else:
        print('Not using the model')

    if data_save:
        print(f'global_rms for targets is {global_rms}')
        centered_shaped = shaped_signal - np.mean(shaped_signal)
        centered_shaped /= global_rms #rms_calc(centered_shaped)
        np.save(f'model_targets_{mod_order}_qam.npy', centered_shaped)
    
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
                            filter_span, ts, mod_order, noise_en = 1, debug_check = 1, data_save = 0, model_apply = 0):
    
    bers = np.zeros_like(snr_arr, dtype = np.float64)
    nmse_final_arr = np.zeros_like(bers)

    global_rms = rms_calc(shaped_signal_pure)
    ######### DAC with the distortions
    if dac_gain != 0:
            factor = 0
            if inl_en == 0:
                #clip_ratio = 2.6 # ideal
                clip_ratio = 1.8 if mod_order == 64 else 1.6
                #dac_gain = 2.928 if mod_order == 64 else 4.179
                #dac_gain = 2 if mod_order == 64 else 2.5
                pass
            
            elif inl_en == 2:
                clip_ratio = 1.17 if mod_order == 64 else 1.4
                #dac_gain = 2 if mod_order == 64 else 2.5
                factor = 6/8 
            else:
                clip_ratio = 3.51 if mod_order == 64 else 3.58
                #dac_gain = 1.41 if mod_order == 64 else 1.7
                factor = 5/8 #3.8 / 8
            #dac_gain = 15.0 / (rms_amp * clip_ratio)
            print(np.max(np.abs(shaped_signal_pure.real)), np.max(np.abs(shaped_signal_pure.imag))) 
            current_shaped = cf.quantizer(shaped_signal_pure, resolution = 5, gain = dac_gain, inl_en = inl_en * factor) 
            print(np.max(np.abs(current_shaped.real)), np.max(np.abs(current_shaped.imag))) 
    else:
            current_shaped = shaped_signal_pure


    if data_save:
        print('Data saving mode')
        print(np.max(np.abs(current_shaped.real)), np.max(np.abs(current_shaped.imag)))    
        model_objects = cf.quantizer(current_shaped, resolution = 8, gain = adc_gain)
        print(np.max(np.abs(model_objects.real)), np.max(np.abs(model_objects.imag)))
        model_objects /= (adc_gain * dac_gain)
        print(f'{np.max(np.abs(model_objects.real))}, {np.max(np.abs(model_objects.imag))}: After the de-embedding')

        model_objects -= np.mean(model_objects)
        model_objects /= global_rms #rms_calc(model_objects)
        print(f'global_rms for objects is {global_rms}')
        np.save(f'model_objects_{mod_order}_qam_INL_{inl_en}_LSB.npy', model_objects)
        return 1, 1, 1, 1, 1


    ### Upsampling to 40 SPS
    shaped_upsampled = cf.upsample(current_shaped, sps = sps_2)
    ### Shaped signal spectrum check
    if debug_check:
            cf.spectrum_plot(shaped_upsampled, Fs = sps_2 * sps * fs, title = f'Spectrum after the pulse shaping and upsampling, {sps * sps_2} SPS', plt_en = 1)
    ###
        
    ### Add a LPF to shaped_upsampled
    shaped_upsampled_filtered = cf.apply_fixed_lpf(shaped_upsampled, fs / 2 * (1 + rolloff) * 1.5 , fs * sps_2 * sps) #* 1.5
    ### Shaped upsampled and filtered through LPF signal spectrum check
    if debug_check:
            cf.spectrum_plot(shaped_upsampled_filtered, Fs = sps_2 * sps * fs, title = 'Spectrum after the pulse shaping and upsampling to 40 SPS, then adding a LPF', plt_en = 1)
    ###

    ######## Upconversion
    passband_signal_clean = cf.upconversion(shaped_upsampled_filtered, Fc = fs * sps_2, Fs=fs * sps_2 * sps, plt_en = debug_check)

    for i in tqdm(range(len(snr_arr)), desc = f"Simulating with INL = {inl_en} LSB"):
        
        if noise_en:
            passband_signal = awgn(passband_signal_clean, snr_dB = snr_arr[i])
        else:
            passband_signal = passband_signal_clean
        ######## Downconversion
        baseband_signal = cf.downconversion(passband_signal, Fc = fs * sps_2, Fs = fs * sps_2 * sps, plt_en = debug_check)

        ######### ADC with distortions
        baseband_after_lpf = time_syncronization(shaped_upsampled, baseband_signal, time_delay = 200)
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
        if adc_gain != 0:
            print(np.max(np.abs(downsampled.real)), np.max(np.abs(downsampled.imag)))
            downsampled = cf.quantizer(downsampled, resolution = 8, gain = adc_gain)

        ######## Matched filter
        downsampled_shaped = cf.pulse_shaping(downsampled, rolloff, filter_span, sps, ts, fs * sps, plt_en=0)
        
        ### Correlation and downsampling
        recovered = time_syncronization(up_signal, downsampled_shaped, time_delay = 128)
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
        #final_symbols = cf.dd_lms_equalizer(downsampled, qam, num_taps=31, mu=0.05)
        #final_symbols = cf.dd_lms_equalizer(downsampled, symbol_signal, qam, num_taps=21, mu=0.005, train_len = len(downsampled))



        final_nmse = nmse_calc(symbol_signal, final_symbols)
        nmse_final_arr[i] = final_nmse

        ####### Demapping
        demodulated_bits = qam.demodulate(final_symbols, "hard")
        ber = cf.ber_calc(bits[20000:-5000], demodulated_bits[20000:-5000])
        bers[i] = ber
        
    return bers, nmse_final_arr, final_symbols, dac_gain, adc_gain

def ber_gain_plot(ber_array, nmse_array, gain_array, mod_order, title, title1):
    plt.figure(8)
    plt.plot(gain_array, ber_array, marker = 'o', color = 'red')
    plt.grid()
    #title = f'{mod_order}_QAM_BER(GAIN)_ADC'
    plt.title(title)
    plt.xlabel('Gain')
    plt.ylabel('BER')
    #plt.yscale('log')
    plt.savefig(f'{title}_{mod_order}_QAM.png')
    plt.show()

    plt.figure(9)
    plt.plot(gain_array, nmse_array, marker = 'o', color = 'green')
    plt.grid()
    #title = f'{mod_order}_QAM_NMSE(GAIN)_ADC'
    plt.title(title1)
    plt.xlabel('Gain')
    plt.ylabel('NMSE')
    plt.savefig(f'{title1}_{mod_order}_QAM.png')
    plt.show()


def get_snr_from_ber(target_ber: float, snr_array: list, ber_array: list) -> float:

    snr_arr = np.array(snr_array)
    ber_arr = np.array(ber_array)
    
    valid_indices = np.where(ber_arr > 0)[0]
    if len(valid_indices) < 2:
        return np.nan 
        
    valid_snr = snr_arr[valid_indices]
    valid_ber = ber_arr[valid_indices]
    
    log_ber = np.log10(valid_ber)
    log_target = np.log10(target_ber)
    
    log_ber_asc = log_ber[::-1]
    snr_asc = valid_snr[::-1]
    
    if log_target < np.min(log_ber_asc) or log_target > np.max(log_ber_asc):
        print(f"Warning: target_ber = {target_ber:.1e} is out of the range!")
        
    estimated_snr = np.interp(log_target, log_ber_asc, snr_asc)
    
    return float(estimated_snr)

def main():
    # Parameters
    BITS_NUM = 12_000_00 #6_000_000 #1_000_002
    MOD_ORDER = 64
    F_SYM = 10e3 # FS / SPS
    FS = 10e3 # for SPS = 1 !!!
    SPS = 2
    SPS_2 = 10 * 2 #was not there
    TS = 1 / F_SYM
    ROLLOFF = 0.2 #0.125
    FILTER_SPAN = 64 # added was w/o * 2
    DATA_SAVE = 0
    DEBUG_CHECK = 0
    INL_COEFF = (5/8)
    INL_VAL = 2 #* INL_COEFF
    MODEL_APPLY = 1
    predist_64_mlp = 0
    predist_32_mlp = 0
    predist_64_kan = 0
    predist_32_kan = 0
    with_inl_64 = 1
    with_inl_32 = 1
    without_inl_64 = 1
    without_inl_32 = 1
    SEED = 1000
    snr_arr = np.arange(14, 34, 1)



    ### Without pre-distorter
    bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER, sps = SPS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = DEBUG_CHECK, data_save = DATA_SAVE, model_apply = 0, seed = SEED)
    
    gains = np.linspace(14, 18, 50)
    bers_dac_64_qam = []
    bers_dac_32_qam = []
    nmse_dac_64_qam = []
    nmse_dac_32_qam = []

    #for g in gains:
    DAC_GAIN_64_QAM = 15 / 9 * 0.85#2.2 #* 0.8#2.05 * 0.75 #2.265
    ADC_GAIN_64_QAM = 127 / 15 #0#10 * 0.75# 20.5

    if without_inl_64:
        ### Simulation without INL
        bers_no_inl_64_qam, nmse_final_arr_no_inl_64_qam, final_symbols_64_qam_no_inl, dac_gain, adc_gain = simulate_channel_and_rx(bits, qam,
                            shaped_signal, up_signal, symbol_signal, 
                            snr_arr, inl_en = 0, dac_gain = DAC_GAIN_64_QAM, adc_gain = ADC_GAIN_64_QAM, sps = SPS,
                            sps_2 = SPS_2, fs = FS,
                            rolloff = ROLLOFF, filter_span = FILTER_SPAN,
                            ts = TS, mod_order = MOD_ORDER, debug_check = DEBUG_CHECK, noise_en = 1, data_save = 0)
            #bers_dac_64_qam.append(bers_no_inl_64_qam)
            #nmse_dac_64_qam.append(nmse_final_arr_no_inl_64_qam)

    if with_inl_64:
        #DAC_GAIN_64_QAM *= 0.8
        #ADC_GAIN_64_QAM *= 0.5
        ### Simulation with INL
        bers_with_inl_64_qam, nmse_final_arr_with_inl_64_qam, final_symbols_64_qam_with_inl, _, _ = simulate_channel_and_rx(bits, qam, shaped_signal,
                up_signal, symbol_signal, 
                snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN_64_QAM, adc_gain = ADC_GAIN_64_QAM, 
                sps = SPS, sps_2 = SPS_2, fs = FS,
                rolloff = ROLLOFF, filter_span = FILTER_SPAN,
                ts = TS, mod_order = MOD_ORDER, debug_check = DEBUG_CHECK, 
                noise_en = 1, data_save = DATA_SAVE, model_apply = 0)

    if predist_64_mlp:
        ### With pre-distorter MLP
        bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER, sps = SPS,
                rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = 0, data_save = DATA_SAVE, model_apply = 1, inl_val = INL_VAL, seed = SEED)
        
        ### Simulation with INL and pre-distorter MLP
        bers_with_inl_64_qam_mlp, nmse_final_arr_with_inl_64_qam_mlp, final_symbols_64_qam_with_inl_mlp, _, _ = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
            snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN_64_QAM, adc_gain = ADC_GAIN_64_QAM, sps = SPS, sps_2 = SPS_2, fs = FS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN,
            ts = TS, mod_order = MOD_ORDER, debug_check = DEBUG_CHECK, noise_en = 1, data_save = DATA_SAVE)


    '''### With pre-distorter KAN
    bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER, sps = SPS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = 0, data_save = DATA_SAVE, model_apply = 2, inl_val = INL_VAL)
    
    ### Simulation with INL and pre-distorter KAN
    bers_with_inl_64_qam_kan, nmse_final_arr_with_inl_64_qam_kan, final_symbols_64_qam_with_inl_kan, _, _ = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
        snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN, adc_gain = ADC_GAIN, sps = SPS, sps_2 = SPS_2, fs = FS,
        rolloff = ROLLOFF, filter_span = FILTER_SPAN,
        ts = TS, mod_order = MOD_ORDER, debug_check = DEBUG_CHECK, noise_en = 1, data_save = DATA_SAVE)'''

    '''DAC_GAIN = 2.928 if MOD_ORDER == 64 else 4.179
    ADC_GAIN = 14 if MOD_ORDER == 64 else 11.3425'''
    
    ### Without pre-distorter 32 QAM
    bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER // 2, sps = SPS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = DEBUG_CHECK, data_save = DATA_SAVE, model_apply = 0, seed = SEED)

    #for g in gains:
    DAC_GAIN_32_QAM = 2.2 #* 0.9#3.3# * 0.8 #3 * 0.75 #3.2
    ADC_GAIN_32_QAM = 127 / 15#0#20.5#10 * 0.75 #20.4 #20.5 # 20.5

    if without_inl_32:
        ### Simulation without INL
        bers_no_inl_32_qam, nmse_final_arr_no_inl_32_qam, final_symbols_32_qam_no_inl, dac_gain, adc_gain = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
                        snr_arr, inl_en = 0, dac_gain = DAC_GAIN_32_QAM, adc_gain = ADC_GAIN_32_QAM, sps = SPS,
                        sps_2 = SPS_2, fs = FS,
                        rolloff = ROLLOFF, filter_span = FILTER_SPAN,
                        ts = TS, mod_order = MOD_ORDER // 2, debug_check = DEBUG_CHECK, noise_en = 1)
            #bers_dac_32_qam.append(bers_no_inl_32_qam)
            #nmse_dac_32_qam.append(nmse_final_arr_no_inl_32_qam)

    if with_inl_32:
        #DAC_GAIN_32_QAM *= 0.8 #3.2
        #ADC_GAIN_32_QAM *= 0.5
        ### Simulation with INL
        bers_with_inl_32_qam, nmse_final_arr_with_inl_32_qam, final_symbols_32_qam_with_inl, _, _ = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
                snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN_32_QAM, adc_gain = ADC_GAIN_32_QAM, 
                sps = SPS, sps_2 = SPS_2, fs = FS,
                rolloff = ROLLOFF, filter_span = FILTER_SPAN,
                ts = TS, mod_order = MOD_ORDER // 2, debug_check = DEBUG_CHECK, 
                noise_en = 1, data_save = DATA_SAVE, model_apply = 0)

    if predist_32_mlp:
        ### With pre-distorter MLP
        bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER // 2, sps = SPS,
                rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = 0, data_save = DATA_SAVE, model_apply = 1, inl_val = INL_VAL, seed = SEED)
        
        ### Simulation with INL and pre-distorter MLP
        bers_with_inl_32_qam_mlp, nmse_final_arr_with_inl_32_qam_mlp, final_symbols_32_qam_with_inl_mlp, *_ = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
            snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN_32_QAM, adc_gain = ADC_GAIN_32_QAM, sps = SPS, sps_2 = SPS_2, fs = FS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN,
            ts = TS, mod_order = MOD_ORDER // 2, debug_check = DEBUG_CHECK, noise_en = 1, data_save = DATA_SAVE)
    

    '''### With pre-distorter KAN
    bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(bits_num = BITS_NUM, mod_order = MOD_ORDER // 2, sps = SPS,
            rolloff = ROLLOFF, filter_span = FILTER_SPAN, fs = FS, ts = TS, debug_check = 0, data_save = DATA_SAVE, model_apply = 2, inl_val = INL_VAL)
    
    ### Simulation with INL and pre-distorter KAN
    bers_with_inl_64_qam_kan, nmse_final_arr_with_inl_64_qam_kan, final_symbols_32_qam_with_inl_kan, _, _ = simulate_channel_and_rx(bits, qam, shaped_signal, up_signal, symbol_signal, 
        snr_arr, inl_en = INL_VAL, dac_gain = DAC_GAIN, adc_gain = ADC_GAIN, sps = SPS, sps_2 = SPS_2, fs = FS,
        rolloff = ROLLOFF, filter_span = FILTER_SPAN,
        ts = TS, mod_order = MOD_ORDER // 2, debug_check = DEBUG_CHECK, noise_en = 1, data_save = DATA_SAVE)'''
    

    DC = 'ADC_no_work_point'
    #ber_gain_plot(bers_dac_64_qam, nmse_dac_64_qam, gains, mod_order = MOD_ORDER, title = f'BER(gain) in {DC}', title1 = f'NMSE(gain) in {DC}')
    #ber_gain_plot(bers_dac_32_qam, nmse_dac_32_qam, gains, mod_order = MOD_ORDER // 2, title = f'BER(gain) in {DC}', title1 = f'NMSE(gain) in {DC}')
    
    FEC_LIMIT = 3.84e-3
    
    snr_ideal = get_snr_from_ber(FEC_LIMIT, snr_arr, bers_no_inl_32_qam)
    snr_ideal_2 = get_snr_from_ber(FEC_LIMIT, snr_arr, bers_no_inl_64_qam)

    snr_distorted = get_snr_from_ber(FEC_LIMIT, snr_arr, bers_with_inl_32_qam)
    snr_distorted_2 = get_snr_from_ber(FEC_LIMIT, snr_arr, bers_with_inl_64_qam)
    
    
    penalty = snr_distorted - snr_ideal
    penalty_2 = snr_distorted_2 - snr_ideal_2
    print(f"Penalty on FEC 32 QAM = {FEC_LIMIT}: {penalty:.2f} dB")
    print(f"Penalty on FEC 64 QAM = {FEC_LIMIT}: {penalty_2:.2f} dB")

    plt.figure(10)
    plt.plot(snr_arr, bers_no_inl_64_qam, marker = 'o', color = 'red', label = f'{MOD_ORDER} QAM, without INL')
    if with_inl_64:
        plt.plot(snr_arr, bers_with_inl_64_qam, marker = 'o', color = 'blue', label = f'{MOD_ORDER} QAM, with INL {INL_VAL} LSB')
    if predist_64_mlp:
        plt.plot(snr_arr, bers_with_inl_64_qam_mlp, marker = 'o', color = 'green', label = f'{MOD_ORDER} QAM, with INL {INL_VAL} LSB + MLP')
    #plt.plot(snr_arr, bers_with_inl_64_qam_kan, marker = 'o', color = 'purple', label = f'{MOD_ORDER} QAM, with INL {INL_VAL} LSB + KAN')

    plt.plot(snr_arr, bers_no_inl_32_qam, marker = 's', color = 'red', label = f'{MOD_ORDER // 2} QAM, without INL')
    if with_inl_32:
        plt.plot(snr_arr, bers_with_inl_32_qam, marker = 's', color = 'blue', label = f'{MOD_ORDER // 2} QAM, with INL {INL_VAL} LSB')
    if predist_32_mlp:
        plt.plot(snr_arr, bers_with_inl_32_qam_mlp, marker = 's', color = 'green', label = f'{MOD_ORDER // 2} QAM, with INL {INL_VAL} LSB + MLP')
    #plt.plot(snr_arr, bers_with_inl_32_qam_kan, marker = '+', color = 'purple', label = f'{MOD_ORDER // 2} QAM, with INL {INL_VAL} LSB + KAN')


    plt.legend()
    plt.ylabel('BER')
    plt.ylim(bottom = 1e-5, top = 1e-2)
    plt.axhline(y = FEC_LIMIT)
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.title(f'BER(SNR), {MOD_ORDER // 2}/{64} QAM, INL {INL_VAL} LSB')
    plt.grid()
    plt.savefig(f'BER(SNR)_{MOD_ORDER // 2}_{64}_QAM_INL_{INL_VAL}_LSB.png')
    plt.show()
    exit()

    plt.figure(11)
    plt.plot(snr_arr, nmse_final_arr_no_inl, marker = 'o', color = 'purple', label = f'{64} QAM, without INL')
    plt.plot(snr_arr, nmse_final_arr_with_inl_64_qam, marker = 'o', color = 'orange', label = f'{64} QAM, with INL {INL_VAL / INL_COEFF} LSB')
    plt.plot(snr_arr, nmse_final_arr_no_inl_32_qam, marker = 'o', color = 'red', label = f'{MOD_ORDER} QAM, with INL {INL_VAL/INL_COEFF} LSB')
    plt.plot(snr_arr, nmse_final_arr_with_inl_32_qam, marker = 'o', color = 'green', label = f'{MOD_ORDER} QAM, with INL {INL_VAL/INL_COEFF} LSB')
    #plt.plot(snr_arr, nmse_final_arr_with_inl_predist, marker = 'o', color = 'black', label = f'{MOD_ORDER} QAM, with INL {INL_VAL / INL_COEFF} LSB and pre-distorter MLP')
    #plt.plot(snr_arr, nmse_final_arr_with_inl_predist_kan, marker = 'o', color = 'yellow', label = f'{MOD_ORDER} QAM, with INL {INL_VAL / INL_COEFF} LSB and pre-distorter KAN')
    plt.legend()
    plt.ylabel('NSME')
    plt.xlabel('SNR')
    plt.title(f'NMSE(SNR), {MOD_ORDER}/{64} QAM, INL {INL_VAL / INL_COEFF / 1.2} LSB')
    plt.grid()
    plt.savefig(f'NMSE(SNR)_{MOD_ORDER}_{64}_QAM_INL_{INL_VAL / INL_COEFF / 1.2}_LSB.png')
    plt.show()

    cf.constellation_plot(final_symbols_no_inl, MOD_ORDER, '5 bit DAC')
    cf.constellation_plot(final_symbols_with_inl, MOD_ORDER, f'5 bit DAC with INL {INL_VAL} LSB')
    #cf.constellation_plot(final_symbols_with_inl_predist, MOD_ORDER, f'5 bit DAC with INL {INL_VAL/ INL_COEFF} LSB and pre-distorter MLP')
    #cf.constellation_plot(final_symbols_with_inl_predist_kan, MOD_ORDER, f'5 bit DAC with INL {INL_VAL/ INL_COEFF} LSB pre-distorter KAN')

if __name__ == "__main__":
    main()
