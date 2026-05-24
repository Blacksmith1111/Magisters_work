import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as mod
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d
from commpy.channels import awgn
import channel_funcs as cf
from tqdm import tqdm
from mlp_model import MLP_model, inference
from efficient_kan_model import inference_kan, KAN_model
import os
from cnn_model_improved import CNN_DPD_Micro, inference_cnn
import matplotlib.ticker as ticker
from numba import njit, prange


DEVICE = 'cuda:0'
model_mlp = MLP_model.to(DEVICE)
total_params_mlp = sum(p.numel() for p in model_mlp.parameters())
print(f'Number of parameters in MLP: {total_params_mlp}')
model_kan = KAN_model.to(DEVICE)
total_params_kan = sum(p.numel() for p in model_kan.parameters())
print(f'Number of parameters in KAN: {total_params_kan}')
model_cnn = CNN_DPD_Micro().to(DEVICE)
total_params_cnn = sum(p.numel() for p in model_cnn.parameters())
print(f'Number of parameters in CNN: {total_params_cnn}')

model_params_num = {
    'MLP': total_params_mlp,
    'KAN': total_params_kan,
    'CNN': total_params_cnn
}

### 64 QAM weights
WEIGHTS_FILE_64_QAM_MLP_2_LSB = 'qam_64_mlp_2_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_KAN_2_LSB = 'qam_64_kan_2_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_CNN_2_LSB = 'qam_64_cnn_2_LSB_weights_1.pt'
WEIGHTS_FILE_64_QAM_MLP_4_LSB = 'qam_64_mlp_4_LSB_weights.pt'
WEIGHTS_FILE_64_QAM_KAN_4_LSB = 'qam_64_kan_4_lsb_weights.pt'
WEIGHTS_FILE_64_QAM_CNN_4_LSB = 'qam_64_cnn_4_LSB_weights_1.pt'

### 32 QAM weights
WEIGHTS_FILE_32_QAM_MLP_2_LSB = 'qam_32_mlp_2_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_MLP_4_LSB = 'qam_32_mlp_4_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_KAN_2_LSB = 'qam_32_kan_2_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_KAN_4_LSB = 'qam_32_kan_4_LSB_weights.pt'
WEIGHTS_FILE_32_QAM_CNN_2_LSB = 'qam_32_cnn_2_LSB_weights_1.pt'
WEIGHTS_FILE_32_QAM_CNN_4_LSB = 'qam_32_cnn_4_LSB_weights_1.pt'


SIMULATION_WEIGHTS = {
    64: {
        'MLP': {2: WEIGHTS_FILE_64_QAM_MLP_2_LSB, 4: WEIGHTS_FILE_64_QAM_MLP_4_LSB},
        'KAN': {2: WEIGHTS_FILE_64_QAM_KAN_2_LSB, 4: WEIGHTS_FILE_64_QAM_KAN_4_LSB},
        'CNN': {2: WEIGHTS_FILE_64_QAM_CNN_2_LSB, 4: WEIGHTS_FILE_64_QAM_CNN_4_LSB} 
    },
    32: {
        'MLP': {2: WEIGHTS_FILE_32_QAM_MLP_2_LSB, 4: WEIGHTS_FILE_32_QAM_MLP_4_LSB},
        'KAN': {2: WEIGHTS_FILE_32_QAM_KAN_2_LSB, 4: WEIGHTS_FILE_32_QAM_KAN_4_LSB},
        'CNN': {2: WEIGHTS_FILE_32_QAM_CNN_2_LSB, 4: WEIGHTS_FILE_32_QAM_CNN_4_LSB}
    }
}

def complexity_vs_penalty_plot(results, fec_snr, model_params, snr_arr, folder_name,
                                inl_vals, mod_orders, models_to_test, fec_limit):

    COLOR_MAP  = {'MLP': 'green', 'CNN': 'goldenrod', 'KAN': 'purple'}
    MARKER_MAP = {'MLP': 'o', 'CNN': 'o', 'KAN': 'o'}

    for inl_val in inl_vals:
        for mod_order in mod_orders:
            fig, ax = plt.subplots(figsize=(7, 6))

            ideal_snr = fec_snr.get((mod_order, 0, 'Ideal'), np.nan)
            plotted = False

            for model_name in models_to_test:
                n_params = model_params.get(model_name)
                if n_params is None:
                    continue
                snr_val = fec_snr.get((mod_order, inl_val, model_name), np.nan)
                if np.isnan(snr_val) or np.isnan(ideal_snr):
                    continue

                penalty = snr_val - ideal_snr
                color = COLOR_MAP.get(model_name, 'black')
                marker = MARKER_MAP.get(model_name, 'o')

                ax.scatter(n_params, penalty,
                           color=color, marker=marker, s=150,
                           zorder=5, edgecolors='black', linewidths=0.8,
                           label=model_name)
                ax.annotate(
                    f'{model_name}\n({n_params:,} параметров)\n{penalty:+.2f} дБ',
                    xy=(n_params, penalty),
                    xytext=(12, 10), textcoords='offset points',
                    fontsize=9, color=color,
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', ec=color, alpha=0.88)
                )
                plotted = True

            ax.axhline(0, color='red', linestyle='--', linewidth=1.4, alpha=0.7, label='Идеальный случай (без штрафа)')
            ax.set_title(f'{mod_order}-QAM | ИНЛ {inl_val} МЗР', fontsize=13, fontweight='bold')
            ax.set_xlabel('Сложность модели (Количество параметров)', fontsize=11)
            ax.set_ylabel('Штраф FEC (дБ)', fontsize=11)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            
            ax.legend(
                loc='lower left',
                fontsize=9,
                labelspacing=1.2,
                handleheight=1.8,
                borderpad=1.0,
                framealpha=0.9
            )
            ax.grid(True, linestyle='--', alpha=0.5)
            if plotted:
                ax.margins(x=0.35, y=0.35)

            fig.tight_layout()
            save_path = os.path.join(
                folder_name,
                f'Complexity_vs_FEC_Penalty_{mod_order}QAM_INL{inl_val}LSB.png'
            )
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)


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
    signal = signal / rms_calc(signal) * constellation_rms
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
    plt.stem(signal_1.real[mid : mid + 100], linefmt='r-', label = f'Исходный, действ. часть, nmse = {nmse:.2f}')
    plt.stem(signal_2.real[mid : mid + 100], linefmt='g-', label = f'Обработанный, действ. часть, nmse = {nmse:.2f}')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

    plt.figure(101)
    plt.stem(signal_1.imag[mid : mid + 100], linefmt='r-', label = f'Исходный, мним. часть, nmse = {nmse:.2f}')
    plt.stem(signal_2.imag[mid : mid + 100], linefmt='g-', label = f'Обработанный, мним. часть, nmse = {nmse:.2f}')
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
    title = 'Сигнал 4 SPS до формирования импульса;\nСигнал 4 SPS после согласованной фильтрации'
    compare_2_signals(up_signal, recovered, title)

def generate_tx_base(bits_num, mod_order, sps, rolloff, filter_span, fs, ts, debug_check = 1,
                      data_save = 0, model_apply = 0, inl_val = 2, seed = 100):
    np.random.seed(seed)
    bits = np.random.randint(0, 2, bits_num)
    qam = mod.QAMModem(mod_order) if mod_order == 64 else cf.Modulator32QAM()
    
    symbol_signal = qam.modulate(bits)
    up_signal = cf.upsample(symbol_signal, sps)
    if debug_check:
        cf.spectrum_plot(symbol_signal, Fs = fs, title = 'Спектр исходного символьного сигнала', plt_en = 1)
        cf.spectrum_plot(up_signal, Fs = sps * fs, title = 'Спектр исходного сигнала, передискретизированного до 4 SPS', plt_en = 1)
    
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
        shaped_rms = rms_calc(shaped_signal)
        shaped_normalized = shaped_signal / shaped_rms
        if model_apply == 1:
            prediction = inference(shaped_normalized,
                batch_size = len(shaped_normalized),
                model = model_mlp,
                device = DEVICE,
                weights_file = SIMULATION_WEIGHTS[mod_order]['MLP'][inl_val]) 
            print(SIMULATION_WEIGHTS[mod_order]['MLP'][inl_val])
        elif model_apply == 2:
            current_max = max(np.max(np.abs(shaped_normalized.real)), np.max(np.abs(shaped_normalized.imag)))
            print(f'With KAN; mod order = {mod_order}; inl val = {inl_val} current max is {current_max}')
            if mod_order == 64:
                current_max = 2.3325 if inl_val == 2 else 2.6355
            else:
                current_max = 2.373 if inl_val == 2 else 2.6844
            print(f'With KAN; mod order = {mod_order}; inl val = {inl_val} current max is {current_max}')

            prediction = inference_kan(
                shaped_normalized,
                batch_size=8192,
                model=model_kan,
                device=DEVICE,
                weights_file=SIMULATION_WEIGHTS[mod_order]['KAN'][inl_val],
                max_val=current_max
            )
            print(SIMULATION_WEIGHTS[mod_order]['KAN'][inl_val])
        elif model_apply == 3:
             
            pred_complex = inference_cnn(
                shaped_normalized,
                batch_size=32, 
                model=model_cnn,
                device=DEVICE,
                weights_file=SIMULATION_WEIGHTS[mod_order]['CNN'][inl_val],
                seq_len=128 
            )
            
            prediction = np.column_stack((pred_complex.real, pred_complex.imag))

        prediction *= shaped_rms # Switched back to the initial rms
        de_centered_signal = prediction
        
        shaped_signal = de_centered_signal[:, 0] + 1j * de_centered_signal[:, 1]
        
    else:
        print('Not using the model')

    if data_save:
        print(f'global_rms for targets is {global_rms}')
        centered_shaped = np.copy(shaped_signal)# - np.mean(shaped_signal)
        centered_shaped /= rms_calc(centered_shaped)
        np.save(f'model_targets_{mod_order}_qam.npy', centered_shaped)
    
    ### Shaped signal spectrum check
    cf.spectrum_plot(shaped_signal, Fs = sps * fs, title = 'Спектр после формирования импульса', plt_en = debug_check)
    ###
    if debug_check:
        ### TX Pulse shaping block check, NMSE = -67 dB
        pulse_shaping_check(shaped_signal, up_signal, rolloff, filter_span, sps, fs, ts)
        ###
    return bits, qam, symbol_signal, up_signal, shaped_signal

def simulate_channel_and_rx(bits, qam, shaped_signal_pure, up_signal, symbol_signal, 
                            snr_arr, inl_en, dac_gain, adc_gain, sps, sps_2, fs, rolloff,
                            filter_span, ts, mod_order, noise_en = 1, debug_check = 1, data_save = 0,
                            phase_noise_en = 0, delta_nu = 200e3):
    
    bers = np.zeros_like(snr_arr, dtype = np.float64)
    nmse_final_arr = np.zeros_like(bers)

    global_rms = rms_calc(shaped_signal_pure)
    ######### DAC with the distortions
    if dac_gain != 0:
            factor = 0
            if inl_en == 0:
                pass
            elif inl_en == 2: 
                factor = 5/8
            else:
                factor = 4/8
            #print(f'Max values of the real and imag components before the DAC: {np.max(np.abs(shaped_signal_pure.real))}; {np.max(np.abs(shaped_signal_pure.imag))}') 
            current_shaped = cf.quantizer(shaped_signal_pure, resolution = 5, gain = dac_gain, inl_en = inl_en * factor) 
            #print(f'Max values of the real and imag components after the DAC: {np.max(np.abs(current_shaped.real))}; {np.max(np.abs(current_shaped.imag))}')
    else:
            current_shaped = shaped_signal_pure


    if data_save:
        print('Data saving mode')
        print(np.max(np.abs(current_shaped.real)), np.max(np.abs(current_shaped.imag)))    
        model_objects = cf.quantizer(current_shaped, resolution = 8, gain = adc_gain)
        print(np.max(np.abs(model_objects.real)), np.max(np.abs(model_objects.imag)))
        model_objects /= (adc_gain * dac_gain)
        print(f'{np.max(np.abs(model_objects.real))}, {np.max(np.abs(model_objects.imag))}: After the de-embedding')

        #model_objects -= np.mean(model_objects)
        model_objects /= rms_calc(model_objects) 
        print(f'global_rms for objects is {global_rms}')
        np.save(f'model_objects_{mod_order}_qam_INL_{inl_en}_LSB.npy', model_objects)
        return 1, 1, 1, 1, 1


    ### Upsampling to 40 SPS
    shaped_upsampled = cf.upsample(current_shaped, sps = sps_2)
    ### Shaped signal spectrum check
    if debug_check:
            cf.spectrum_plot(shaped_upsampled, Fs = sps_2 * sps * fs, title = f'Спектр после формирования импульса и передискретизации, {sps * sps_2} SPS', plt_en = 1)
    ###
        
    ### Add a LPF to shaped_upsampled
    shaped_upsampled_filtered = cf.apply_fixed_lpf(shaped_upsampled, fs / 2 * (1 + rolloff) * 1.5 , fs * sps_2 * sps) #* 1.5
    ### Shaped upsampled and filtered through LPF signal spectrum check
    if debug_check:
            cf.spectrum_plot(shaped_upsampled_filtered, Fs = sps_2 * sps * fs, title = 'Спектр после передискретизации до 40 SPS и применения ФНЧ', plt_en = 1)
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
            shaped_up_energy, recovered_energy = energy_calc(shaped_upsampled), energy_calc(recovered)
            print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')
            shaped_upsampled, recovered = normalize_energy(shaped_upsampled), normalize_energy(recovered)
            shaped_up_energy, recovered_energy = energy_calc(shaped_upsampled), energy_calc(recovered)
            print(f'Signals energies: {shaped_up_energy}; {recovered_energy}')

            title = f'Сформированный сигнал ({sps * sps_2} SPS) до ФНЧ;\nВосстановленный сигнал ({sps * sps_2} SPS) после ФНЧ с синхронизацией'
            compare_2_signals(shaped_upsampled, recovered, title)

        ### ADC quantizer
        if adc_gain != 0:
            #print(f'Max values of the real and imag components before the ADC: {np.max(np.abs(downsampled.real))}; {np.max(np.abs(downsampled.imag))}')
            downsampled = cf.quantizer(downsampled, resolution = 8, gain = adc_gain)
            #print(f'Max values of the real and imag components after the ADC: {np.max(np.abs(downsampled.real))}; {np.max(np.abs(downsampled.imag))}')

        ######## Matched filter
        downsampled_shaped = cf.pulse_shaping(downsampled, rolloff, filter_span, sps, ts, fs * sps, plt_en=0)
        
        ### Correlation and downsampling
        recovered = time_syncronization(up_signal, downsampled_shaped, time_delay = 128)
        downsampled = cf.downsample(recovered, sps)
        if debug_check:
            recovered = cf.upsample(downsampled, sps)
            print(f'Initial upsampled on {sps} SPS signal and Recovered signal after the matched filtering on {sps} SPS nmse calculation')
            up_energy, recovered_energy = energy_calc(up_signal), energy_calc(recovered)
            print(f'Signals energies: {up_energy}; {recovered_energy}')
            up_signal, recovered = normalize_energy(up_signal), normalize_energy(recovered)
            up_energy, recovered_energy = energy_calc(up_signal), energy_calc(recovered)
            print(f'Signals energies: {up_energy}; {recovered_energy}')
            title = f'Исходный сигнал ({sps} SPS);\nВосстановленный сигнал после согласованной фильтрации ({sps} SPS)'
            compare_2_signals(up_signal, recovered, title)

        ######## Phase noise adding
        if phase_noise_en:
            _sigma_dphi = np.sqrt(2 * np.pi * delta_nu / fs)  # fs = baud_rate
            _pn = np.cumsum(np.random.normal(0, _sigma_dphi, len(downsampled)))
            # Frequency offset
            _n = np.arange(len(downsampled))
            _cfo = 300e6 
            downsampled = downsampled * np.exp(1j * (2 * np.pi * _cfo * _n / fs + _pn))

        ######## Getting symbols back on SPS = 1
        if phase_noise_en:
            if mod_order == 32 and inl_en == 4:
                downsampled = downsampled * np.exp(-1j * 2 * np.pi * _cfo * np.arange(len(downsampled)) / fs)
            else:
                downsampled, _ = cfo_estimate_and_correct(downsampled, fs)
            
            rms_in = rms_calc(downsampled) #np.sqrt(np.mean(np.abs(downsampled)**2))
            sym_norm = downsampled / rms_in

            if mod_order == 64:
                _m = int(np.sqrt(mod_order))
                _lv = np.arange(-(_m-1), _m, 2)
                _re, _im = np.meshgrid(_lv, _lv)
                const_bps = (_re + 1j * _im).flatten().astype(complex)
            else:
                const_bps = np.array(qam.constellation, dtype=complex)
            const_bps /= rms_calc(const_bps) #np.sqrt(np.mean(np.abs(const_bps)**2))

            compensated = bps_phase_compensation(sym_norm, const_bps, B=128, Nw=64, block=5000)
            final_symbols = compensated * cf.qam_constellation_rms_calc(mod_order)
        else:
            #final_symbols = constellation_normalization(downsampled, mod_order)
            final_symbols = cf.dd_lms_equalizer(downsampled, qam, num_taps=31, mu=0.05)
            
        final_nmse = nmse_calc(symbol_signal, final_symbols)
        nmse_final_arr[i] = final_nmse

        ####### Demapping
        demodulated_bits = qam.demodulate(final_symbols, "hard")
        ber = cf.ber_calc(bits[20000:-5000], demodulated_bits[20000:-5000])
        bers[i] = ber
        
    return bers, nmse_final_arr, final_symbols

def ber_gain_plot(ber_array, nmse_array, gain_array, mod_order, title, title1):
    plt.figure(8)
    plt.plot(gain_array, ber_array, marker = 'o', color = 'red')
    plt.grid()
    plt.title(title)
    plt.xlabel('Коэффициент усиления')
    plt.ylabel('BER')
    plt.savefig(f'{title}_{mod_order}_QAM.png')
    plt.show()

    plt.figure(9)
    plt.plot(gain_array, nmse_array, marker = 'o', color = 'green')
    plt.grid()
    plt.title(title1)
    plt.xlabel('Коэффициент усиления')
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


@njit(parallel=True, cache=True)
def _bps_cost_numba(symbols_re, symbols_im, const_re, const_im, tp, cost):
    N = symbols_re.shape[0]
    B = tp.shape[0]
    C = const_re.shape[0]
    for n in prange(N):
        sr = symbols_re[n]
        si = symbols_im[n]
        for b in range(B):
            # exp(j*tp[b]) = cos(tp[b]) + j*sin(tp[b])
            cos_b = np.cos(tp[b])
            sin_b = np.sin(tp[b])
            rot_re = sr * cos_b - si * sin_b
            rot_im = sr * sin_b + si * cos_b
            min_d = 1e18
            for c in range(C):
                dr = rot_re - const_re[c]
                di = rot_im - const_im[c]
                d  = dr*dr + di*di
                if d < min_d:
                    min_d = d
            cost[n, b] = min_d


def bps_phase_compensation(symbols, const, B = 128, Nw = 64, block = 5000):
    N = len(symbols)
    tp = np.linspace(-np.pi/4, np.pi/4, B, endpoint = False).astype(np.float64)

    cost = np.empty((N, B), dtype=np.float32)
    _bps_cost_numba(
        symbols.real.astype(np.float64),
        symbols.imag.astype(np.float64),
        const.real.astype(np.float64),
        const.imag.astype(np.float64),
        tp, cost
    )

    smoothed = uniform_filter1d(cost, size = Nw, axis = 0)
    phi_raw = tp[np.argmin(smoothed, axis = 1)]

    phi_full = np.empty(N)
    phi_offset = 0.0
    for blk in range(int(np.ceil(N / block))):
        s = blk * block
        e = min(s + block, N)
        phi_uw = np.unwrap(phi_raw[s:e] * 4) / 4
        if blk > 0:
            raw_jump = phi_uw[0] - phi_offset
            phi_uw = phi_uw - np.round(raw_jump / (np.pi/2)) * (np.pi/2)
        phi_full[s:e] = phi_uw
        phi_offset = phi_full[e - 1]

    return symbols * np.exp(1j * phi_full)

def cfo_estimate_and_correct(symbols, fs, M=4):
    N = len(symbols)
    powered = symbols ** M

    spectrum = np.fft.fft(powered, n = N)
    freqs = np.fft.fftfreq(N, d = 1.0 / fs)
    peak_idx = np.argmax(np.abs(spectrum))
    f_cfo_M = freqs[peak_idx]
    f_cfo = f_cfo_M / M

    n = np.arange(N)
    corrected = symbols * np.exp(-1j * 2 * np.pi * f_cfo * n / fs)

    return corrected, f_cfo

def main():
    TEST_MOD_ORDERS = [64, 32]
    TEST_INL_VALS = [2, 4]
    
    # 'MLP', 'KAN', 'CNN'
    MODELS_TO_TEST = ['CNN', 'MLP', 'KAN']         
    
    RUN_NO_INL = True              
    RUN_WITH_INL = True
    
    BITS_NUM = 1_200_000            
    F_SYM = 32e9 # (FS / SPS)
    FS = 32e9  # SPS = 1
    SPS = 2
    SPS_2 = 10 * 2
    TS = 1 / F_SYM
    ROLLOFF = 0.2 
    FILTER_SPAN = 64
    DATA_SAVE = 0
    DEBUG_CHECK = 0
    SEED = 100
    PHASE_NOISE_EN = 1
    DELTA_NU = 200e3
    snr_arr = np.arange(14, 30, 1)

    GAINS = {
        64: {'dac': 15 / 9, 'adc': 127 / 15},
        32: {'dac': 2.2, 'adc': 127 / 15}
    }

    MODEL_FLAGS = {'MLP': 1, 'KAN': 2, 'CNN': 3}
    
    results = {}

    for mod_order in TEST_MOD_ORDERS:
        print(f"\n{'='*40}\nSTARTING SIMULATION FOR {mod_order}-QAM\n{'='*40}")
        
        dac_gain = GAINS[mod_order]['dac']
        adc_gain = GAINS[mod_order]['adc']

        if RUN_NO_INL:
            print(f"\n>>> Running Baseline: NO INL ({mod_order}-QAM) <<<")
            bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(
                bits_num=BITS_NUM, mod_order=mod_order, sps=SPS, rolloff=ROLLOFF, 
                filter_span=FILTER_SPAN, fs=FS, ts=TS, debug_check=DEBUG_CHECK, 
                data_save=DATA_SAVE, model_apply=0, seed=SEED
            )
            bers, nmses, symbols = simulate_channel_and_rx(
                bits, qam, shaped_signal, up_signal, symbol_signal, snr_arr,
                inl_en=0, dac_gain=dac_gain, adc_gain=adc_gain, sps=SPS, sps_2=SPS_2, fs=FS,
                rolloff=ROLLOFF, filter_span=FILTER_SPAN, ts=TS, mod_order=mod_order, 
                debug_check=DEBUG_CHECK, noise_en=1, data_save=DATA_SAVE,
                phase_noise_en=PHASE_NOISE_EN, delta_nu=DELTA_NU
            )
            results[(mod_order, 0, 'Ideal')] = {'ber': bers, 'nmse': nmses, 'symbols': symbols}

        for inl_val in TEST_INL_VALS:
            if RUN_WITH_INL:
                print(f"\n>>> Running Baseline: WITH INL {inl_val} LSB ({mod_order}-QAM) <<<")
                bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(
                    bits_num=BITS_NUM, mod_order=mod_order, sps=SPS, rolloff=ROLLOFF, 
                    filter_span=FILTER_SPAN, fs=FS, ts=TS, debug_check=DEBUG_CHECK, 
                    data_save=DATA_SAVE, model_apply=0, seed=SEED
                )
                bers, nmses, symbols = simulate_channel_and_rx(
                    bits, qam, shaped_signal, up_signal, symbol_signal, snr_arr,
                    inl_en=inl_val, dac_gain=dac_gain, adc_gain=adc_gain, sps=SPS, sps_2=SPS_2, fs=FS,
                    rolloff=ROLLOFF, filter_span=FILTER_SPAN, ts=TS, mod_order=mod_order, 
                    debug_check=DEBUG_CHECK, noise_en=1, data_save=DATA_SAVE,
                    phase_noise_en=PHASE_NOISE_EN, delta_nu=DELTA_NU
                )
                results[(mod_order, inl_val, 'No_DPD')] = {'ber': bers, 'nmse': nmses, 'symbols': symbols}

            for model_name in MODELS_TO_TEST:
                print(f"\n>>> Running Model: {model_name} | INL {inl_val} LSB ({mod_order}-QAM) <<<")
                
                bits, qam, symbol_signal, up_signal, shaped_signal = generate_tx_base(
                    bits_num=BITS_NUM, mod_order=mod_order, sps=SPS, rolloff=ROLLOFF, 
                    filter_span=FILTER_SPAN, fs=FS, ts=TS, debug_check=DEBUG_CHECK, 
                    data_save=DATA_SAVE, model_apply=MODEL_FLAGS[model_name], 
                    inl_val=inl_val, seed=SEED
                )
                
                bers, nmses, symbols = simulate_channel_and_rx(
                    bits, qam, shaped_signal, up_signal, symbol_signal, snr_arr,
                    inl_en=inl_val, dac_gain=dac_gain, adc_gain=adc_gain, sps=SPS, sps_2=SPS_2, fs=FS,
                    rolloff=ROLLOFF, filter_span=FILTER_SPAN, ts=TS, mod_order=mod_order, 
                    debug_check=DEBUG_CHECK, noise_en=1, data_save=DATA_SAVE,
                    phase_noise_en=PHASE_NOISE_EN, delta_nu=DELTA_NU
                )
                results[(mod_order, inl_val, model_name)] = {'ber': bers, 'nmse': nmses, 'symbols': symbols}


    folder_name = 'Pictures'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    COLOR_MAP = {
        'Ideal':'red',
        'No_DPD':'blue',
        'MLP':'green',
        'CNN':'goldenrod',
        'KAN':'purple'
    }

    FEC_LIMIT = 3.84e-3

    fec_snr = {}
    for (m_order, i_val, m_name), data in results.items():
        snr_at_fec = get_snr_from_ber(FEC_LIMIT, snr_arr.tolist(), data['ber'].tolist())
        fec_snr[(m_order, i_val, m_name)] = snr_at_fec

    print("\n" + "=" * 70)
    print(f"{'ТАБЛИЦА ШТРАФОВ FEC':^70}")
    print(f"{'(Требуемый SNR при BER = 3.84e-3)':^70}")
    print("=" * 70)
    print(f"{'Модуляция':<12} {'INL':>5} {'Модель':<10} {'SNR@FEC':>10} {'Штраф':>10}")
    print("-" * 70)

    for mod_order in TEST_MOD_ORDERS:
        ideal_snr = fec_snr.get((mod_order, 0, 'Ideal'), np.nan)
        for inl_val in TEST_INL_VALS:
            for m_name in ['No_DPD'] + MODELS_TO_TEST:
                snr_val = fec_snr.get((mod_order, inl_val, m_name), np.nan)
                penalty = snr_val - ideal_snr if not np.isnan(snr_val) else np.nan
                penalty_str = f"+{penalty:.2f} дБ" if not np.isnan(penalty) else "Н/Д"
                snr_str = f"{snr_val:.2f} дБ"  if not np.isnan(snr_val) else "Н/Д"
                print(f"{mod_order}-QAM      {inl_val:>5} {m_name:<10} {snr_str:>10} {penalty_str:>10}")
        print("-" * 70)

    for inl_val in TEST_INL_VALS:

        fig, ax = plt.subplots(figsize=(13, 8))

        for (m_order, i_val, m_name), data in results.items():
            if not (i_val == inl_val or m_name == 'Ideal'):
                continue

            marker = 's' if m_order == 32 else 'o'
            ls = '--' if m_order == 32 else '-'
            color = COLOR_MAP.get(m_name, 'black')

            ideal_snr = fec_snr.get((m_order, 0, 'Ideal'), np.nan)
            snr_val = fec_snr.get((m_order, i_val, m_name), np.nan)
            penalty = snr_val - ideal_snr

            if m_name == 'Ideal':
                label = f'{m_order}-QAM  Идеальный случай (без ИНЛ) | SNR@FEC={snr_val:.1f} дБ'
            elif m_name == 'No_DPD':
                label = (f'{m_order}-QAM  ИНЛ {inl_val} МЗР | '
                         f'SNR@FEC = {snr_val:.1f} дБ  Штраф = +{penalty:.1f} дБ')
            else:
                label = (f'{m_order}-QAM  ИНЛ {inl_val} МЗР + {m_name} | '
                         f'SNR@FEC={snr_val:.1f} дБ  Штраф = +{penalty:.1f} дБ')

            ax.plot(snr_arr, data['ber'], marker=marker, linestyle=ls, color=color, label=label)

            if not np.isnan(snr_val):
                ax.axvline(x=snr_val, color=color, linestyle=':', linewidth=1.2, alpha=0.55)

        ax.axhline(y = FEC_LIMIT, color = 'black', linestyle = ':', linewidth = 2, label = f'Предел FEC ({FEC_LIMIT:.2e})')
        ax.set_yscale('log')
        ax.set_ylim(bottom = 1e-5, top = 1e-2)
        ax.set_xlabel('SNR (дБ)', fontsize = 12)
        ax.set_ylabel('BER', fontsize = 12)
        ax.set_title(f'Зависимость BER от SNR | 32-QAM и 64-QAM | ИНЛ {inl_val} МЗР', fontsize = 13)
        ax.legend(loc = 'lower left', fontsize = 9)
        ax.grid(True, which = 'both', ls = '--', alpha = 0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(folder_name, f'BER_Both_QAM_INL_{inl_val}LSB.png'), dpi = 150)
        plt.show()

        fig2, ax2 = plt.subplots(figsize = (13, 7))

        for (m_order, i_val, m_name), data in results.items():
            if not (i_val == inl_val or m_name == 'Ideal'):
                continue

            marker = 's' if m_order == 32 else 'o'
            ls = '--' if m_order == 32 else '-'
            color = COLOR_MAP.get(m_name, 'black')

            if m_name == 'Ideal':
                label = f'{m_order}-QAM  Идеальный случай (без ИНЛ)'
            elif m_name == 'No_DPD':
                label = f'{m_order}-QAM  ИНЛ {inl_val} МЗР'
            else:
                label = f'{m_order}-QAM  ИНЛ {inl_val} МЗР + {m_name}'
            
            ### Save the final constellation
            cf.constellation_plot(data['symbols'], m_order, title = label, 
                save_file = os.path.join(folder_name, f'Constellation_{label}.png'))

            ax2.plot(snr_arr, data['nmse'], marker = marker, linestyle = ls, color = color, label = label)

        ax2.set_xlabel('SNR (дБ)', fontsize = 12)
        ax2.set_ylabel('NMSE (дБ)', fontsize = 12)
        ax2.set_title(f'Зависимость NMSE от SNR | 32-QAM и 64-QAM | ИНЛ {inl_val} МЗР', fontsize = 13)
        ax2.legend(loc = 'lower left', fontsize = 9)
        ax2.grid(True, ls = '--', alpha = 0.6)
        fig2.tight_layout()
        fig2.savefig(os.path.join(folder_name, f'NMSE_Both_QAM_INL_{inl_val}LSB.png'), dpi = 150)
        plt.show()

        fig3, axes = plt.subplots(1, len(TEST_MOD_ORDERS), figsize = (6 * len(TEST_MOD_ORDERS), 6), sharey = False)
        if len(TEST_MOD_ORDERS) == 1:
            axes = [axes]

        for ax3, mod_order in zip(axes, TEST_MOD_ORDERS):
            ideal_snr = fec_snr.get((mod_order, 0, 'Ideal'), np.nan)
            cases  = ['No_DPD'] + MODELS_TO_TEST
            labels = ['Без DPD'] + MODELS_TO_TEST
            penalties = []
            bar_colors = []
            for case in cases:
                snr_val = fec_snr.get((mod_order, inl_val, case), np.nan)
                pen = snr_val - ideal_snr if not np.isnan(snr_val) else 0.0
                penalties.append(pen)
                bar_colors.append(COLOR_MAP.get(case, 'grey'))

            bars = ax3.bar(labels, penalties, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.5)

            for bar, pen in zip(bars, penalties):
                ypos = bar.get_height() + 0.03 if pen >= 0 else bar.get_height() - 0.15
                ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
                         f'+{pen:.2f} дБ' if pen >= 0 else f'{pen:.2f} дБ',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax3.axhline(0, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
            ax3.set_title(f'{mod_order}-QAM | ИНЛ {inl_val} МЗР', fontsize=12)
            ax3.set_ylabel('Штраф FEC (дБ)', fontsize=11)
            ax3.set_xlabel('Предысказитель (Модель)', fontsize=11)
            ax3.grid(axis='y', ls='--', alpha=0.5)

            y_min, y_max = ax3.get_ylim()
            ax3.set_ylim(y_min, y_max * 1.25 if y_max > 0 else y_max)

        fig3.suptitle(
            f'Штраф FEC для разных моделей | ИНЛ {inl_val} МЗР\n'
            f'(SNR относительно идеального, BER = {FEC_LIMIT:.2e})',
            fontsize=13
        )
        fig3.tight_layout()
        fig3.savefig(os.path.join(folder_name, f'FEC_Penalty_INL_{inl_val}LSB.png'), dpi=150)
        plt.show()

    complexity_vs_penalty_plot(
        results = results,
        fec_snr = fec_snr,
        model_params = model_params_num,
        snr_arr = snr_arr,
        folder_name = folder_name,
        inl_vals = TEST_INL_VALS,
        mod_orders = TEST_MOD_ORDERS,
        models_to_test= MODELS_TO_TEST,
        fec_limit = FEC_LIMIT
    )


if __name__ == "__main__":
    main()