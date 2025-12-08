import numpy as np
import matplotlib.pyplot as plt
import channel_funcs
import commpy.modulation as mod


import numpy as np
import commpy.modulation as mod
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter


def constellation_plot(modulated_signal: complex) -> None:
    x = modulated_signal.real
    y = modulated_signal.imag

    plt.figure(1)
    plt.scatter(x, y)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
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
    filter_len = filter_span * sps * 2 + 1
    time_stamps, h = rrcosfilter(filter_len, alpha = rolloff, Ts=Ts, Fs=Fs)
    if normaliztion == 'L2':
        h = h / np.sqrt(np.sum(h**2))  # L2 normalization
    else:
        h = h / np.sum(h)  # L1 normalization
    shaped_signal = np.convolve(upsampled_signal, h, mode = 'full')

    return shaped_signal

def ber_calc(initial_bits, final_bits) -> float:
    return np.sum(np.logical_xor(final_bits, initial_bits)) / len(initial_bits)

def quantizer(signal, resolution: int, S_max: float):
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
    return I_quantized + 1j * Q_quantized, scaling_factor_dac

def ADC(signal, adc_bits, dac_bits):
    scaling_factor_adc = 2**(adc_bits - dac_bits)
    scaled_signal = signal * scaling_factor_adc
    return scaled_signal, scaling_factor_adc

def upconversion(baseband_signal, Fc, Fs):
    t = np.arange(len(baseband_signal)) / Fs
    
    passband_signal_complex = baseband_signal * np.exp(2 * np.pi * Fc * t * 1j)
    passband_signal_real = passband_signal_complex
    return passband_signal_real

def downconversion(passband_signal, Fc, Fs):
    t = np.arange(len(passband_signal)) / Fs
    mixed_signal_complex = passband_signal * np.exp(-2 * np.pi * Fc * t * 1j)
    
    return mixed_signal_complex 

def main():
    ### Parameters
    bits_num = 6 * 10000000
    mod_order = 64
    Fs = 10e3
    Fc = Fs / 4           
    sps = 10             
    f_sym = Fs / sps     
    Ts = 1 / f_sym       
    rolloff = 0.25
    filter_span = 8      
    time_delay_tx = filter_span * sps 
    S_max_dict = {'64QAM': np.sqrt(2*7**2), '32QAM': np.sqrt(3**2 + 5**2)}
    dac_bits = 6
    adc_bits = 8

    ### PBRS
    bits  = np.random.randint(0, 2, bits_num)

    ### QAM modulation
    qam = mod.QAMModem(mod_order)
    modulated_signal = qam.modulate(bits)
    print(f'Modulated signal ({mod_order} QAM): {modulated_signal}')

    ### Upsampling
    upsampled_signal = channel_funcs.upsample(modulated_signal, sps)
    print(f'Signal after upsampling: {upsampled_signal}; shape = {upsampled_signal.shape}')

    ### Pulse shaping (TX)
    shaped_signal = channel_funcs.pulse_shaping(
        upsampled_signal, rolloff=rolloff, filter_span=filter_span, 
        sps=sps, Fs=Fs, Ts=Ts, normaliztion='L2'
    )
    print(f'Signal shape after shaping = {shaped_signal.shape}')
    

    ### Quantizer
    #quantized_signal, scaling_factor_dac = channel_funcs.quantizer(shaped_signal, resolution = dac_bits, S_max = S_max_dict['64QAM'])
    quantized_signal = shaped_signal
    '''### Upconversion
    passband_signal = channel_funcs.upconversion(quantized_signal, Fs = Fs, Fc = Fc)
    
    ### Downconversion
    baseband_signal = channel_funcs.downconversion(passband_signal, Fs = Fs, Fc = Fc)'''

    '''spectrum = np.fft.fftshift(np.fft.fft(passband_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(passband_signal), 1 / Fs))
    spectrum_orig = np.fft.fftshift(np.fft.fft(shaped_signal))
    freqs_orig = np.fft.fftshift(np.fft.fftfreq(len(shaped_signal), 1 / Fs))
    spectrum_back = np.fft.fftshift(np.fft.fft(baseband_signal))
    freqs_back = np.fft.fftshift(np.fft.fftfreq(len(baseband_signal), 1 / Fs))
    plt.figure(figsize=(12, 10))

    # --- 1. Baseband (original shaped) ---
    plt.subplot(3, 1, 1)
    plt.plot(freqs_orig, 20 * np.log10(np.abs(spectrum_orig) + 1e-12))
    plt.title("Spectrum of original baseband signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    # --- 2. Passband (after mixing to carrier) ---
    plt.subplot(3, 1, 2)
    plt.plot(freqs, 20 * np.log10(np.abs(spectrum) + 1e-12))
    plt.title("Spectrum of passband signal (shifted to carrier)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    # --- 3. Downconverted baseband ---
    plt.subplot(3, 1, 3)
    plt.plot(freqs_back, 20 * np.log10(np.abs(spectrum_back) + 1e-12))
    plt.title("Spectrum after downconversion (back to baseband)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()'''


    ### ADC
    #scaled_signal, sacling_factor_adc = channel_funcs.ADC(baseband_signal, adc_bits=adc_bits, dac_bits=dac_bits)
    #scaled_signal = baseband_signal
    scaled_signal = shaped_signal
    ### Signal recovery (RX)
    recovered_signal = channel_funcs.pulse_shaping(
        scaled_signal, rolloff = rolloff, filter_span = filter_span, 
        sps = sps, Fs = Fs, Ts = Ts, normaliztion = 'L2'
    )

    time_delay_rx = 2 * time_delay_tx 
    recovered_signal = recovered_signal[time_delay_rx : time_delay_rx + len(upsampled_signal)]

    ### Downsampling
    downsampled_signal = channel_funcs.downsample(recovered_signal, sps)
    ### Scaling
    #downsampled_signal /= (scaling_factor_dac * sacling_factor_adc)
    print(f'Signal after downsampling: {downsampled_signal}')
    ### Demapping
    demodulated_bits = qam.demodulate(downsampled_signal, 'hard') 
    print(f'Initial bits: {bits};\nDemodulated bits: {demodulated_bits}')
    ### BER calculating
    ber = channel_funcs.ber_calc(bits, demodulated_bits)
    print(f'BER = {ber}')


if __name__ == '__main__':
    main()