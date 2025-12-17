import numpy as np
import matplotlib.pyplot as plt
import channel_funcs
import commpy.modulation as mod
from scipy import signal as sig
from commpy.channels import awgn


def main():
    ### Parameters
    bits_num = 6 * 1_0000
    mod_order = 64
    Fs = 10e3
    Fc = Fs / 4           
    sps = 10           
    f_sym = Fs / sps     
    Ts = 1 / f_sym       
    rolloff = 0.25
    filter_span = 10      
    dac_bits = 6
    adc_bits = 8
    ADCEn = 1
    quantizerEn = 1
    up_down_convEn = 1
    snr_db = 10

    ### PBRS
    bits  = np.random.randint(0, 2, bits_num)

    ### QAM modulation
    qam = mod.QAMModem(mod_order)
    signal = qam.modulate(bits)
    print(f'Modulated signal ({mod_order} QAM): {signal}')

    ### Upsampling
    up_signal = channel_funcs.upsample(signal, sps)
    print(f'Signal after upsampling: {up_signal}; shape = {up_signal.shape}')

    ### Pulse shaping (TX)
    shaped_signal = channel_funcs.pulse_shaping(
        up_signal, rolloff=rolloff, filter_span=filter_span, 
        sps=sps, Fs=Fs, Ts=Ts, normaliztion='L2'
    )
    print(f'Signal shape after shaping = {shaped_signal.shape}')
    title  = 'Original baseband signal spectrum'
    channel_funcs.spectrum_plot(shaped_signal, Fs, title = title)

    if quantizerEn:
        ### Quantizer
        quantized_signal, scaling_factor_dac = channel_funcs.quantizer(shaped_signal, resolution = dac_bits)
    else:
        quantized_signal = shaped_signal
    if up_down_convEn:    
        ### Upconversion
        passband_signal = channel_funcs.upconversion(quantized_signal, Fs = Fs, Fc = Fc)
        title = 'Passband signal spectrum'
        channel_funcs.spectrum_plot(signal = passband_signal, Fs = Fs, title = title)
        ### AWGN
        #passband_signal = awgn(passband_signal, snr_dB = snr_db)
        ### Downconversion
        baseband_signal = channel_funcs.downconversion(passband_signal, Fs = Fs, Fc = Fc, B = 1200)
        title = 'Baseband (shifted back) signal spectrum'
        channel_funcs.spectrum_plot(signal=baseband_signal, Fs=Fs, title = title)
    else:
        baseband_signal = shaped_signal

    if ADCEn:
        ### ADC
        scaled_signal, sacling_factor_adc = channel_funcs.ADC(baseband_signal, adc_bits=adc_bits, dac_bits=dac_bits)
    else:
        scaled_signal = baseband_signal

    ### Signal recovery (RX)
    recovered_signal = channel_funcs.pulse_shaping(
        scaled_signal, rolloff = rolloff, filter_span = filter_span, 
        sps = sps, Fs = Fs, Ts = Ts, normaliztion = 'L2'
    )

    plt.figure(5)
    plt.stem(recovered_signal.real)
    plt.grid()
    plt.title('Recovered signal real part')
    #plt.show()
    
    correlation = sig.correlate(up_signal, recovered_signal, mode='full')
    lags = sig.correlation_lags(up_signal.size, recovered_signal.size, mode='full')

    delay_index = np.argmax(correlation)
    actual_delay_samples = lags[delay_index]

    print(f"Calculated delay (in samples): {actual_delay_samples}")
    rec_without_delay = recovered_signal[-actual_delay_samples:-actual_delay_samples + len(up_signal)]
    plt.figure(90)
    plt.stem(rec_without_delay.real)
    plt.grid()
    plt.title("Shifted recovered signal real part")
    
    ### Downsampling
    downsampled_signal = channel_funcs.downsample(rec_without_delay, sps)
    ### Scaling
    downsampled_signal /= (scaling_factor_dac * sacling_factor_adc)
    
    ### NMSE calculation
    nmse = 20 * np.log10(np.sum((np.abs(downsampled_signal - signal)) ** 2) / np.sum(np.abs(signal) ** 2))
    print(f'NMSE between the received and the transmitted signal: {nmse} db')
    ### Demapping
    demodulated_bits = qam.demodulate(downsampled_signal, 'hard') 
    print(f'Initial bits: {bits};\nDemodulated bits: {demodulated_bits}')
    ### BER calculating
    ber = channel_funcs.ber_calc(bits, demodulated_bits)
    print(f'BER = {ber}')


if __name__ == '__main__':
    main()