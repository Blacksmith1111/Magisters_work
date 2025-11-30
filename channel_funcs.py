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

def quantize(signal, resolution: int, S_max: float):
    ### DAC full-scale range
    DAC_rng = np.arange(-2**resolution / 2, 2**resolution / 2, 1)
    ### Scaling factor
    G = DAC_rng[-1] / np.abs(S_max)
    signal_normalized = signal * G
    I, Q = signal_normalized.real, signal_normalized.imag
    DAC_indicies_I = np.argmin(np.abs(I[:, None] - DAC_rng[None, :]), axis = 1)
    DAC_indicies_Q = np.argmin(np.abs(Q[:, None] - DAC_rng[None, :]), axis = 1)
    I_quantized = DAC_rng[DAC_indicies_I]
    Q_quantized = DAC_rng[DAC_indicies_Q]
    print(I_quantized / G + 1j * Q_quantized / G, '\n', signal)
    return I_quantized + 1j * Q_quantized

def main():
    ### Parameters
    bits_num = 12
    mod_order = 64
    Fs = 10e3           
    sps = 10             
    f_sym = Fs / sps     
    Ts = 1 / f_sym       
    rolloff = 0.25
    filter_span = 8      
    S_max_dict = {'64QAM': np.sqrt(2*7**2), '32QAM': np.sqrt(3**2 + 5**2)}


    ### PBRS
    bits  = np.random.randint(0, 2, bits_num)

    ### QAM modulation
    qam = mod.QAMModem(mod_order)
    modulated_signal = qam.modulate(bits)
    print(f'Modulated signal ({mod_order} QAM): {modulated_signal}')

    ### Upsampling
    upsampled_signal = upsample(modulated_signal, sps)
    print(f'Signal after upsampling: {upsampled_signal}')

    ### Pulse shaping (TX)
    shaped_signal = pulse_shaping(
        upsampled_signal, rolloff=rolloff, filter_span=filter_span, 
        sps=sps, Fs=Fs, Ts=Ts, normaliztion='L2'
    )
    time_delay_tx = filter_span * sps 

    ### Quantizing
    s = quantize(modulated_signal, resolution = 6, S_max = S_max_dict['64QAM'])


    plt.figure()
    plt.stem(upsampled_signal.real, label='Upsampled Signal')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(shaped_signal[time_delay_tx:-time_delay_tx].real, label='Shaped Signal')
    plt.grid()
    plt.legend()
    plt.show()

    ### Signal recovery (matched filter)
    recovered_signal = pulse_shaping(
        shaped_signal, rolloff=rolloff, filter_span=filter_span, 
        sps=sps, Fs=Fs, Ts=Ts, normaliztion='L2'
    )
    time_delay_rx = 2 * time_delay_tx 
    recovered_signal = recovered_signal[time_delay_rx : time_delay_rx + len(upsampled_signal)]

    plt.figure()
    plt.stem(recovered_signal.real, label='Recovered Signal')
    plt.grid()
    plt.legend()
    plt.show()

    ### Downsampling
    downsampled_signal = downsample(recovered_signal, sps)
    print(f'Signal after downsampling: {downsampled_signal}')
    ### Demapping
    demodulated_bits = qam.demodulate(downsampled_signal, 'hard') 
    print(f'Initial bits: {bits};\nDemodulated bits: {demodulated_bits}')
    ### BER calculating
    ber = ber_calc(bits, demodulated_bits)
    print(f'BER = {ber}')


if __name__ == '__main__':
    main()