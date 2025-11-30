import numpy as np
import commpy.modulation as mod
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter


def constellation_plot(modulated_signal: complex) -> None:
    fig, ax = plt.subplots()
    x = modulated_signal.real
    y = modulated_signal.imag

    ax.scatter(x, y)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

    ax.grid(True)
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

    demodulated_bits = qam.demodulate(downsampled_signal, 'hard') 
    print(f'Demodulated bits: {demodulated_bits}')


if __name__ == '__main__':
    main()