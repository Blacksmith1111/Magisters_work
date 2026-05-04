import numpy as np
from channel_funcs import constellation_plot

class Modulator32QAM:
    def __init__(self):
        self.mapping_table = {
            (0,0,0,0,0): -3+5j, (0,0,0,0,1): -1+5j, (0,0,0,1,1):  1+5j, (0,0,0,1,0):  3+5j,
            (0,0,1,0,0): -5+3j, (0,0,1,0,1): -3+3j, (0,0,1,1,1): -1+3j, (0,0,1,1,0):  1+3j, (0,1,1,1,0):  3+3j, (0,1,1,0,0):  5+3j,
            (0,1,0,0,0): -5+1j, (0,1,0,0,1): -3+1j, (0,1,0,1,1): -1+1j, (0,1,0,1,0):  1+1j, (0,1,1,1,1):  3+1j, (0,1,1,0,1):  5+1j,
            (1,1,0,0,0): -5-1j, (1,1,0,0,1): -3-1j, (1,1,0,1,1): -1-1j, (1,1,0,1,0):  1-1j, (1,1,1,1,1):  3-1j, (1,1,1,0,1):  5-1j,
            (1,0,1,0,0): -5-3j, (1,0,1,0,1): -3-3j, (1,0,1,1,1): -1-3j, (1,0,1,1,0):  1-3j, (1,1,1,1,0):  3-3j, (1,1,1,0,0):  5-3j,
            (1,0,0,0,0): -3-5j, (1,0,0,0,1): -1-5j, (1,0,0,1,1):  1-5j, (1,0,0,1,0):  3-5j
        }

        self.bits_tuples = list(self.mapping_table.keys())
        self.constellation = np.array(list(self.mapping_table.values()))
        
        self.avg_power = np.mean(np.abs(self.constellation)**2)

    def modulate(self, bits):

        reshaped_bits = np.reshape(bits, (-1, 5))
        symbols = np.zeros(len(reshaped_bits), dtype=complex)

        for i, bit_group in enumerate(reshaped_bits):
            symbols[i] = self.mapping_table[tuple(bit_group)]

        return symbols

    def demodulate(self, symbols):
        symbols_complex = np.array(symbols)
        points = self.constellation

        distances = np.abs(symbols_complex[:, np.newaxis] - points[np.newaxis, :])
        min_indices = np.argmin(distances, axis=1)

        demodulated_bits = np.array([self.bits_tuples[idx] for idx in min_indices]).flatten()

        return demodulated_bits


if __name__ == "__main__":
    qam32 = Modulator32QAM()

    n_bits = 100000
    tx_bits = np.random.randint(0, 2, n_bits)

    tx_symbols = qam32.modulate(tx_bits)
    constellation_plot(tx_symbols, 32, title = 'Titile')
    SNR_dB = 15
    SNR_linear = 10 ** (SNR_dB / 10)
    N0 = qam32.avg_power / SNR_linear
    
    noise = np.sqrt(N0 / 2) * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
    rx_symbols = tx_symbols + noise
    constellation_plot(rx_symbols, 32, title = 'Titile')
    rx_bits = qam32.demodulate(rx_symbols)

    errors = np.sum(tx_bits != rx_bits)
    ber = errors / n_bits
    
    print(f"Передано бит: {n_bits}")
    print(f"Количество ошибок: {errors}")
    print(f"BER (Bit Error Rate): {ber:.5f}")