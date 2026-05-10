import numpy as np
import scipy.fft as fft

w = np.exp(1j * np.pi * np.arange(16) * 0.5)
padded = np.pad(w, (0, 16*8 - 16))
F_ifft = fft.ifft(padded)
k = np.argmax(np.abs(F_ifft))
print("ifft max at k=", k, "which is", k / (16*8))

F_fft = fft.fft(padded)
k_fft = np.argmax(np.abs(F_fft))
print("fft max at k=", k_fft, "which is", k_fft / (16*8))
