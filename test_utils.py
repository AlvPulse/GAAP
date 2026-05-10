import numpy as np
from beamformer.utils import oversampled_fft, oversampled_ifft
from beamformer.synthesis import synthesize_pencil

w = synthesize_pencil(16, 0.5)
u, F = oversampled_fft(w, 8)
print("max at u=", u[np.argmax(np.abs(F))])
w2 = oversampled_ifft(F, 16)
print("w == w2?", np.allclose(w, w2))
