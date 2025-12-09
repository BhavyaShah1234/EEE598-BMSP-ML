from scipy.fft import fft, ifft
from scipy import signal
import numpy as np

def zero_cross(x):
	sign_changes = np.diff(np.sign(x))
	num_zero_crossings = np.count_nonzero(sign_changes)
	zcr = num_zero_crossings / (len(x) - 1)
	return zcr