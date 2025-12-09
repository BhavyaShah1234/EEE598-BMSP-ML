#Function to decompose the EEG signal into its appropriate signal
from scipy.fft import fft, ifft
from scipy import signal
import numpy as np


def decompose_EEG(input_signal, win_length, filter_len, fs, filt_type, cutoff, taps):
# #will implement a Butterworth Bandpass filter for, later we might
# #want another type of filter.
# ##################################
# #Lines below will get the "Infraslow oscillations" from 0.02-0.2HZ
	sos_infra = signal.butter(N=taps, Wn=cutoff, btype=filt_type, output='sos', fs=fs )
	w,h = signal.freqz_sos(sos_infra, worN=filter_len, fs = fs)#may want to include the worN option

	#Uncomment this to look at the frequency response for our "Infraslow Oscillation"
	#signal generation
	if ((win_length % filter_len) != 0):
		target_len = int(np.round(win_length/filter_len)) * filter_len
		# print(target_len)
		# print(len(input_signal))
		pad_amount = abs(target_len - win_length)
		# print(pad_amount)
		input_signal = np.pad(input_signal, pad_width=pad_amount, mode='constant', constant_values=0)

	infra_slow_wave = np.zeros(len(input_signal))
	fft_window_start= 0
	fft_window_end  = filter_len
	

	for i in range(int(len(input_signal)/filter_len)):
		EEG_Filtered_frequency_domain = np.fft.fft(h) * np.fft.fft((input_signal[fft_window_start:fft_window_end]))
		infra_slow_wave[fft_window_start:fft_window_end] = np.fft.ifft(EEG_Filtered_frequency_domain) #filtered out EEG
		fft_window_start = fft_window_start + filter_len
		fft_window_end   = fft_window_end + filter_len
		

	return infra_slow_wave #returns the filtered signal out
