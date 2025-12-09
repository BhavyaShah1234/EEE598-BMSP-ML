import numpy as np
import matplotlib.pyplot as plt
#import wfdb as wfdb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
import pyedflib
from sklearn.metrics import cohen_kappa_score
from scipy.fft import fft, ifft
from scipy import signal
from event_splitter import event_splitter   #my own function
from decompose_EEG import decompose_EEG
from add_temporal_context import add_temporal_context
import neurokit2 as nk
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from zero_cross import zero_cross
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from wonambi.detect.spindle import DetectSpindle
from wonambi.detect.spindle import detect_Moelle2011
from make_temporal_feature_names import make_temporal_feature_names
from sklearn.metrics import precision_recall_fscore_support

# @title
patient = 4001
print("How many patients?")
amount_of_patients = int(input())

stages = ['1', '2', '3', '4','W', 'R']# 1=1,2=2,3=3,4=4,W=5,R=6
colors =[]
EEG_Fpz_Cz_power = []
EMG_amplitude = []
EOG_rate   = []
EOG_AMP=[]
EOG_RATE_VAR = []
EEG_Fpz_Cz_RANGE = []
EEG_Fpz_oz_RANGE = []

GAMMA_pow = []
BETA_pow=[]
ALPHA_pow = []
THETA_pow = []
DELTA_pow = []

GAMMA_pow_oz = []
BETA_pow_oz=[]
ALPHA_pow_oz = []
THETA_pow_oz = []
DELTA_pow_oz = []

EMG_VAR = []
EMG_MAV = []
EMG_entropy = []

ZRC_EEG = []
ZRC_EMG = []
ZRC_EOG = []

SPIND_DENSE_FPZ_OZ = []
SPIND_DENSE_FPZ = []

patient_dat = []

opts = DetectSpindle()
opts.frequency = (12,15)
opts.duration = (0.3, 5)
opts.det_remez = {
    'freq': [12, 15],
    'order': 1001,
    'rolloff': 2.9,
    'window': 'hann',
    'dur':.3,
}

opts.moving_rms = {
    'step': .01,   # 200 ms RMS window
    'dur':.25,
}

opts.smooth = {
    'win': 'flat',   # smoothing window
    'dur':.05, #convolution window
}

opts.det_thresh = .8

for i in range(amount_of_patients):#511

	try:
		file ="SC"+str(patient)+"E-PSG.edf"
		hypno_file = "SC"+str(patient)+"E-Hypnogram.edf"#file name for our annotations labels
		path = "/home/humberto/Documents/GRAD Classes/EEE 598 BIOMED DSP/Final Project/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
		psg_file_path = path + "/" + file
		hypno_file_path = path + "/" + hypno_file


		#from the hypnogram file we have event stamps to look at and annotations
		event_times,duration,annotations = event_splitter(hypno_file_path)


		AVG_LEN_OF_EVENT = np.round(np.mean(duration)) #the event times are different lengths, so I decided to take the average window length

		signal_bio, sig_header, header = pyedflib.highlevel.read_edf(psg_file_path) #reading the edf file since wfdb cannot read these. At least the python lib can't
		signal_bio = np.array(signal_bio, dtype=object) #this signal an array of size 7, i.e., there are 7 channels


		patient +=1
		#Output from the above line
		#0) {'label': 'EEG Fpz-Cz', 'dimension': 'uV', 'sample_frequency': 100.0, 'physical_max': 192.0, 'physical_min': -192.0, 'digital_max': 2047, 'digital_min': -2048, 'prefilter': 'HP:0.5Hz LP:100Hz [enhanced cassette BW]', 'transducer': 'Ag-AgCl electrodes'}
		#1) {'label': 'EEG Pz-Oz', 'dimension': 'uV', 'sample_frequency': 100.0, 'physical_max': 196.0, 'physical_min': -197.0, 'digital_max': 2047, 'digital_min': -2048, 'prefilter': 'HP:0.5Hz LP:100Hz [enhanced cassette BW]', 'transducer': 'Ag-AgCl electrodes'}
		#2) {'label': 'EOG horizontal', 'dimension': 'uV', 'sample_frequency': 100.0, 'physical_max': 1009.0, 'physical_min': -1009.0, 'digital_max': 2047, 'digital_min': -2048, 'prefilter': 'HP:0.5Hz LP:100Hz [enhanced cassette BW]', 'transducer': 'Ag-AgCl electrodes'}
		#3) {'label': 'Resp oro-nasal', 'dimension': '', 'sample_frequency': 1.0, 'physical_max': 2047.0, 'physical_min': -2048.0, 'digital_max': 2047, 'digital_min': -2048, 'prefilter': 'HP:0.03Hz LP:0.9Hz', 'transducer': 'Oral-nasal thermistors'}
		#4) {'label': 'EMG submental', 'dimension': 'uV', 'sample_frequency': 1.0, 'physical_max': 5.0, 'physical_min': -5.0, 'digital_max': 2500, 'digital_min': -2500, 'prefilter': 'HP:16Hz Rectification LP:0.7Hz', 'transducer': 'Ag-AgCl electrodes'}
		#5) {'label': 'Temp rectal', 'dimension': 'DegC', 'sample_frequency': 1.0, 'physical_max': 40.0, 'physical_min': 34.0, 'digital_max': 2731, 'digital_min': -2849, 'prefilter': '', 'transducer': 'Rectal thermistor'}
		#6) {'label': 'Event marker', 'dimension': '', 'sample_frequency': 1.0, 'physical_max': 2048.0, 'physical_min': -2047.0, 'digital_max': 2048, 'digital_min': -2047, 'prefilter': 'Hold during 2 seconds', 'transducer': 'Marker button'}]


		#@title
		epoch_len      = 3000 #30 second window of data standard for sleep stages
		epoch_len_slow = 60   #this is for the signals sampled at 1 Hz
		win_length     = len(signal_bio[0])                             #number of points we want to look at
		filter_len     = 4096
		infraslow_cutoff = [.02,0.2]
		delta_cutoff     = [.5,4]
		theta_cutoff     = [4,7]
		sigma_cutoff     = [12,16]

		taps=40
		filter_type    = "bandpass"
		EEG_Fpz_Cz     = signal_bio[0]
		EEG_Pz_Oz      = signal_bio[1]
		EOG_horizontal = nk.eog_clean(signal_bio[2], sampling_rate=100, method='neurokit')# signal_bio[2]
		Resp_Oro_nasal = signal_bio[3]
		EMG_submental  = nk.emg_clean(signal_bio[4], sampling_rate=1, method='none')
		Temp_rectal    = signal_bio[5]
		Event_marker   = signal_bio[6]


		fs0 = 100                                        #sampling frequency of channel 0
		fs1 = 1   #sampling frequency for the waves sampled at 1 HZ,  Resp oro-nasal, EMG, Tem rectal, Event marker
		t0 = np.linspace(0, win_length/fs0, win_length)  #time vector
		t1 = np.linspace(0, win_length/fs1, win_length)  #time vector for the slower sampled signals

		# t_spindle_window = np.arange(len(EEG_Fpz_Cz)) / fs0#np.linspace(epoch_start,epoch_end,epoch_len) /fs0

		# detected_spind, spind_dict, spind_density = detect_Moelle2011(EEG_Fpz_Cz, fs0,t_spindle_window, opts)
		# print(len(detected_spind))

		# infra_slow_wave = decompose_EEG(EEG_Fpz_Cz, win_length, filter_len,fs0,filter_type,infraslow_cutoff,taps)
		# delta_wave      = decompose_EEG(EEG_Fpz_Cz, win_length,filter_len,fs0,filter_type,delta_cutoff,taps)
		theta_wave      = decompose_EEG(EEG_Fpz_Cz, win_length,filter_len,fs0,filter_type,theta_cutoff,taps)
		sigma_wave      = decompose_EEG(EEG_Fpz_Cz, win_length,filter_len,fs0,filter_type,sigma_cutoff,taps) 
		print(patient)
		for i in range(len(annotations)):
			if i == 0:
				epoch_start = int(0) #we will iterate through epochs, denoted by i
				epoch_end  = int((epoch_len))
				epoch_start_slow = int(0) #for the slow sampled signals at 1Hz
				epoch_end_slow = int(epoch_len_slow)
			else:
				epoch_start = int(event_times[i]-(epoch_len/2)) #we will iterate through epochs, denoted by i
				epoch_end  = int(event_times[i]+(epoch_len/2))

				epoch_start_slow = int(event_times[i]-(epoch_len_slow/2)) #for the slow sampled signals at 1Hz
				epoch_end_slow  = int(event_times[i]+(epoch_len_slow/2))
			if stages[0] in annotations[i]:
				annotations[i] = 1 #stage 1 gets value 1
				# colors.append('red')
				colors.append(0)
			elif stages[1] in annotations[i]:
				annotations[i] = 2 #stage 2 gets value 2
				# colors.append('green')
				colors.append(1)
				
			elif stages[2]  in annotations[i] :
				annotations[i] = 3 #stage 3 gets value 3
				# colors.append('blue')
				colors.append(2)

			elif stages[3] in annotations[i]:
				annotations[i] = 4 #stage 4 gets value 4
				# colors.append('orange')
				colors.append(3)

			elif stages[4] in annotations[i]:
				annotations[i] = 5 #stage W gets value 5
				# colors.append('purple')
				colors.append(4)

			else:
				annotations[i] = 6 #stage R gets value 6
				# colors.append('yellow')
				colors.append(5)


			EEG_Fpz_Cz_RANGE.append(np.max(EEG_Fpz_Cz[epoch_start:epoch_end]) - np.min(EEG_Fpz_Cz[epoch_start:epoch_end]))	
			EEG_Fpz_oz_RANGE.append(np.max(EEG_Pz_Oz[epoch_start:epoch_end]) - np.min(EEG_Fpz_Cz[epoch_start:epoch_end]))
			eeg_power_df_fpz = nk.eeg_power(EEG_Fpz_Cz[epoch_start:epoch_end], fs0, frequency_band=['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'])
			eeg_power_df_oz = nk.eeg_power(EEG_Pz_Oz[epoch_start:epoch_end], fs0, frequency_band=['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'])

			t_spindle_window = np.arange(len(EEG_Fpz_Cz[epoch_start:epoch_end])) / fs0#np.linspace(epoch_start,epoch_end,epoch_len) /fs0

			detected_spind, spind_dict, spind_density = detect_Moelle2011(sigma_wave[epoch_start:epoch_end], fs0,t_spindle_window, opts)
			SPIND_DENSE_FPZ.append(spind_density)
			# print(spind_density)

			# print(eeg_power_df)
			GAMMA_pow.append([eeg_power_df_fpz.iloc[0,1]])
			BETA_pow.append([eeg_power_df_fpz.iloc[0,2]])
			ALPHA_pow.append([eeg_power_df_fpz.iloc[0,3]])
			THETA_pow.append([eeg_power_df_fpz.iloc[0,4]])
			DELTA_pow.append([eeg_power_df_fpz.iloc[0,5]])

			GAMMA_pow_oz.append([eeg_power_df_oz.iloc[0,1]])
			BETA_pow_oz.append([eeg_power_df_oz.iloc[0,2]])
			ALPHA_pow_oz.append([eeg_power_df_oz.iloc[0,3]])
			THETA_pow_oz.append([eeg_power_df_oz.iloc[0,4]])
			DELTA_pow_oz.append([eeg_power_df_oz.iloc[0,5]])

			#zero cross rate detectors for EEG,EMG,EOG
			zrc_eeg = zero_cross(EEG_Fpz_Cz[epoch_start:epoch_end])
			ZRC_EEG.append(zrc_eeg)
			zrc_emg = zero_cross(EMG_submental[epoch_start:epoch_end])
			ZRC_EMG.append(zrc_emg)
			zrc_eog = zero_cross(EOG_horizontal[epoch_start:epoch_end])
			ZRC_EOG.append(zrc_eog)
			
			emg_amp = nk.emg_amplitude(EMG_submental[epoch_start_slow:epoch_end_slow])
			emg_var = np.var(EMG_submental[epoch_start_slow:epoch_end_slow])
			emg_mav = np.max(np.abs(EMG_submental[epoch_start_slow:epoch_end_slow]))
			peaks = nk.eog_findpeaks(EOG_horizontal[epoch_start:epoch_end], sampling_rate=fs0)
			eog_rate = len(peaks) / 30.0  # blinks per second
			eog_amp = np.mean(np.abs(EOG_horizontal[epoch_start:epoch_end]))

			# eog_rate=nk.eog_rate( nk.eog_findpeaks(EOG_horizontal[epoch_start:epoch_end]) ,fs0, epoch_len)
			EMG_amplitude.append(np.sqrt(np.mean(np.square(emg_amp))))
			EMG_VAR.append(emg_var)
			EMG_MAV.append(emg_mav)

			EOG_AMP.append(eog_amp)
			EOG_RATE_VAR.append(np.var(eog_rate))
			EOG_rate.append(np.mean(eog_rate))

			patient_dat.append(patient)
			#print(f"{max( [GAMMA_pow[i], BETA_pow[i], ALPHA_pow[i], THETA_pow[i] ])}")
			#print(f"EEG Power:{DELTA_pow[i]} | SLEEP stage: {annotations[i]} | EOG Rate: {EOG_rate[i]}")
	except FileNotFoundError:
		patient = patient+1
		continue




# colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow']


#Features we currently have: power for 4 EEG frequency bands, EMG Amplitude, EOG rate


#Plotting histogram to show distribution of sleep stages
plt.figure()
labels, counts = np.unique(colors, return_counts=True)
plt.bar(labels,counts, align='center')
plt.xlabel("Sleep Stages")
plt.ylabel("Frequency")
plt.gca().set_xticks(labels)
plt.title("Distribution of Sleep Stages n1:0, n2:1, n3:2, n4:3, W:4,REM:5 ")
plt.show()
############################################################
#PCA Analysis
GAMMA_pow =np.squeeze(np.array(GAMMA_pow))
DELTA_pow = np.squeeze(np.array(DELTA_pow))
BETA_pow  = np.squeeze(np.array(BETA_pow))
ALPHA_pow = np.squeeze(np.array(ALPHA_pow))
THETA_pow = np.squeeze(np.array(THETA_pow))
#Power Ratios
GAMMA_DELTA = np.log10(GAMMA_pow / DELTA_pow + 1e-8)
GAMMA_BETA  =  np.log10(GAMMA_pow / BETA_pow + 1e-8)
GAMMA_ALPHA =  np.log10(GAMMA_pow/ ALPHA_pow + 1e-8)
GAMMA_THETA = np.log10(GAMMA_pow/ THETA_pow + 1e-8)
DELTA_BETA = np.log10(DELTA_pow/BETA_pow + 1e-8)
DELTA_ALPHA = np.log10(DELTA_pow/ALPHA_pow + 1e-8)
DELTA_THETA = np.log10(DELTA_pow/THETA_pow + 1e-8)
BETA_ALPHA = np.log10(BETA_pow/ALPHA_pow + 1e-8)
BETA_THETA = np.log10(BETA_pow/THETA_pow + 1e-8)
ALPHA_THETA = np.log10(ALPHA_pow/THETA_pow + 1e-8)

GAMMA_pow_oz =np.squeeze(np.array(GAMMA_pow_oz))
DELTA_pow_oz = np.squeeze(np.array(DELTA_pow_oz))
BETA_pow_oz  = np.squeeze(np.array(BETA_pow_oz))
ALPHA_pow_oz = np.squeeze(np.array(ALPHA_pow_oz))
THETA_pow_oz = np.squeeze(np.array(THETA_pow_oz))

GAMMA_DELTA_oz = np.log10(GAMMA_pow_oz / DELTA_pow_oz + 1e-8)
GAMMA_BETA_oz  =  np.log10(GAMMA_pow_oz / BETA_pow_oz + 1e-8)
GAMMA_ALPHA_oz =  np.log10(GAMMA_pow_oz/ ALPHA_pow_oz + 1e-8)
GAMMA_THETA_oz = np.log10(GAMMA_pow_oz/ THETA_pow_oz + 1e-8)
DELTA_BETA_oz = np.log10(DELTA_pow_oz/BETA_pow_oz + 1e-8)
DELTA_ALPHA_oz = np.log10(DELTA_pow_oz/ALPHA_pow_oz + 1e-8)
DELTA_THETA_oz = np.log10(DELTA_pow_oz/THETA_pow_oz + 1e-8)
BETA_ALPHA_oz = np.log10(BETA_pow_oz/ALPHA_pow_oz + 1e-8)
BETA_THETA_oz = np.log10(BETA_pow_oz/THETA_pow_oz + 1e-8)
ALPHA_THETA_oz = np.log10(ALPHA_pow_oz/THETA_pow_oz + 1e-8)

EOG_rate  = np.array(EOG_rate)
EMG_amplitude = np.array(EMG_amplitude)

feature_list = [EEG_Fpz_Cz_RANGE,EEG_Fpz_oz_RANGE,SPIND_DENSE_FPZ,EOG_RATE_VAR,ZRC_EOG,ZRC_EMG,ZRC_EEG,EOG_AMP,EMG_VAR,EMG_MAV,ALPHA_THETA,GAMMA_DELTA, GAMMA_BETA, GAMMA_ALPHA, GAMMA_THETA, DELTA_BETA, DELTA_ALPHA, DELTA_THETA, BETA_ALPHA, BETA_THETA,ALPHA_THETA_oz,GAMMA_DELTA_oz, GAMMA_BETA_oz, GAMMA_ALPHA_oz, GAMMA_THETA_oz, DELTA_BETA_oz, DELTA_ALPHA_oz, DELTA_THETA_oz, BETA_ALPHA_oz, BETA_THETA_oz, EOG_rate,EMG_amplitude]

# feature_list = [EEG_Fpz_Cz_RANGE,EEG_Fpz_oz_RANGE,SPIND_DENSE_FPZ,EOG_RATE_VAR,ZRC_EOG,ZRC_EMG,ZRC_EEG,EOG_AMP,EMG_VAR,EMG_MAV,ALPHA_THETA,GAMMA_DELTA, GAMMA_BETA, GAMMA_ALPHA, GAMMA_THETA, DELTA_BETA, DELTA_ALPHA, DELTA_THETA, BETA_ALPHA, BETA_THETA,ALPHA_THETA_oz,GAMMA_DELTA_oz, GAMMA_BETA_oz, GAMMA_ALPHA_oz, GAMMA_THETA_oz, DELTA_BETA_oz, DELTA_ALPHA_oz, DELTA_THETA_oz, BETA_ALPHA_oz, BETA_THETA_oz,GAMMA_pow ,DELTA_pow,BETA_pow,ALPHA_pow,THETA_pow, GAMMA_pow_oz, DELTA_pow_oz, BETA_pow_oz, ALPHA_pow_oz, THETA_pow_oz, EOG_rate,EMG_amplitude]
base_features = ["EEG_Fpz_Cz_RANGE", "EEG_Fpz_oz_RANGE", "SPIND_DENSE_FPZ","EOG_RATE_VAR", "ZRC_EOG", "ZRC_EMG", "ZRC_EEG","EOG_AMP", "EMG_VAR", "EMG_MAV", "ALPHA_THETA", "GAMMA_DELTA", "GAMMA_BETA", "GAMMA_ALPHA","GAMMA_THETA","DELTA_BETA","DELTA_ALPHA","DELTA_THETA","BETA_ALPHA","BETA_THETA","ALPHA_THETA_oz","GAMMA_DELTA_oz","GAMMA_BETA_oz","GAMMA_ALPHA_oz","GAMMA_THETA_oz","DELTA_BETA_oz","DELTA_ALPHA_oz","DELTA_THETA_oz","BETA_ALPHA_oz","BETA_THETA_oz","GAMMA_pow","DELTA_pow","BETA_pow","ALPHA_pow","THETA_pow","GAMMA_pow_oz","DELTA_pow_oz","BETA_pow_oz","ALPHA_pow_oz","THETA_pow_oz","EOG_rate","EMG_amplitude"]
feature_name_temp_window = make_temporal_feature_names(base_features, window=1)
print(len(feature_name_temp_window))
features = np.stack(feature_list, axis=1)


concated_features = add_temporal_context(features)
y_time_context = colors[1:-1] 


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
catted_scale =  scaler.fit_transform(concated_features)

pca = PCA(n_components=20, svd_solver='full') #I want to reduce my data down into the 2 most important variables
# pca_psg_data = pca.fit_transform(catted_scale)


# model_tsne = TSNE(n_components = 3, random_state = 0)
# tsne_psg_data = model_tsne.fit_transform(catted_scale)

# # print(f"PCA values : {pca.explained_variance_ratio_}")
# # # print(pca_HRV)
# # print(pca_psg_data)

# plt.figure()
# plt.scatter(tsne_psg_data[:,0],tsne_psg_data[:,1], c = colors)
# plt.xlabel('TSNE1')
# plt.ylabel('TSNE2')
# plt.title('TSNE2 vs TSNE1')

# plt.figure()
# plt.scatter(pca_psg_data[:,0],pca_psg_data[:,1], c = colors)
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('PCA2 vs PCA1')


# plt.show()


#########################################################################################
#Machine Learning:
#1)Logisitic Regression
X=features_scaled
X_cat = catted_scale
pca_psg_data = pca.fit_transform(X_cat)
print(f"PCA values : {pca.explained_variance_ratio_}")
pca_components= pca.explained_variance_ratio_


plt.figure()
plt.stem(np.arange(1,len(pca_components)+1,1), pca_components)
plt.xlabel('component for PCA')
plt.ylabel('Percent Contribution')
plt.title('Scree Plot')
plt.show()

plt.figure()
plt.scatter(pca_psg_data[:,0],pca_psg_data[:,1], c = y_time_context)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA2 vs PCA1')


plt.show()

y= np.array(colors)
groups = patient_dat[1:-1]#np.array(patient_dat)
# 
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

# X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(pca_psg_data, y_time_context, test_size = 0.2, random_state=42) #this test split doesn't apply SMOTE
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_cat, y_time_context, test_size = 0.3, random_state=42) #this test split doesn't apply SMOTE

train_idx, test_idx = next(gss.split(X_cat, y_time_context, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

sm=SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# model = LogisticRegression()
# logreg_psg = model.fit(X_train,y_train)

# y_pred = logreg_psg.predict(X_test)

# print(classification_report(y_test,y_pred))


#**************************************************
#2)K-Nearest Neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=6)
trained_neigh = neigh.fit(X_pca_train, y_pca_train)
KNN_predict = trained_neigh.predict(X_pca_test)
print(KNN_predict)
print(classification_report(y_pca_test,KNN_predict))
# # print(f"Cross validation f-1 scores for kfold=5 : {cvs(nay_bor, X_train, y_train, cv=5, scoring='f1_macro')}" ) #this does k-fold validation by default

#**************************************************

#3)SVM with RBF Kernel
#Rules:The RBF helps us find the effect of any two points in infinite dimensions
#Gamma parameter:this controls the overfitting/underfit aspect of our ML model
#Big gamma gives overfitting and vice-versa
#The RBF can map our data to higher dimensions in order to separate it
#
#Modified example from: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


svm_rbf = SVC(kernel='rbf',C=1, class_weight='balanced',gamma='scale')
support_vec_result = svm_rbf.fit(X_pca_train, y_pca_train)
svm_predict = support_vec_result.predict(X_pca_test)
print(classification_report(y_pca_test,svm_predict))
# print(svm_predict)
#print(f"SVM with RBF Kernel Cross validation f-1 scores for kfold=5 : {cvs(support_vec_result, X_train_res, y_train_res, cv=5, scoring='f1_macro')}" ) #this does k-fold validation by default

#**************************************************
#4)Random Forest


from sklearn.ensemble import RandomForestClassifier

rf_class = RandomForestClassifier( n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
rf_fit = rf_class.fit(X_train_res, y_train_res)
rf_predict = rf_fit.predict(X_test)
print(classification_report(y_test,rf_predict))

#**************************************************
#4)XGboost
import xgboost as xgb

model = xgb.XGBClassifier(
max_depth=3,
learning_rate=0.05,
n_estimators=300,
subsample=0.8,
colsample_bytree=0.6,
reg_lambda=1.5,
reg_alpha=0.5
)

# xgb_model=model.fit(X_pca_train, y_pca_train)
xgb_model=model.fit(X_pca_train, y_pca_train)

# Predict
xgb_predict = xgb_model.predict(X_pca_test)

print(classification_report(y_pca_test,xgb_predict))

#******************************************************************************************************************
#Plotting the confusion Matrix for the results of 
#KNN Confusion Matrix
cm_KNN = confusion_matrix(y_pca_test, KNN_predict)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_KNN,  display_labels=['Class 0', 'Class 1','Class 2','Class 3', 'Wake', 'REM'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for K-Nearest Neighbor')
plt.show()






#******************************************************************************************************************

#SVM with RBF 
cm_sv_rbf = confusion_matrix(y_pca_test, svm_predict)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_sv_rbf, display_labels=['Class 0', 'Class 1','Class 2','Class 3', 'Wake', 'REM'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for SVM with RBF Kernel')
plt.show()

sens_svm = []
speci_svm = []
for c in [0,1,2,3,4,5]:
	 prec,recall,_,_ = precision_recall_fscore_support(np.array(y_pca_test)==c,
                                                      np.array(xgb_predict)==c,
                                                      pos_label=True,average=None)
	 sens_svm.append(recall[0])
	 speci_svm.append(recall[1])
print(speci_svm)

svm_rbf_cohen_kappa = cohen_kappa_score(y_pca_test, svm_predict)
print(f"Cohen-Kappa Score for SVM with RBF Kernel Pipeline : {svm_rbf_cohen_kappa}")

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
f1_svm_class0 = f1_score(y_pca_test, svm_predict, average=None)


print(f"F1-score for class 0: {f1_svm_class0}")

#******************************************************************************************************************
#Plotting the confusion Matrix for the results of 
#RF Confusion Matrix
cm_RF = confusion_matrix(y_test, rf_predict)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_RF,  display_labels=['Class 0', 'Class 1','Class 2','Class 3', 'Wake', 'REM'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.show()


#******************************************************************************************************************
#XGboost
cm_xgboost = confusion_matrix(y_pca_test, xgb_predict)
# tn,fp,fn,tp = confusion_matrix(y_pca_test, xgb_predict).ravel()
sens_xgb = []
speci_xgb = []
for c in [0,1,2,3,4,5]:
	 prec,recall,_,_ = precision_recall_fscore_support(np.array(y_pca_test)==c,
                                                      np.array(xgb_predict)==c,
                                                      pos_label=True,average=None)
	 sens_xgb.append(recall[0])
	 speci_xgb.append(recall[1])
print(speci_xgb)

# xgb_specificity = tn/(tn+fp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgboost, display_labels=['Class 0', 'Class 1','Class 2','Class 3', 'Wake', 'REM'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for XGBoost')
plt.show()


xgb_cohen_kappa = cohen_kappa_score(y_pca_test, xgb_predict)
print(f"Cohen-Kappa Score for XGBoost Pipeline : {xgb_cohen_kappa}")


f1_xgb_class0 = f1_score(y_pca_test, xgb_predict,average=None)

print(f"F1-score for class 0-5: {f1_xgb_class0}")

#******************************************************************************************************************
#Getting and Plotting Shapely values
import shap

# exp = shap.Explanation(xgb_predict, data = X_test, feature_names=feature_name_temp_window)
# shap.summary_plot(exp[0],X_train)

window_offsets = [-1, 0, 1]
feature_names = [
    f"{name}_t{offset:+d}"
    for name in base_features
    for offset in window_offsets
]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_pca_train)

class_idx = int(np.bincount(xgb_predict).argmax())

if isinstance(shap_values, list):
    shap_values_for_summary = np.mean(shap_values, axis=0)
else:
    shap_values_for_summary = shap_values

print(X_pca_train.shape)
print(np.array(shap_values).shape)

# shap_mean = np.mean(np.abs(shap_values), axis=2) 

# shap.summary_plot(
#     shap_mean,
#     X_pca_train,
#     feature_names=feature_names
# )

for cls in range(shap_values.shape[2]):
    print(f"\n=== SHAP Summary Plot for Class {cls} ===")
    
    shap.summary_plot(
        shap_values[:, :, cls],   # class-specific SHAP matrix
        X_pca_train,
        feature_names=feature_names,
        show=False)
    plt.title(f"SHAP Summary Plot – Class {cls}", fontsize=16)
    plt.show()


