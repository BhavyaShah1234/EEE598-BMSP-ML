#Splits a hypnogram file into:
#event time, duration of event, and annotation
import pyedflib
import numpy as np
import matplotlib.pyplot as plt

def event_splitter(hypno_file_path):
	sleep_annotations= pyedflib.EdfReader(hypno_file_path) #read the annnotations
	headers_hypno = sleep_annotations.getSignalHeaders()
	sleep_annotations_list = sleep_annotations.readAnnotations()

	#making zero filled arrays of the length of the annotations
	event_times = np.zeros(len(sleep_annotations_list[0])) 
	duration    = np.zeros(len(sleep_annotations_list[1])) 
	annotations = []#np.zeros(len(sleep_annotations_list[2]), dtype='str') 
	#The Hypnogram EDF is organized as follows:
	#Time series point for the even | Duration | Event 
	# print(len(sleep_annotations_list))

	for i in range(len(sleep_annotations_list[0])):	
		event_times[i] = sleep_annotations_list[0][i]
		duration[i]    = sleep_annotations_list[1][i]
		annotations.append(sleep_annotations_list[2][i])
		
	return event_times, duration, annotations   #returns a np.array,np.array,and list