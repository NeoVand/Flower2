#### 		Read from Enobio EEG Data		####

 # The stream_name must be the same as in NIC/COREGUI.
 # The streams vector gathers all the streams available.
 # If the NIC stream is inside this vector, the element index is saved in the index variable.
 # The stream inlet attempts to connect to the NIC stream.
 # If the stream has not been found within the available streams, the scripts raises an error and stops.
 # If not, the script starts retrieving data from the NIC stream.

from pylsl import StreamInlet, resolve_stream

stream_name = 'NIC'
streams = resolve_stream('type', 'EEG')

try:
	for i in range (len(streams)):

		if (streams[i].name() == stream_name):
			index = i
			print ("NIC stream available")

	print ("Connecting to NIC stream... \n")
	inlet = StreamInlet(streams[index])   

except NameError:
	print ("Error: NIC stream not available\n\n\n")

while True:
    sample, timestamp = inlet.pull_sample()
    print("Timestamp: \t %0.5f\n Sample: \n %s\n\n" %(timestamp,   sample))