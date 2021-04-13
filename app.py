import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyampd.ampd import find_peaks_adaptive
import numpy as np
from vitals import predict_vitals,hear_rate,load_model

st.title('Calculate heart rate from video')

@st.cache
def loadmodel():
	model=load_model()
	return model

model=load_model()

#@st.cache
uploaded_file = st.file_uploader("Upload Files",accept_multiple_files=False,type=['mp4','avi'])#upload file
if uploaded_file is not None:#check if file is present
	with st.spinner('Wait for it...'):
		tfile = tempfile.NamedTemporaryFile(delete=False) 
		tfile.write(uploaded_file.read())
		pulse,resp,fs,sample_img=predict_vitals(tfile.name,model)
	st.success('Done!')

	st.write('frame rate',np.round(fs,3))
	# d=st.slider('Select threshold', min_value=5 , max_value=15 , value=10 , step=2 )
	# st.write(d)
	# peaks1, _ = find_peaks(pulse,distance=d)
	np.savetxt('pulse.txt',pulse)
	hr_adaptive=[]
	for i in range(10,100,5):
		peaks=find_peaks_adaptive(pulse, window=i)
		hr_adaptive.append(hear_rate(peaks,fs))
	#print(hrs)
	hrAmed=np.median(hr_adaptive)
	hrAstd=np.std(hr_adaptive)
	hrAmean=np.mean(hr_adaptive)
	
	# hr=hear_rate(peaks1,fs)
	# print(hr)
	st.header("heart rate is {}, {}, {}".format(hrAmean.round(),hrAmed.round(),hrAstd.round()))

	fig,ax=plt.subplots(2,1)
	ax[0].plot(pulse) 
	# ax[0].plot(peaks1, pulse[peaks1],label='threshold',marker= "x")
	ax[0].plot(peaks, pulse[peaks],label='adaptive',marker= "*")
	ax[0].set_title('Pulse Prediction')
	ax[0].legend()

	ax[1].plot(resp)
	ax[1].set_title('Respiration Prediction')
	fig.tight_layout()
	st.write(fig)

	st.image(sample_img)