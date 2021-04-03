import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from vitals import predict_vitals,hear_rate,load_model

st.title('Calculate heart rate from video')

@st.cache
def loadmodel():
	model=load_model()
	return model

model=load_model()

uploaded_file = st.file_uploader("Upload Files",accept_multiple_files=False,type=['mp4'])#upload file
if uploaded_file is not None:#check if file is present
	with st.spinner('Wait for it...'):
		tfile = tempfile.NamedTemporaryFile(delete=False) 
		tfile.write(uploaded_file.read())
		pulse,resp,fs=predict_vitals(tfile.name,model)
	st.success('Done!')
	d=st.slider('Select threshold', min_value=5 , max_value=15 , value=10 , step=2 )
	st.write(d)
	peaks1, _ = find_peaks(pulse,distance=d)
	hr=hear_rate(peaks1,fs)
	print("heart rate is",hr)
	st.header("heart rate is {}".format(hr.round()))

	fig,ax=plt.subplots(2,1)
	ax[0].plot(pulse) 
	ax[0].plot(peaks1, pulse[peaks1], "x")
	ax[0].set_title('Pulse Prediction')
	ax[1].plot(resp)
	ax[1].set_title('Respiration Prediction')
	fig.tight_layout()
	st.write(fig)