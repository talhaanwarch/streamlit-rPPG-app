# stream-app
 
This app need to be run on a GPU system.
For this purpose, we will use google colab.
Steps.
Clone this repo on colab  
`!git clone https://github.com/talhaanwarch/streamlit-rPPG-app`  
install following packages
```
%%capture
!pip install colab-everything
!pip install streamlit==0.79.0
!pip install opencv-python==4.5.1.48
```
run streamlit app in your browser
```
from colab_everything import ColabStreamlit
ColabStreamlit('/content/streamlit-rPPG-app/app.py') 
```
