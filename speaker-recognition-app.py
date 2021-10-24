import streamlit as st
import joblib
st.title("speaker classifier app")
import streamlit as st
import os
import IPython
from IPython.display import Audio
from scipy.io.wavfile import read,write
header =st.container()
dataset=st.container()
model_training=st.container()
extractwavFeatures=st.container()
csvFileName=st.container()
x_data=st.container()
printDigit=st.container()
y_data=st.container()
preprocessData=st.container()
getSpeaker=st.container()
prediction=st.container()
report=st.container()

@st.cache
def get_data(filename):
    data=pd.read_csv(filename,index_col=False,skiprows=[1])
    #testdata=pd.read_csv(csvFileName,index_col=False)
    return data
import pydub
from pydub import AudioSegment
from pathlib import Path
import librosa

with header:
    
    
    #datapath1 = "C:\\Users\\renis\\audio\\neetha\\"
    datapath1=  "neethus-123/speaker-identification/tree/main/soundtest/testv"
    datapath2= "neethus-123/speaker-identification/tree/main/soundtest/testv"
#def identify():
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader("Choose a wav file that you extracted from the work site")
    uploaded_file1 = st.file_uploader("Select")
    
    if uploaded_file1 is not None:
            
            
            audio_bytes = uploaded_file1.read()
            st.audio(audio_bytes, format='audio/wav')
            sound = AudioSegment.from_wav(datapath1+uploaded_file1.name)
            sound.export(datapath1+uploaded_file1.name[:-4]+'.wav', format="wav")
            wav_file1 = datapath1+uploaded_file1.name[:-4]+'.wav'
            y, sr = librosa.load(wav_file1)
    st.set_option('deprecation.showfileUploaderEncoding', False)        
    st.subheader("choose a wave file for test")        
    uploaded_file2 = st.file_uploader("Select file")
    if uploaded_file2 is not None:
            
            
            audio_bytes = uploaded_file2.read()
            st.audio(audio_bytes, format='audio/wav')
            sound = AudioSegment.from_wav(datapath2+uploaded_file2.name)
            sound.export(datapath2+uploaded_file2.name[:-4]+'.wav', format="wav")
            wav_file2 = datapath2+uploaded_file2.name[:-4]+'.wav'
            y, sr = librosa.load(wav_file2)



import streamlit as st

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import os
from PIL import Image
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential 
import warnings
warnings.filterwarnings('ignore')

import IPython
from scipy.io.wavfile import read,write
import noisereduce as nr
from IPython.display import Audio
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,5):
    path ="C:\\Users\\renis\\audio"
    header += f' mfcc{i}'
header += ' label'
header = header.split()
print(header)
file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    path=os.path.dirname(__file__)
   #my_file = path+'audio'

audio_file = path+'neetha'.split()
#t=audio_file[0]
for t in audio_file:
    #t=t+t[1]
    for filename in os.listdir(path+'\\'+t):
        #t=t+t[1]
        name =  path+'\\'+t+'\\'+filename
        sname =f' {name}'
        y, sr = librosa.load(name, mono=True, duration=1)
        [rate, data] = sr,y
        data=data/1.0
# select section of data that is noise
        noisy_part = data[10000:15000]
#name=np.ndarray
        reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
        y=reduced_noise
        #st.write(reduced_noise)
        rmse = librosa.feature.rms(y=y)[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {t}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split()) 
print(filename)
st.write(filename)
Audio(reduced_noise, rate=sr)
st.subheader("Audio{}".format(Audio(reduced_noise, rate=sr)))

import pandas as pd
import csv

data = pd.read_csv('dataset.csv',index_col=False,skiprows=[1])
   # data.columns=data.columns.str.strip()
   # data.drop(data.index[i])
    #data.to_csv('dataset.csv',index=False)
data['name']= data['filename'].astype(str).str[:1]             
#data.head()# Dropping unneccesary columns
#ata=data.drop(['filename'],inplace=True)
data.head()
data.drop(columns=['filename'],axis=1,inplace=True)
#data=data.drop(columns=['label'],axis=1,inplace=True)
#data=data.drop(columns=['chroma_stft'],axis=1)
data.shape
#print(data)
st.write(data.head())
audio_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(audio_list)#Scaling the Feature columns
data.tail()
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
st.write(data.head())
from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
import numpy
from tensorflow.keras import layers
from tensorflow.keras import models
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(13, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
import tensorflow.keras

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1)

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test,y_test),
                    epochs=100,
                    batch_size=128,
                )
with model_training:
        
        sel_col,disp_col=st.columns(2)
        max_depth=sel_col.slider('time to train the model',min_value=10,max_value=100)
        n_estimators=sel_col.selectbox('here you get to choose the hyperparametersof the model and see how the performance changes',options=[100,200,'No'],index=0)
        input_feature=sel_col.text_input('')
        st.subheader("Train Set Score: {}".format (X_train))
        st.subheader("Test Set Score: {}".format(y_train))
from matplotlib import pyplot
fig = plt.figure()
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.show()
st.pyplot(fig)
CREATE_CSV_FILES=True
TEST_CSV_FILE="test.csv"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa
import csv
import os
import IPython

def extractwavFeatures(soundFilesFolder,csvFileName):
    print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    st.write(header)
    for i in range(1, 6):
            header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ',header)
    st.table(header)
    file = open(csvFileName, 'w', newline='')
#with file:
    writer = csv.writer(file)
    writer.writerow(header)
    #path="C:\\Users\\renis\\audio"
    testv= 'neethaL uday1'.split()
    #trainv='five four hello tabala three two'.split()
    #soundFilesFolder= "C:\\Users\\renis\\soundtest" 
     #for t in audiotest:
    #t=t+t[1]
    for filename in os.listdir(soundFilesFolder):
        #t=t+t[1]
            name = f'{soundFilesFolder}\\{filename}'
            #sname =f' {name}'
            y, sr = librosa.load(name, mono=True, duration=3)
            rmse = librosa.feature.rms(y=y)[0]
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            writer.writerow(to_append.split())
            st.write(to_append.split())
    file.close()
    print("end of extractwavfeatures")
    st.subheader("end of extractwavfeatures")
if(CREATE_CSV_FILES==True):
    extractwavFeatures("https://github.com/neethus-123/speaker-identification/tree/main/soundtest/testv",TEST_CSV_FILE)
import pandas as pd
import csv
from sklearn import preprocessing
def preprocessData(csvFileName):
    print(csvFileName+ "will be preprocessing")
    data=pd.read_csv(csvFileName,index_col=False)
    filenameArray=data['filename']
    speakerArray=[]
    for i in range(len(filenameArray)):
        speaker=filenameArray[i]
        if speaker==wav_file1:
            speaker="0"
        elif speaker==wav_file2:
             speaker="1"
       # elif speaker=="S":
        #     speaker="2"
        #elif speaker=="S":
        #    speaker="3"
       # elif speaker=="N":
         #   speaker="4"
        #elif speaker=="S":
        #    speaker="5"
       # elif speaker=="R":
       #     speaker="6"
        else:
             speaker="2"
        speakerArray.append(speaker)
    data["name"]=speakerArray
    data=data.drop(['filename'],axis=1)
    data=data.drop(['label'],axis=1)
    data=data.drop(['chroma_stft'],axis=1)
    #data=data.drop(['name'],axis=1)
    data.shape
    #data.info()
    print("preprocessing is finished")
    print(data.head())
    return data
testdata=preprocessData(TEST_CSV_FILE)
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
#X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)
print("x from training data:",X_train.shape)
#print("x from validation data:",X_val.shape)
print("x from test data:",X_test.shape)
st.subheader("x from training data:")

st.table(X_train.shape)
st.subheader("x from test data:")
st.table(X_test.shape) 
def getSpeaker(speaker):
    speaker=str(speaker)
    if speaker=="0":
        return wav_file1
    elif speaker=="1":
        return wav_file2
    #elif speaker=="2":
    #    return"Saku"
    #elif speaker=="3":
    #    return"Sakunthala"
    #elif speaker=="4":
    #    return"Neetha"
    #elif speaker=="5":
    #    return"Sakunthala"
   # elif speaker=="6":
    #    return"Ravindran"
    else:
        speaker="unknown"

def printPrediction(x_data,y_data,printDigit):
    print('\n#Generate prediction')
    st.subheader("\n#Generate prediction")
    for i in range(len(y_data)):
        prediction=getSpeaker(model.predict_classes(x_data[i:i+1])[0])
        speaker=getSpeaker(y_data[i])
        if printDigit==True:
            print('Number={0:d}, y={1:10s}- prediction={2:10s} match={3}'.format(i, speaker, prediction, speaker==prediction))
            st.subheader('speaker {} and prediction{} match{} '.format(i, speaker, prediction, speaker==prediction))
        else:
                print("y={0:10s} prediction={1:10s} match={2}".format(speaker, prediction, speaker==prediction))
                st.subheader('speaker {} and prediction{} match{} '.format(speaker, prediction, speaker==prediction))
                
import numpy as np
from keras import backend as k
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras import layers

def report(x_data,y_data):
    Y_pred=model.predict_classes(x_data)
    y_test_num=y_data.astype(np.int64)
    conf_mt=confusion_matrix(y_test_num,Y_pred)
    print(conf_mt)
    
    plt.matshow(conf_mt)
    plt.show()
    print('\nclassification report')
    target_names=[wav_file1,wav_file2,"unknown"]
    st.subheader("cassification report for test data{}".format(conf_mt))
    print(classification_report(y_test_num,Y_pred))
    st.subheader("classification report {}".format(classification_report(y_test_num,Y_pred)))
print('\n TEST DATA \n')
score=model.evaluate(X_test,y_test)
st.subheader("\n TEST DATA \n")
print("%s: %.2f%%" % (model.metrics_names[1],score[1]*100))
st.subheader("Test Data {}".format("%s: %.2f%%" %(model.metrics_names[1],score[1]*100)))                                         
printPrediction(X_test[0:10],y_test[0:10],False)
print("classification report for test data")
report(X_test,y_test) 
st.subheader("report: {}".format (X_test,y_test))
