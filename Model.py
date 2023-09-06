#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[2]:


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
        return result


# In[3]:


#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'depression',
  '06':'suicide',
  '07':'disgust',
  '08':'surprised',

}

#DataFlair - Emotions to observe
observed_emotions=['neutral', 'calm', 'happy', 'sad', 'depression', 'suicide', 'disgust', 'surprised']


# In[4]:


# DataFlair - Load the data and extract features for each sound file
Crema = "D:\\Final Year\\Crema\\"
Savee = "D:\\Final Year\\Savee\\"
Tess = "D:\\Final Year\\Tess\\"
x,y=[],[]
def load_data(test_size=0.2):

#<------------------------ Ravdess Dataset------------------------>
    for file in glob.glob("D:\\Final Year\\ravdess data\\Actor_*\\*.wav"):
#     for file in glob.glob("D:\\Final Year\\Crema\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
#<------------------------ Ravdess Dataset------------------------>

#<------------------------ Crema Dataset------------------------>
    for wav in os.listdir(Crema):
        info = wav.partition(".wav")[0].split("_")
        if info[2] == 'SAD':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif info[2] == 'DEP':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif info[2] == 'DIS':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif info[2] == 'SUI':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion) 
            
        elif info[2] == 'HAP':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif info[2] == 'NEU':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
#<------------------------ Crema Dataset------------------------> 

#<------------------------ Tess Dataset------------------------>

#     tess_directory_list = os.listdir(Tess)
    
    for dir in os.listdir(Tess):
        for wav in os.listdir(os.path.join(Tess, dir)):
            info = wav.partition(".wav")[0].split("_")
            emo = info[2]
            if emo == "depression":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)

            elif emo == "disgust":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
                
            elif emo == "suicide":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
                
            elif emo == "happy":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
                
            elif emo == "neutral":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
                
            elif emo == "ps":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
                
            elif emo == "sad":
                file_name=os.path.basename(file)
                emotion=emotions[file_name.split("-")[2]]
                if emotion not in observed_emotions:
                    continue
                feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)              
#<------------------------ Tess Dataset------------------------>   

#<------------------------ Savee Dataset------------------------>

    for wav in os.listdir(Savee):
        info = wav.partition(".wav")[0].split("_")[1].replace(r"[0-9]", "")
        emotion = re.split(r"[0-9]", info)[0]
        if emotion=='de':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif emotion=='d':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif emotion=='su':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif emotion=='h':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif emotion=='n':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        elif emotion=='sa':
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            
        else:
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)

#<------------------------ Savee Dataset------------------------>         
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
        
        


# In[5]:


#DataFlair - Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)


# In[6]:


#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))


# In[7]:


#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# In[8]:


#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[9]:


#DataFlair - Train the model
model.fit(x_train,y_train)


# In[10]:


#DataFlair - Predict for the test set
y_pred=model.predict(x_test)


# In[11]:


#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[12]:


from sklearn.metrics import accuracy_score, f1_score


# In[13]:


f1_score(y_test, y_pred,average=None)


# In[14]:


import pandas as pd
df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df.head(10)


# In[15]:


import pickle
# Writing different model files to file
with open( 'modelForPrediction1.sav', 'wb') as f:
    pickle.dump(model,f)

