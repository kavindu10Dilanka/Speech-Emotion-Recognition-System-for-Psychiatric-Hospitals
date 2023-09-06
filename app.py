from flask import Flask,render_template, request
import Model
import pickle

app = Flask(__name__) 

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/result" ,methods = ['POST', "GET"])
def result():
    output = request.form.to_dict()
    name = output["name"]

    # *************************************************************************************************

    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

    # *********Spliting the wav file in to segments

    from pydub import AudioSegment 
    from pydub.utils import make_chunks 

    file_size = []
    # a = "real_conversation.wav"
    myaudio = AudioSegment.from_file("D:\\inputs\\"+name , "wav") #input audio
    chunk_length_ms = 5000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks):
        chunk_name = 'D:\\clips\\clip'+ str(i) +'.wav' 
        print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav") 
        file_size.append(i)
    # *******************************************

    print(len(file_size))


    # In[24]:


    split_emo = []
    count = 0
    while (count < len(file_size)):
        feature=Model.extract_feature('D:\\clips\\clip'+ str(count) +'.wav', mfcc=True, chroma=True, mel=True)
        feature=feature.reshape(1,-1)
        prediction=loaded_model.predict(feature)
        split_emo.append(prediction)
        count = count + 1


    # In[26]:


    print("Conversation duration "+str(len(file_size)*5000/1000/60) + " minutes" )


    # In[28]:


    for i in range(len(split_emo)):
        print (i, end = " ")
        print(*split_emo[i])


    # In[29]:


    print(*split_emo[0:30])


    # In[33]:

    emo_sessions = [] 

    time_one = 0
    time_two = 5

    while (time_one < len(split_emo)):

        print("25 sec")
        print(*split_emo[time_one:time_two])
        print(" ")
        
        for i in split_emo[time_one:time_two]:
            if i not in emo_sessions:
                emo_sessions.append(i)  
        
        time_one = time_one + 5
        time_two = time_two + 5
  
    status = split_emo[-1]
    emo_sessions.append(status)


    print ("The list after removing duplicates : " + str(emo_sessions))


    # In[32]:


    # ['neutral*', 'calm*', 'happy*', 'sad*', 'angry*', 'fearful*', 'disgust*', 'surprised']

    # status = split_emo[-1]

    if status == "happy":
        print("Happy - Patient")
        a = "happy"

    elif status == "calm":
        print("Calm - Patient")
        a = "calm"

    elif status == "neutral":
        print("Neutral - Patient")
        a = "neutral"

    elif status == "sad":
        print("Sad - Patient")
        a = "sad"


    elif status == "depression":
        print("depression - Patient")
        a = "depression"



    elif status == "suicide":
        print("suicide - Patient")
        a = "suicide"


    elif status == "disgust":
        print("disgust - Patient")
        a = "disgust"

    elif status == "surprised":
        print("surprised can't be judge")
        a = split_emo[-2]
        print(a)




        # *************************************************************************************************

    return render_template("index.html", name = str(emo_sessions), sentiment=a)

if __name__ == '__main__':
    app.run(debug= True,port=5001)