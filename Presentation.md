# Table of Contents
1. [Transforming MP3 Files into .wav files](#transformation)
2. [Reading .wav files with Librosa and analyzing with Vampy](#Vamp)
3. [Sonic Annotator](#Sonic)
4. [Spotify API Analysis](#Spotipy)
5. [Data Analaysis](#Data)
5. [Creating a new model based off Vamp data](#New)

# Required Packages 
>Run the following cell to import the required packages:


```python
#All of the classic plugins to run data analysis and create dataframes
import numpy as np
import pandas as pd
from os import path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from scipy import linalg

#Plugins that allow us to listen to a song and pull data from that specific song ; includes notes, 
import librosa
from pydub import AudioSegment
import vamp

#All of the plugins used to pull data from Spotify and communicate with Spotify's API within Python
import spotipy
import spotipy.oauth2
import spotipy.util as util
import os

#Reconfigures the path for Vamp plugins ; allows us to use plugins that are installed on the local computer
! export VAMP_PATH=/Library/Audio/Plug-Ins/ 
! echo $VAMP_PATH
```

    


Transforming MP3 Files into .WAV files
---
<a class="anchor" id="transformation"></a>

>Below, we use the library for AudioSegment in order to source our .mp3 files and convert them into .wav files. Below, we ran an example in which we took Justin Beiber's song Yummy and converted it into a .wav file


```python
src = 'MP3_Files/Yummy.mp3'
dst = 'wav_Files/Yummy.wav'

Example_song = AudioSegment.from_mp3(src)
Example_song.export(dst, format="wav")
```




    <_io.BufferedRandom name='wav_Files/Yummy.wav'>



Reading .wav files with Librosa and analyzing with Vampy
---
<a class="anchor" id="Vamp"></a>

>Using librosa, we can individually load in a song in .wav format. Once loaded, librosa allows Python to see the data and rate behind each element in the song. This will later be useful for Vampy. Vampy is a plugin that allows python to run Vamp. Now what is Vamp? Vamp is a open source music package that has individual plugins (installed into your local computer). These plugins allow you to read and analyze songs.


```python
# Using librosa to create two new items named data and rate
data, rate = librosa.load('wav_Files/Yummy.wav')
```

To see what plugins can be utilized by Vampy the following command is run:


```python
#Below are all the plugins that are currently installed on the local computer and are able to be run. 
print(vamp.list_plugins())
```

    ['bbc-vamp-plugins:bbc-energy', 'bbc-vamp-plugins:bbc-intensity', 'bbc-vamp-plugins:bbc-peaks', 'bbc-vamp-plugins:bbc-rhythm', 'bbc-vamp-plugins:bbc-spectral-contrast', 'bbc-vamp-plugins:bbc-spectral-flux', 'bbc-vamp-plugins:bbc-speechmusic-segmenter', 'beatroot-vamp:beatroot', 'nnls-chroma:chordino', 'nnls-chroma:nnls-chroma', 'nnls-chroma:tuning', 'silvet:silvet']



```python
# chroma = vamp.collect(data, rate, "silvet:silvet")
# stepsize, chromadata = chroma["matrix"]
# plt.imshow(chromadata)
```


```python
# Beat = vamp.collect(data, rate, "beatroot-vamp:beatroot")
# Chord = vamp.collect(data, rate, "nnls-chroma:chordino")
```

Sonic Annotator
---
<a class="anchor" id="Sonic"></a>

>Sonic Annotator is a command line program for batch extraction of audio features from multiple audio files. The basic idea is to abstract out the process of doing the feature extraction from the extraction methods, by using Vamp plugins for feature extraction. The output format is .csv or .txt.

The following line refers to the command line program (sonnic annotator) and allows us to see all the plugins installed under vamp that are able to be run by sonic. 


```python
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -l
```


```python
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:bbc-vamp-plugins:bbc-rhythm:tempo --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Tempo
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:silvet:silvet:notes --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Notes
```


```python
def get_beat(file):
    with open(file) as f:
        beats = {}
        for line in f:
            beats[line.strip().split(',')[0][68:-5]] = float(line.strip().split(',')[2][:7])
    return beats
songs = []
beat = []
for key, value in get_beat('Tempo').items():
    songs.append(key)
    beat.append(value)

Tempo = pd.DataFrame({'Names' : songs, 'bpm' : beat})
Tempo = Tempo.set_index('Names')
```


```python
def get_chroma(file):
    with open(file) as f:
        name = ''
        matrix = []
        places = []
        count = 0
        for line in f:
            matrix.append(line.strip().split(','))
            if type(line.strip().split(',')[0]) is str:
                places.append(count)
            count += 1
        name = matrix[0][0][58:-1]
        new_mat = pd.DataFrame(matrix)
        new_mats = np.split(new_mat, places, axis = 0)
        return new_mats

mats = get_chroma('Matrix')
```


```python
def get_notes(file):
    '''This function reads in a file that contains information about every note in every song, and returns a list containing lists of each song and the important features of the notes in them. The features it returns are the song name, the number of notes, the frequency of notes, the average length of the note, the average frequency (hz), the standard deviation of notes (hz), the most common note (hz), and how often the most common note occurs.'''
    with open(file) as f:
        songs = []
        for line in f:
            if 'Users' in line.strip().split(',')[0]:
                songs.append([])
            songs[-1].append(line.strip().split(','))
        song_data = []
        for song in songs:
            number_notes = len(song)
            freq_notes = len(song)/float(song[-1][1])
            song_name = song[0][0][68:-5]
            htz_list = []
            length_list = []
            for element in song:
                length_list.append(float(element[2]))
                htz_list.append(float(element[3]))
            avg_length = np.mean(length_list)
            avg_htz = np.mean(htz_list)
            most_common_note = Counter(htz_list).most_common()[0]
            std_htz = np.std(htz_list)
            song_data.append([song_name, number_notes, freq_notes, avg_length, avg_htz, std_htz, most_common_note[0], most_common_note[1]])
        return song_data
    
```


```python
Notes = pd.DataFrame(get_notes('Notes'))
Notes = Notes.rename(columns = {0 : 'Names' , 1: 'Number Notes' , 2 : 'Frequent Notes', 3 : 'Average Length of Note', 4 : 'Average Htz' , 5 : 'Standard Deviation of Htz' , 6 : 'Most Common Note' , 7 : 'Most Common Note'})
Notes = Notes.set_index('Names')

```

Spotify API Analysis
---
<a class="anchor" id="Spotipy"></a>
>Spotipy is a jupyter notebook package that allows python to communicate with the spotify API


```python
scope = 'user-library-read'
token = util.prompt_for_user_token('ethan.dinh',scope,client_id='bd4a55649bd242cb94ceadaf98c1ff21',client_secret='67d9d6c64d024428a93bbd8f669b503f',redirect_uri='http://localhost:8888/notebooks/Final_Project%20/Music%20reader%20experiments.ipynb')
sp = spotipy.Spotify(auth=token)

playlist_id = 'spotify:playlist:37i9dQZEVXbMDoHDwVN2tF'
results = sp.playlist(playlist_id)

names = []
for entry in results['tracks']['items']:
    names.append(entry['track']['name'])
    
tracks = []
for entry in results['tracks']['items']:
    tracks.append(entry['track']['uri'])
    
ratings = []
for entry in results['tracks']['items']:
    ratings.append(entry['track']['popularity'])
    
analysis = []
for i in range(len(tracks)):
    analysis.append(sp.audio_features(tracks[i]))
    
popularity = pd.DataFrame({'rating':ratings, 'Names':names})
popularity = popularity.set_index('Names')
popularity = popularity.rename(index={'Sunflower - Spider-Man: Into the Spider-Verse': 'Sunflower'})
```


```python
ana = pd.DataFrame(analysis[0])
Df = pd.DataFrame(analysis[0])
for i in range(1,50):
    ana = ana.append(pd.DataFrame(analysis[i]))
    
for i in range(1,50):
    Df = Df.append(pd.DataFrame(analysis[i]))
```


```python
my_list = []
for song in analysis:
    my_list.append(song[0]['tempo'])
```


```python
ana['Names'] = names
ana = ana.set_index('Names')
ana = ana.drop(columns = ['type', 'id', 'uri', 'track_href', 'analysis_url'])
ana = ana.rename(index={'Sunflower - Spider-Man: Into the Spider-Verse': 'Sunflower'})
ana.to_csv('Spotify.csv')
```

Data Analysis (Spotify API)
---
<a class="anchor" id="Data"></a>
>Spotipy is a jupyter notebook package that allows python to communicate with the spotify API

Scales the dataframe to ensure that the weight is unaffected by the size of the values


```python
scaler = StandardScaler()
scaler.fit(ana)
rank = []
for i in range(1,51):
    rank.append(i)
```

### The following creates a model to predict the rank of the song on Spotify's Global Top 50:


```python
A=np.matrix(ana)
b=np.matrix(rank)
b = b.T
X=linalg.inv(A.T*A)*(A.T)*b
print(A.shape, b.shape)
```

    (50, 13) (50, 1)


Prints the weight of each catagory (Both positve and negative)


```python
categories = ana.columns                           # column names

tuples = []                                        # create tuples containing the category weights and names
for i in range(len(categories)):
    tuples.append((X[i][0,0], categories[i]))
    
tuples.sort(reverse = True)                        # sort in decending order

for i in range(len(categories)):                   # print the output
    print(tuples[i])
```

    (62.78186683534804, 'energy')
    (8.763829026087407, 'valence')
    (5.754218079936674, 'speechiness')
    (-4.589870200537435e-07, 'duration_ms')
    (-0.05924298705592834, 'mode')
    (-0.17499091666564215, 'tempo')
    (-0.19424044245721284, 'acousticness')
    (-0.781781722126988, 'time_signature')
    (-0.8792381028852305, 'key')
    (-4.30644117093755, 'loudness')
    (-15.290106547492408, 'liveness')
    (-17.85018053625503, 'danceability')
    (-38.43631962912502, 'instrumentalness')


Testing whether the scaling would help decrease the range in weights of each catagory


```python
A_scaled = np.matrix(scaler.transform(ana))
```


```python
X_scaled=linalg.inv(A_scaled.T*A_scaled)*(A_scaled.T)*b

categories = ana.columns                             # column names

tuples = []                                          # create tuples containing the category weights and names
for i in range(len(categories)):
    tuples.append((X_scaled[i][0,0], categories[i]))
    
tuples.sort(reverse = True)                          # sort in decending order

for i in range(len(categories)):                     # print the output
    print(tuples[i])
```

    (6.853532482657053, 'energy')
    (1.5189883014803613, 'valence')
    (1.1914164375926797, 'speechiness')
    (-0.010960796836248843, 'duration_ms')
    (-0.8554519478004918, 'mode')
    (-1.0087160294700075, 'acousticness')
    (-1.4431828508039113, 'time_signature')
    (-2.7481957863844393, 'liveness')
    (-2.7998124428222075, 'instrumentalness')
    (-3.000798082007339, 'danceability')
    (-3.314043927114307, 'key')
    (-5.342156076269346, 'tempo')
    (-7.537188989649976, 'loudness')



```python
song_title = input("What is the title of your song (Should match Spotify)? ")
dancem = ana.loc[[song_title]]
dancem = scaler.transform(dancem)   
dancem = np.matrix(dancem)
projection_scaled = dancem*X_scaled
print('Estimated rank of your song:',int(projection_scaled))
```

    What is the title of your song (Should match Spotify)? BOP
    Estimated rank of your song: -2



```python
error = 0
for i in range(len(A)):
    projection = A[i][:]*X
    newerror = abs(projection[0,0] - b[i,0])
    error = error + newerror
    
print('Average Absolute Prediction Error:', error/len(A))
```

    Average Absolute Prediction Error: 10.655638472700097


### The following creates a model to predict the popularity rating of the song on Spotify's Global Top 50:


```python
A=np.matrix(ana)
b_1=np.matrix(ratings)
b_1 = b_1.T
X_1=linalg.inv(A.T*A)*(A.T)*b_1
print(A.shape, b_1.shape)
```

    (50, 13) (50, 1)


Prints the weight of each catagory (Both positve and negative)


```python
categories = ana.columns                           # column names

tuples = []                                        # create tuples containing the category weights and names
for i in range(len(categories)):
    tuples.append((X_1[i][0,0], categories[i]))
    
tuples.sort(reverse = True)                        # sort in decending order

for i in range(len(categories)):                   # print the output
    print(tuples[i])
```

    (20.55886918988135, 'danceability')
    (17.160095933144277, 'energy')
    (11.637222873960944, 'liveness')
    (7.577787545598886, 'time_signature')
    (7.510255480427027, 'valence')
    (5.5613641145991135, 'mode')
    (3.7412508887214635, 'acousticness')
    (1.5031385183483081, 'instrumentalness')
    (0.44854774749917325, 'key')
    (0.18113890542352257, 'tempo')
    (-3.0271007423049868e-05, 'duration_ms')
    (-1.4661499695405427, 'loudness')
    (-11.263535047721042, 'speechiness')



```python
song_title = input("What is the title of your song (Should match Spotify)? ")
dancem = ana.loc[[song_title]]
dancem = np.matrix(dancem)

projection = dancem*X_1
print('The Popularity of your song is:',int(projection))
print('The true rating:', int(popularity.loc[song_title]))
```

    What is the title of your song (Should match Spotify)? BOP
    The Popularity of your song is: 98
    The true rating: 93



```python
song_title = input("What is the title of your song (Should match Spotify)? ")
Uri = sp.search(q=[song_title], type = 'track', limit = 1)
name_1 = []
for entry in Uri['tracks']['items']:
    name_1.append(entry['uri'])
    
for entry in Uri['tracks']['items']:
    pop = entry['popularity']
    
data_temp = sp.audio_features(name_1)
data_temp = pd.DataFrame(data_temp)
data_temp = data_temp.drop(columns = ['type', 'id', 'uri', 'track_href', 'analysis_url'])

dancem = data_temp  
dancem = np.matrix(dancem)
projection_new = dancem*X_1
print('The estimated popularity of your song is:',int(projection_new))
print('The true rating:', pop)
```

    What is the title of your song (Should match Spotify)? Locked Out of Heaven
    The estimated popularity of your song is: 99
    The true rating: 78



```python
error = 0
for i in range(len(A)):
    projection = A[i][:]*X_1
    newerror = abs(projection[0,0] - b_1[i,0])
    error = error + newerror
    
print('Average Absolute Prediction Error:', error/len(A))
```

    Average Absolute Prediction Error: 5.553918306065198


Creating a new model based off Vamp data
---
<a class="anchor" id="New"></a>


```python
NewANA = ana.merge(Notes, left_index = True, right_index = True)
```


```python
NewANA = NewANA.drop(columns = (['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']))

```


```python
NewANA = NewANA.sort_index()
popularity = popularity.sort_index()
popularity_list = popularity['rating'].values.tolist()
```


```python
A_new = np.matrix(NewANA)
b_new=np.matrix(popularity_list)
b_new = b_new.T
X_new=linalg.inv(A_new.T*A_new)*(A_new.T)*b_new
print(A_new.shape, b_new.shape)
```

    (50, 10) (50, 1)



```python
categories = NewANA.columns                           # column names

tuples = []                                        # create tuples containing the category weights and names
for i in range(len(categories)):
    tuples.append((X_new[i][0,0], categories[i]))
    
tuples.sort(reverse = True)                        # sort in decending order

for i in range(len(categories)):                   # print the output
    print(tuples[i])
```

    (62.671211105070256, 'Average Length of Note')
    (10.14723994925517, 'Frequent Notes')
    (1.3040152514846177, 'time_signature')
    (0.553463819117075, 'key')
    (0.05093160869888281, 'Average Htz')
    (0.005351438087299862, 'Most Common Note')
    (8.38831033840526e-05, 'duration_ms')
    (-0.013426779509758388, 'Number Notes')
    (-0.01386938315184956, 'Standard Deviation of Htz')
    (-0.02561054246682852, 'Most Common Note')



```python
song_title = input("What is the title of your song (Should match Spotify)? ")
dancem = NewANA.loc[[song_title]]
dancem = np.matrix(dancem)

projection = dancem*X_new
print('The Popularity of your song is:',int(projection))
print('The true rating:', int(popularity.loc[song_title]))
```

    What is the title of your song (Should match Spotify)? BOP
    The Popularity of your song is: 89
    The true rating: 93



```python
error = 0
for i in range(len(A_new)):
    projection = A_new[i][:]*X_new
    newerror = abs(projection[0,0] - b_new[i,0])
    error = error + newerror
    
print('Average Absolute Prediction Error:', error/len(A_new))
```

    Average Absolute Prediction Error: 5.031142342494794

