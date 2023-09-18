import streamlit as st
import pickle
import requests
import librosa
import numpy as np
import pandas as pd
import yt_dlp
import base64
import plotly.express as px

from pprint import pprint
from youtube_search import YoutubeSearch
from pychorus import find_and_output_chorus
import scipy
from sklearn.pipeline import Pipeline

# loading our XGBOOST model
# pickle_in_XGBOOST = open("model_pickle.pkl", "rb")
# pickle_in_PCA = open("pca.pkl", "rb")
# pickle_in_STD = open("std (1).pkl", "rb")


# XGBOOST = pickle.load(pickle_in_XGBOOST)a
# PCA = pickle.load(pickle_in_PCA)
# STD = pickle.load(pickle_in_STD)

pipe = pickle.load(open("pipeline.pkl", "rb"))


# Loading Our Model Pipeline
# pipe = Pipeline([('scaler', STD),
#                  ('pca', PCA),
#                  ('classifier', XGBOOST)]
#                 )

def get_video_link_by_name(video_name):
    results = YoutubeSearch(video_name, max_results=1).to_dict()

    if results:
        video_id = results[0]['id']
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        return video_link

    return None


def download_audio_from_youtube(video_url, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'opus',
            'preferredquality': '192',
        }],
        'ffmpeg-location': r''
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def get_audio_duration(file_path):
    audio, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration


def success_message():
    st.success("File has been uploaded successfully")


# Function to download audio from YouTube using the artist and song name
def download_audio(artist, song):
    video_link = get_video_link_by_name(f'{artist} - {song}')
    download_audio_from_youtube(video_link, 'audio')


# Function to find the chorus start time in seconds
def find_chorus_start_time():
    if get_audio_duration('audio.opus') > 480:
        raise ValueError("Video length exceeds 8 minutes.")

    # this part of code is flawed and not clean to understand.. it raises an error when chorus duration cant reach 15 sec.. but actually this is required :D
    chorus_start_sec = 15
    while chorus_start_sec >= 0:
        chorus_start_sec = find_and_output_chorus("audio.opus", "output.mp3", chorus_start_sec)
        if chorus_start_sec is not None:
            return chorus_start_sec
        chorus_start_sec -= 1
    return None


functions = {
    'mfcc': lambda audio, sr: librosa.feature.mfcc(y=audio, sr=sr),
    'chroma_stft': lambda audio, sr: librosa.feature.chroma_stft(y=audio, sr=sr),
    'chroma_cqt': lambda audio, sr: librosa.feature.chroma_cqt(y=audio, sr=sr),
    'chroma_cens': lambda audio, sr: librosa.feature.chroma_cens(y=audio, sr=sr),
    'rms': lambda audio, sr: librosa.feature.rms(y=audio),
    'spectral_centroid': lambda audio, sr: librosa.feature.spectral_centroid(y=audio, sr=sr),
    'spectral_bandwidth': lambda audio, sr: librosa.feature.spectral_bandwidth(y=audio, sr=sr),
    'spectral_contrast': lambda audio, sr: librosa.feature.spectral_contrast(y=audio, sr=sr),
    'spectral_rolloff': lambda audio, sr: librosa.feature.spectral_rolloff(y=audio, sr=sr),
    'tonnetz': lambda audio, sr: librosa.feature.tonnetz(y=audio, sr=sr),
    'zero_crossing_rate': lambda audio, sr: librosa.feature.zero_crossing_rate(y=audio)
}


# Function to extract a feature statistics from the chorus
def extract_soundtrack_feature_statistics(audio_feature):
    min_val = np.min(audio_feature, axis=1)
    mean_val = np.mean(audio_feature, axis=1)
    median_val = np.median(audio_feature, axis=1)
    max_val = np.max(audio_feature, axis=1)
    std_val = np.std(audio_feature, axis=1)
    skew_val = scipy.stats.skew(audio_feature, axis=1)
    kurtosis_val = scipy.stats.kurtosis(audio_feature, axis=1)
    song_track_features = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val),
                                         axis=0)
    return song_track_features


def create_chorus_file():
    if get_audio_duration('audio.opus') > 480:
        raise ValueError("Video duration exceeds 8 minutes.")
    chorus_duration = 15
    while chorus_duration > 5:
        chorus_start_sec = find_and_output_chorus("audio.opus", "output.mp3", chorus_duration)
        if chorus_start_sec is not None:
            return chorus_duration

        chorus_duration -= 1

    raise ValueError("No chorus detected")


# Setting Background

df = px.data.iris()


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("background12.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
position: relative;
@media (min-width: 1510px) {{
    height: 100%;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    }}
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

new_features_dict = {}

key = "text"
input_style = f"""
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    font-size: 16px;
    width: 100%;
    background-color: rgb(188, 191, 214);
    color: #f0f0f0;
"""
text_style = """
<style>
body {

    font-size: 16px;
    color: white;
    font-weight: 700;
    margin-top: 50px;
}
</style>
"""
st.markdown(text_style, unsafe_allow_html=True)

st.write(
    "Paste here the Song URL or name from Spotify. For name, please the enter the name in the following format artist name - song name ")
st.text_input(
    "Song URL or name",
    key="text")

uploaded_file = st.file_uploader("Upload Song", type=['mp3'], on_change=success_message)

if st.button("Predict"):

    if st.session_state.text:
        # Get the artist of the song and the song name using RapidAPI
        song_url = st.session_state.text
        song_id = song_url[31:]
        song_id = song_id.split("?", 1)[0]
        url = "https://spotify23.p.rapidapi.com/tracks/"
        querystring = {"ids": song_id}
        headers = {
            "X-RapidAPI-Key": "65f04252bdmshd63b320a130ca80p19c4d1jsnc3779e30517d",
            "X-RapidAPI-Host": "spotify23.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        # pprint(response.json())

        artist = response.json()['tracks'][0]['artists'][0]['name']
        name = response.json()['tracks'][0]['name']
        #st.write("Song ID: ", song_id)
        #st.write("Artist Name: ", artist)
        #st.write("Song name: ", name)

        # Download the song
        download_audio(artist, name)
        print(artist + ' - ' + name)

    elif uploaded_file is not None:
        st.write("File is being processed")
        bytes_data = uploaded_file.getvalue()
        with open("audio.opus", "wb") as binary_file:
            binary_file.write(bytes_data)

    # create chorus file: "output.mp3"
    chorus_duration = create_chorus_file()
    # new_features_dict['chorus_duration'] = chorus_duration

    chorus_audio, sr = librosa.load('output.mp3', sr=None)
    for function_name, function in functions.items():

        # Extract a soundtrack feature
        audio_feature = function(chorus_audio, sr)
        audio_feature_statistics = extract_soundtrack_feature_statistics(audio_feature)

        # Update the dataframe with the chorus extracted features and the chorus duration
        for i in range(len(audio_feature_statistics)):
            new_features_dict[function_name + '_' + str(i + 1)] = audio_feature_statistics[i]
    pprint(new_features_dict)
    # new_features_dict = pd.DataFrame(new_features_dict.values())
    data_X = pd.DataFrame()
    data_X.loc[0, new_features_dict.keys()] = new_features_dict.values()
    pprint(data_X)
    # pprint(type(new_features_dict))
    pprint(new_features_dict)
    res = pipe.predict(data_X)[0]
    pprint(f"res = {res}")
    if res == 1:
        st.write("SONG IS POPULAR!")
    else:
        st.write("SONG IS UNPOPULAR!")