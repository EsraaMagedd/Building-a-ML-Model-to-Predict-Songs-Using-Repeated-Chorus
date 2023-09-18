## Hit-Songs-Using-Repeated-Chorus
This project aims to build a website to predict if a song will be a popular song or not using either a Spotify link or directly uploading the song, using machine learning algorithms to do so.<br>
### project owner: Technocolabs softwares company<br>
### contact: https://github.com/Technocolabs100<br>
### Mentor: Nour Ibrahim<br>
### contact: https://github.com/Nour-Ibrahim-1290<br>
## Data Collection

The data collection pipeline will consist of the following steps:
1. Collecting the names of the hot-100 songs from the last 5 years as the popular songs and unpopular songs of the same artists from Billboard.com using billboard.py API. This API can programmatically retrieve chart data, such as the title, artist, and rank of songs, as well as other information such as the peak position, last position, and number of weeks on the chart. 

2. Downloading the full songs from Youtube using youtube-dl-api-server.It provides a REST API for getting information about videos from YouTube and other video sites. It is powered by youtube-dl, so it can be used to download videos as well.

3. Extracting the repeated chorus using pychorus library. It is used for generating song choruses. It is based on the idea of Markov chains, which are probabilistic models that can be used to generate text that is similar to a given text corpus.

4. Extracting the audio features of the hooks using librosa. It is used for music and audio analysis. It provides a number of features for analyzing audio data and we extracted 11 major features which are: <br>
•	choroma_stft <br>
•	chroma_cqt <br>
•	chroma_cens <br>
•	spectral_centroid <br>
•	mfcc <br>
•	rms <br>
•	tonnetz <br>
•	crossing_rate <br>
•	spectral_bandwidth <br>
•	spectral_contrast <br>
•	spectral_rolloff <br>
we used functions to extract these features at different statistics which are:<br>
•	minimum <br>
•	maximum<br>
•	mean<br>
•	median <br>
•	standard diviation<br>
•	skew <br>
•	kurtosis<br>
after extraction we had a total of 518 features and a total of 1767 rows and three columns one for artist name song name and popularity.


## Exploratory Data Analysis (EDA)

Once the data has been collected and preprocessed, it is important to perform exploratory data analysis (EDA) to gain insights into the data. This includes tasks such as:

Data exploration involves understanding the distribution of the data, identifying outliers, and checking for missing values.
Data visualization: This involves creating visualizations of the data to help understand the relationships between different features.
And to study the data we used multiple plots.<br>
We used:<br>
•	Boxplots to check for outliers<br>
•	Hist plots to check for the data distribution<br>
•	 Heat maps to check for correlation and correlation patterns<br>
## data preprocessing
We preprocessed the data by applying standard scaler to it and did PCA on it, after applying PCA we had a total of 171 components after that we did some univariant analysis, Bivariant analysis, and Multivariant analysis to study our new data better to make sure that everything is okay.

 ## Model Building
Once the data has been explored, it is time to build machine learning models.
So we were ordered to make two models SVM and XGBboost 
Each member choose a model of the two to build and the leaders choose the best model to use for the final project and we rested on using SVM as it had the best results of (82% accuracy and 89% f1 score for the unpopular songs and 60% f1-score for the popular song) <br>
 ![image](https://github.com/Technocolabs100/Building-a-Model-to-Predict-Hit-Songs-Using-Repeated-Chorus/assets/133379726/bf92e87d-1ad8-46b9-a912-29fc468e304f)

## Development lifecycle
 ![image](https://github.com/Technocolabs100/Building-a-Model-to-Predict-Hit-Songs-Using-Repeated-Chorus/assets/133379726/277cb3f6-c4c5-48d1-a79e-b9db907dd846)

The team who worked on this was:<br>
•	Ahmed Sallam (lead -- Roaming on every task)<br>
•	Abdulrahman Ahmed(co-lead -- Roaming on every task)<br>
•	AbdulRaouf Monir (AWS -- Roaming on front and backend)<br>
•	Mahmoud Khaled (Backend)<br>
•	Paula Samir (Backend)<br>
•	Youssef Wael (backend)<br>
•	Rithik Koti (pipeline)<br>
•	Sherry Fady (frontend)<br>
•	Jessica Mansi (frontend)<br>
•	Esraa Maged (frontend)<br>
•	Shehab Ashraf (frontend)<br>
## front end
![image](https://github.com/Technocolabs100/Building-a-Model-to-Predict-Hit-Songs-Using-Repeated-Chorus/assets/133379726/a3af7dcf-6b2c-420a-8c1b-371c545934ac)

We used streamlit to build a simple minimalist front end consisting of one page only, that has a Text box to put the link from spotify in it and a box to upload the audio mp3 file of a song to the webpage and the prediction
 
## pipline
The pipeline used the same PCA, Standard scaler and model as the machine learning model and they were extracted using pickle library and plugged in the pipeline for that work and the pipline was then downloaded using pickle and sent to the backend team
## Backend
We made a backend the consisted of taking input as a Spotify link to the extract the name of the song using rapid spotify API to then pass it through youtube dlp and download it and extract the features refer to data extraction for farther explanation on the process and these features were passed to the pipeline
## pipline
The pipeline used the same PCA, Standard scaler and model as the machine learning model and they were extracted using pickle library and plugged in the pipeline for that work and the pipline was then downloaded using pickle the pipeline made a prediction of zero or one, zero for unpopular and one for the popular that got sent to the front end to display popular or unpopular
## deployment (AWS)
We used the most professional option which was deploying via AWS so the whole project is ran through a virtual machine the runs linux and connected to AWS to work online 
## Conclusion
This project is the result of the collaboration of all of Team_c members and efforts with Technocolabs that resulted in this web app, that predicts if a song is popular or not with an accuracy of 82%. <br>
We aim to improve this tools accuracy and usability for future use, we built a very robust and modular plateform to change it yourself to suit your use and add more funcftionalties to it to help you with your future music releases or just marketing.<br>
We would like to thank both Technocolabs and Nour Ibrahim for their immense efforts with us to push this project out.
