# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import moviepy.editor as mp
import tempfile
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.spiceworks.com%2Ftech%2Fartificial-intelligence%2Farticles%2Fwhat-is-reinforcement-learning%2F&psig=AOvVaw3ZZq98CSz30ATHgsH57r1d&ust=1712693298664000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNiht_G1s4UDFQAAAAAdAAAAABAV', use_column_width=True)
    st.title('RLipVision')
    st.info('A comprehensive approach to implement lip vision using Reinforcement Learning')

st.title('RLipVision') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

def convert_mpg_to_mp4(file_path):
    # Load the MPG video
    video_clip = mp.VideoFileClip(file_path)
    # Create a temporary file to save the MP4 version
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()  # Close the file so it can be used by MoviePy
    # Save the video clip as an MP4 file
    video_clip.write_videofile(temp_file.name, codec='libx264', fps=24)
    # Close the video clip
    video_clip.close()
    return temp_file.name

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        mp4_file_path = convert_mpg_to_mp4(file_path)
        # Rendering inside of the app
        video = open(mp4_file_path, 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video_np = video.numpy()
        
        video_np_squeezed = np.squeeze(video_np)

        imageio.mimsave('animation.gif', video_np_squeezed,fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
