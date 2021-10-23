import streamlit as st
# from skimage.io import imread
import matplotlib.pyplot as plt
# from skimage.transform import resize
import pickle
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('trmodel.h5')


rad = st.sidebar.radio("Navigation", ["About","Model","App"])

if rad == "About":
    st.title("SpaceObject Classification WebApp:star:")
    st.image('AI in space.jpg',use_column_width=True)
    st.subheader("The APP is actually made for recognition of space objects")
    if st.checkbox("Description.."):
        st.write("*******The app mainly classifies black holes, pulsar stars and white dwarf***********")
        st.write(
            "**Because, They formed after the death of any neutron or super red giant star having higher chandra shekhara limit**")
        option = ('None','Black Hole', 'Pulsar star', 'White Dwarf')
        option_dis = list(range(len(option)))
        x = st.selectbox("Choose object", option_dis, format_func=lambda x: option[x])
        if x == 1:
            st.image('BlackHole.jpg',use_column_width=True)
            st.text("""A black hole is a place in space where gravity pulls so much that even light can not get out. 
                  The gravity is so strong because matter has been squeezed into a tiny space. 
                  This can happen when a star is dying.
                  Because no light can get out, people can't see black holes. 
                  They are invisible. Space telescopes with special tools can help find black holes. 
                  The special tools can see how stars that are very close to black holes act differently than other stars. 
                  For More Info: Visit""")
            st.markdown('https://www.nasa.gov/audience/forstudents/k-4/stories/nasa-knows/what-is-a-black-hole-k4.html')
        if x == 2:
            st.image('pular star.jpg',use_column_width=True)
            st.text("""Pulsars are spherical, compact objects that are about the size of a large city but contain more mass than the sun. 
            Scientists are using pulsars to study extreme states of matter, search for planets beyond Earth's solar system and 
            measure cosmic distances.
            Pulsar stars are known to be lighthouse of Space. due to immence matter excretion and rotation around it's axis.
            Pulsars also could help scientists find gravitational waves, 
            which could point the way to energetic cosmic events like collisions between 
            supermassive black holes.
            for more info about black hole collision refer LIGO observatory
            """)
            st.markdown('https://www.space.com/32661-pulsars.html')
        if x == 3:
            st.image('WhiteDwarf.jpg',use_column_width=True)
            st.text("""Pushing the limits of its powerful vision, NASA's Hubble Space Telescope uncovered the oldest burned-out stars 
      in our Milky Way Galaxy. 
      These extremely old, dim "clockwork stars" provide a completely independent reading on the age of the universe.

      The ancient white dwarf stars, as seen by Hubble, are 12-13 billion years old. 
      Because earlier Hubble observations show that the first stars formed less than 1 billion years after the universe's 
      birth in the big bang,
      finding the oldest stars puts astronomers well within arm's reach of calculating the absolute age of the universe.""")
            st.markdown('https://www.nasa.gov/multimedia/imagegallery/image_feature_734.html')

if rad == "Model":
    st.subheader("The App classifies 3 category of objects i.e. multiclass classification over custom dataset using VGG16 Pre-traind Model")
    st.write("We have taken 100 images of each class and make a folder for further usage")
    if st.checkbox("show images"):
        st.image('model pics.jpg',use_column_width=True)
    st.write("the Output insights after model fitting are below.....")
    option = ('None', 'Model Architecture', 'prediction visualisation', 'Accuracy')
    option_dis = list(range(len(option)))
    p = st.selectbox("Choose object", option_dis, format_func=lambda x: option[x])
    if p==1:
        st.image('sparchitecture.png',use_column_width=True)
    if p==2:
        st.image('accuracy.jpg',use_column_width=True)
    if p==3:
        st.write("Without CNN using SVC-------------> 50%")
        st.write("Using VGG16 CNN model-------------> 93%")

if rad == "App":
    st.header("Upload your images below...")
    def import_and_predict(image_data, model):
        size = (150, 150)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)) / 255.

        img_reshape = img_resize[np.newaxis, ...]

        prediction = model.predict(img_reshape)

        return prediction


    file = st.file_uploader("Choose an image....", type=["jpg","png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        if st.button("Predict"):
            prediction = import_and_predict(image, model)

            if np.argmax(prediction) == 0:
                st.write("Pulsar Star")
            elif np.argmax(prediction) == 1:
                st.write("Black Hole")
            else:
                st.write("White Dwarf")

            st.text("Probability (0: Pulsar star, 1: Black Hole, 2: White dwarf")
            st.write(prediction)
