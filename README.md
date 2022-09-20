# End To End Dead Star Image Classification using Transfer Learning Project

<img src="[https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png]" width="600" height="300">





Title: Dead Star Image Classification.

Aim: Space Enthusiasts or students can classify between Black Hole,Pulsar Star,White Dwarfs and get their details as well using this App.

Project Description:


1. collecetd 100 images of each classes from the web using bing-image-downloader.
2. After collecting the images does preprocessing over them ie. resize, aligned their shapes etc.
3. added some padding to improve the computation power of the model.
4. After Preprocessing over the images seperate those images in Data Generator Format i.e. folder/subfolder/images--subfolder/images.
5. imported vgg16, mobilenet, inception_v3 models and feeded the data into those. the whole description of the models is given inside the ppt.
6. freezed the last layer of all models and added our own dense layers for performing our particular task. i.e. 3 class Classification.
7. Used Image Augmentation to increase the dataset size by adding more examples to learn by including the parameters as zoom, slide, rotation, shear&flip etc.
8. Tried over various epochs and improved the model performance.
9. used GridSearchCV() for fine tuning the model and the model that has given satisfactory and performed far better is VGG16 & Resnet50.
10. After selecting the model. i've created a webapp using streamlit and deployed it over streamlit share.


![image](https://user-images.githubusercontent.com/73512374/179956826-9093249f-2114-4f78-a9f2-b22ed41dfb20.png)


Tools Used:
* Pycharm
* Google colab
* AWS
* streamlit
* github


Libraries used:
* Tensorflow
* Pandas
* Numpy
* Matplotlib
* Seaborn

Algorithms used:
* Vgg16 CNN model
* Inception_v3
* MobileNet
* ResNet50
* SVC

Model Performance:
* Vgg16 : 
* ResNet50 : 
* MobileNet_v3 : 
* SVC : 

Here is my web app link :
https://share.streamlit.io/kolhesamiksha/spaceobjectclassifier/main/main.py
