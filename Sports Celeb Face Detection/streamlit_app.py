import streamlit as st
import joblib 
import json
import cv2
import pywt
import numpy as np

model = joblib.load( 'saved_model.pkl') 

with open("class_dictionary.json","r") as f:
    celeb_dic =   json.loads(f.read() )
    

#Loading up the Regression model we created

#Caching the model for faster loading
# @st.cache(model)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


def get_cropped_img(img ):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    
    for (fx,fy,fw,fh) in faces:
        crop_color_face = img[fy:fy+fh, fx:fx+fw]
        
        return crop_color_face 
    return None


def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    
    
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    
    #Process Coefficients
    coeffs_H=list(coeffs)  
    
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    # plt.imshow(imArray_H    , cmap='gray')

    return imArray_H


####################

def get_combined_img(img):

    img_har = w2d(img,'db1',5)

    scalled_raw_img = cv2.resize(img, (32, 32))
    scalled_har_img = cv2.resize(img_har, (32, 32))

    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1)  ,   scalled_har_img.reshape(32*32,1)))
    
    return combined_img 



def get_tested_img(img):
    cropped_img = get_cropped_img(img)
    
    if cropped_img is None:
        return
    
    return  get_combined_img(img)


#################


def predict_img(img):    
    
    x_test = get_tested_img(img)

    if x_test is not None:
        x_test = x_test.reshape(  1, -1 ).astype(float)
        return  celeb_dic[ str(model.predict(x_test)[0])]
    else:
        print('Face is not visible !!!')
        
        
    
img = cv2.imread('./datasets/virat_kohli/Virat_Kohli_AP.jpg')



st.title('Diamond Price Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter the characteristics of the diamond:')



carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)

cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])

clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)

table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)

x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)

y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)

z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)







if st.button('Predict Price'):
    result = predict_img(img)
    st.success(f'The predicted price of the diamond is ${price[0]:.2f} USD')