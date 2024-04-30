import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image
#import matplotlib.cm as cm

st.set_page_config(layout='wide')

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()

st.title(""" X-Ray Classification for Pneumonia Detection """)
st.info(
"""
Final Project 2 - By Vedanth Aggarwal | Built using tensorflow covolutional neural networks\n
**Remember that the detection model is a tool and not a substitute for professional medical advice. If you have ongoing health concerns, it's best to consult with a medical professional to ensure accurate diagnosis and appropriate care**
"""
)
st.write('<hr style="height:3px;border:none;color:#333;background-color:#333;" />', unsafe_allow_html=True)

temp = st.file_uploader("Upload ( or drag & drop ) X-Ray Image",label_visibility='collapsed')

col_a,col_b = st.columns([0.5,0.5])

buffer = temp
temp_file = NamedTemporaryFile(delete=False)

if buffer : # buffer
  try:
    temp_file.write(buffer.getvalue())
    with col_a:
      st.write(image.load_img(temp_file.name))
    with col_b:
      progress_bar = st.progress(0)

    for i in range(0,35):
        time.sleep(0.05)
        progress_bar.progress(i,'Loading Image ....')

    chosen_img = image.load_img(temp_file.name, target_size=(400, 400),color_mode='grayscale')

    for i in range(35,70):
          time.sleep(0.05)
          progress_bar.progress(i,'Processing Image ....')
      # Preprocessing the image
    pp_chosen_img = image.img_to_array(chosen_img)
    #colormap = cm.get_cmap('jet')
    #chosen_colormap_img = colormap(pp_chosen_img.squeeze())[:, :, :3]  # Discard the alpha channel
    pp_chosen_img = pp_chosen_img/255
    pp_chosen_img = np.expand_dims(pp_chosen_img, axis=0)

    for i in range(70,95):
          time.sleep(0.05)
          progress_bar.progress(i,'Making Prediction ....')

    #predict
    image_preds= cnn.predict(pp_chosen_img)

    col1,col2 = st.columns(2)

    for i in range(95,101):
          time.sleep(0.1)
          progress_bar.progress(i,'Process succesful ....')
    pneu_msg_1 = '''
  **Dangerous threat of disease! Consider taking the following steps:**
  1. Consult a Medical Professional
  2. Get a Clinical Examination
  3. Follow Treatment Recommendations - If pneumonia is confirmed, your healthcare provider will recommend a tailored treatment plan. This may include prescription medications such as antibiotics or antiviral drugs, as well as rest, hydration, and other supportive measures. Follow your doctor's advice closely to aid in your recovery.
  4. Monitor and Rest - Pneumonia can vary in severity, so it's important to monitor your symptoms and follow medical guidance. Get plenty of rest to allow your body to recover. If your symptoms worsen, such as increased difficulty breathing or persistent high fever, seek immediate medical attention.
      '''

    pneu_msg_2 = '''
  **Prediction is not definitive, Further confirmation is required!!! Consider taking the following steps:**
  1. Consult a Healthcare Professionalrofessional
  2. Monitor Your Symptoms - Pay close attention to any symptoms you may be experiencing, such as coughing, difficulty breathing, fever, or chest discomfort. Keep a record of your symptoms and any changes you notice. If your symptoms worsen or persist, seek medical attention promptly.
  3. Practice Self-Care
  4. Consider Getting Tested
  5. Follow Medical Recommendations
  6. Stay Informed
      '''

    normal_msg_1 = '''
    **Prediction is not definitive, Further confirmation required! Consider following precuations:**
  1. Monitor Your Health and pay attention to any changes in your symptoms
  2. Practice Preventive Measures - Maintain good hygiene practices, such as washing your hands regularly, covering your mouth and nose when sneezing or coughing, and avoiding close contact with individuals who are sick. These measures help prevent the spread of infections, including respiratory illnesses.
  3. Stay Active and Healthy - Engage in regular physical activity, maintain a balanced diet, and get sufficient sleep.
  4. Stay Informedy
  5. Seek Medical Advice if Symptoms Persist
  6. Follow Healthcare Provider's Advice
      '''

    normal_msg_2 = '''
    **You are safe, but maintain the following:**
  1. Consult a Healthcare Professional
  2. Follow Medical Advice
  3. Monitor Your Healthevelopments
  4. Practice Precautions
  5. Stay Hydrated and Rest
  6. Avoid Self-Medication
  7. Stay Informed
  8. Follow Up with Medical Professionals
  '''


    with col1:
      #st.write(image_preds)
      if image_preds >= 0.5:  # Normal or Pneumonia
          percentage = image_preds[0][0]
          if percentage > 0.9:
              st.error('**Picture is {:.2f}% case of Pneumonia!!!\n{}**'.format(percentage * 100, pneu_msg_1))
          else:
              st.warning('**Picture is {:.2f}% case of Pneumonia!\n{}**'.format(percentage * 100, pneu_msg_2))
      else:
          percentage = 1 - image_preds[0][0]
          if percentage > 0.9:
              st.success('**Picture is {:.2f}% NOT a case of Pneumonia!!!{}**'.format(percentage * 100, normal_msg_1))
              st.balloons()
          else:
              st.warning('**Picture is {:.2f}% NOT a case of Pneumonia!{}**'.format(percentage * 100, normal_msg_2))

    with col2:
        image = Image.open(temp)
        st.image(image,use_column_width=True)



  except Exception as e:
    st.error(f'Pls upload a valid file format first!!! {e}')