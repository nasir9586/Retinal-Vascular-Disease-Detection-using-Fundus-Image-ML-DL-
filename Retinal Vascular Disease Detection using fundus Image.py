#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import glorot_uniform,he_uniform
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

# File paths
train_dir = "C:/Users/Nasir/Downloads/OUR/OUR/TRAIN"
val_dir = "C:/Users/Nasir/Downloads/OUR/OUR/VALIDATION"
test_dir = "C:/Users/Nasir/Downloads/OUR/OUR/TEST"


# Load and preprocess data (same as before)
def load_data(directory):
    images = []
    labels = []
    label_dict = {
        'nodr': 0,
        'mild_npdr': 1,
        'moderate_npdr': 2,
        'severe_npdr': 3,
        'pdr': 4
    }
    
    for label_name, label_index in label_dict.items():
        label_dir = os.path.join(directory, label_name)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = image.load_img(img_path, target_size=(299, 299))  # Resize to InceptionV3's required size
            img_array = image.img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(label_index)
    
    return np.array(images), np.array(labels)


# In[ ]:


# Load the datasets
x_train, y_train = load_data(train_dir)
x_val, y_val = load_data(val_dir)
x_test, y_test = load_data(test_dir)


# In[ ]:


# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))  # Total number of classes
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)


# In[ ]:


input_dim = x_train.shape
input_dim


# In[ ]:


y_train.shape,x_val.shape


# In[ ]:


x_test.shape,y_val.shape,y_test.shape


# In[ ]:


def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model initially

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())  # Global average pooling to reduce dimensions
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu', kernel_initializer=he_uniform()))
    model.add(BatchNormalization())  # Batch normalization
    
    model.add(Dense(512, activation='relu', kernel_initializer=he_uniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Dropout layer with rate 0.2 (alternate)
    
    model.add(Dense(256, activation='relu', kernel_initializer=he_uniform()))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))  # Final output layer for classification
    
    return model


# In[ ]:


model = create_model()


# In[ ]:


#optimizer = Adam(learning_rate=0.001)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Data augmentation setup (put this before model.fit)
train_datagen = ImageDataGenerator(
    
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)callbacks=[early_stopping],

# Fit the model with fine-tuning
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=128),  # Corrected flow syntax
    validation_data=(x_val, y_val),
    epochs=20,
    verbose=1
)


# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title("Train Vs Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xlim(0, 50)
plt.xticks(range(0, 51, 2))
plt.ylim(0, 70)
plt.yticks(range(0, 71, 4))
plt.legend(['Val_loss','loss'])
plt.show()


# In[ ]:


import gradio as gr
from tensorflow.keras.preprocessing import image

model = create_model()
model.load_weights('"C:/Users/Nasir/Downloads/OUR/51_r1.jpg".h5')

def predict_image(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
    predicted_class = class_labels[class_idx]
    return f'Predicted class: {predicted_class}'

iface = gr.Interface(
    fn=predict_image, 
    inputs=gr.inputs.Image(shape=(299, 299)), 
    outputs="text", 
    title="Diabetic Retinopathy Detection"
)

iface.launch()

