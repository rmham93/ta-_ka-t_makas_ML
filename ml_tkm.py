# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


# %%
training_data_gen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# %%
test_data_gen = ImageDataGenerator(rescale =1./255)


# %%
train_generator = training_data_gen.flow_from_directory(
    'C:/Users/IDU/OneDrive - GTÜ/Documents/ML/data',
    target_size =(150, 150),
    color_mode="rgb",
    class_mode = 'categorical')
#batch_size=32,


# %%
test_generator = test_data_gen.flow_from_directory(
    'test_set',
    target_size = (150, 150),
    class_mode = 'categorical'
) 


# %%
model = tf.keras.models.Sequential([
    #we prepare input layer with shape (150,150) and 3 bytes colors
    #150x150lik ve 3 bayt renk kodları olan input katmanı, ilk convolution katmanı
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape= (150,150,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Second layer with 64 nodes
    #ikinci convolution katmanı
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # max pooling ile 2x2lik parçalara bölüyoruz fotoğrafı, ve max değeri alıp yeni layera yazıyoruz 
    #(4 elementin en büyüğünü) 3. convolution layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The 4. convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Artık küçültmemiz gerekiyor output layer için
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    #Bu 512 node (nöron), benim outputumu 3 kategoriye nasıl ayırdığımı öğretmek için.
    #Tüm detayları ile öğrenebilmesi için
    tf.keras.layers.Dense(512,activation= 'relu'),
    tf.keras.layers.Dense(3,activation= 'softmax')  
])


# %%
model.summary()


# %%
model.compile( loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
history = model.fit_generator(train_generator, epochs=25, validation_data = test_generator, verbose = 1)

model.save("rps.h5")


# %%



# %%



# %%



