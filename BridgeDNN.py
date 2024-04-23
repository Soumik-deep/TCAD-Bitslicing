import tempfile
import os
import pathlib
import tensorflow as tf
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import numpy as np

from tensorflow import keras

import tensorflow_model_optimization as tfmot

#batch_size = 32
img_height = 224
img_width = 224

num_classes=2


# Load MNIST dataset
#mnist = keras.datasets.mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
#train_images = train_images / 255.0
#test_images = test_images / 255.0


data_dir="/home/user/soumik/workspace/BridgesData/"
data_dir = pathlib.Path(data_dir)


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

cracks = list(data_dir.glob('Positive/*'))
PIL.Image.open(str(cracks[0]))


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width))

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width))

class_names = train_ds.class_names
print(class_names)


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define the model architecture.
#model = keras.Sequential([
 # keras.layers.InputLayer(input_shape=(224, 224)),
 # keras.layers.Reshape(target_shape=(224, 224, 3)),
 #keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
 # keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #keras.layers.Flatten(),
  #keras.layers.Dense(2)
#])


model = keras.Sequential([
  #keras.layers.InputLayer(input_shape=(224, 224,1)),
  keras.layers.Reshape(target_shape=(224, 224, 3)),
 #keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu',strides=(2,2)),
  #keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(128,activation='relu'),
  keras.layers.Dense(10,activation='relu'),
  keras.layers.Dense(2)
])


# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.fit(
 # train_ds,
  #train_labels,
  #epochs=1,
  #validation_split=0.1,)

model.fit(
  normalized_train_ds,
  validation_data=normalized_val_ds,
  epochs=1
)


model.save("Non-quant(Floatingpt)_model.h5")
quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

#q_aware_model.save("Quantized_model.h5")

#train_images_subset = train_images[0:1000] # out of 60000
#train_labels_subset = train_labels[0:1000]

#q_aware_model.fit(train_images_subset,batch_size=500, epochs=1, validation_split=0.1)

q_aware_model.fit(normalized_train_ds, validation_data=normalized_val_ds,epochs=1)


q_aware_model.save("Trained_Quantized_model.h5")

#_, baseline_model_accuracy = model.evaluate(
 #   test_images, verbose=0)


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

#quantized_tflite_model.save("Int8_quantized_model.h5")

print("INT8 Quantized model weights are as follows:")
#print(quantized_tflite_model.layers[0].weights)


tflite_model_quant_file ="INT8_Quant_Model.tflite"
#tflite_model_quant_file.write_bytes(quantized_tflite_model)

tflitefile="INT8_Quant_Model.tflite"

tflite_models_dir = pathlib.Path("/home/user/soumik/workspace/NNmodels/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant_file = tflite_models_dir/"BridgeData_INT8_quant.tflite"
tflite_model_quant_file.write_bytes(quantized_tflite_model)

print("Now running the INT8 quantized model--")
_, q_aware_model_accuracy = quantized_tflite_model.evaluate(
   normalized_val_ds, verbose=0)
   

#print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)
