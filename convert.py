from tensorflow.keras import applications
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, concatenate
import tensorflow as tf


input_shape =[100,100]
# Load model 
def model():
  inputs = Input(shape=(input_shape[0],input_shape[1], 3))
  base_model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_tensor = inputs)
  base_model_output = base_model.output

  for layer in base_model.layers:
    layer.trainable = False
  
  age = Flatten()(base_model_output)
  age = Dense(4096,activation="relu")(age)
  age = Dropout(0.5)(age)
  age = Dense(4096,activation="relu")(age)
  age_output = Dense(4, activation='softmax')(age)
  age_model = Model(inputs=inputs, outputs=age_output)
  age_model.load_weights("models/triple_model/age_model.h5")

  gender = Flatten()(base_model_output)
  gender = Dense(4096,activation="relu")(gender)
  gender = Dropout(0.5)(gender)
  gender = Dense(2048,activation="relu")(gender)
  gender_output = Dense(2, activation='sigmoid')(gender)
  gender_model = Model(inputs=inputs, outputs=gender_output)
  gender_model.load_weights("models/triple_model/gender_model.h5")

  race = Flatten()(base_model_output)
  race = Dense(4096,activation="relu")(race)
  race = Dropout(0.5)(race)
  race = Dense(2048,activation="relu")(race)
  race_output = Dense(5, activation='softmax')(race)
  race_model = Model(inputs=inputs, outputs=race_output)
  race_model.load_weights("models/triple_model/race_model.h5")

  for layer in age_model.layers:
    layer.trainable = False
  for layer in gender_model.layers:
    layer.trainable = False
  for layer in race_model.layers:
    layer.trainable = False

  combined_model_input = Input(shape=(input_shape[0],input_shape[1], 3))
  age_predict = age_model(combined_model_input)
  gender_predict = gender_model(combined_model_input)
  race_predict = race_model(combined_model_input)
  final = concatenate([age_predict,gender_predict,race_predict])
  final = Dense(1024, activation = "relu")(final)
  final = Dropout(0.3)(final)
  final_output = Dense(4, activation = "softmax")(final)
  final = Model(inputs=combined_model_input, outputs=final_output)

  return final

model = model()
model.load_weights("models/triple_model/model.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

print("model converted")

# Save the model.
with open('model_combined.tflite', 'wb') as f:
  f.write(tflite_model)