from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import TensorBoard


import os
from os import environ
import time

###############################################################################
# Set up working directories for data, model and logs.
###############################################################################
model_filename = "flowers_vgg19_4e.h5"

# writing the train model and getting input data
if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
else:
    output_model_folder = "model"
    output_model_path = os.path.join("model", model_filename)

os.makedirs(output_model_folder, exist_ok=True)

# writing metrics
if environ.get('JOB_STATE_DIR') is not None:
    tb_directory = os.path.join(os.environ["JOB_STATE_DIR"], "logs", "tb", "test")
else:
    tb_directory = os.path.join("logs", "tb", "test")

os.makedirs(tb_directory, exist_ok=True)
tensorboard = TensorBoard(log_dir=tb_directory)

# Training and Test data
if environ.get('DATA_DIR') is not None:
    train_data_dir = os.path.join(os.environ["DATA_DIR"], "flw_test", "train")
    validation_data_dir = os.path.join(os.environ["DATA_DIR"], "flw_test", "test")
else:
    train_data_dir = os.path.join("/safe/dev/ngp/projects/face-incep/data", "flw_test", "train")
    validation_data_dir = os.path.join("/safe/dev/ngp/projects/face-incep/data", "flw_test", "test")

###############################################################################


img_width, img_height = 256, 256
nb_train_samples = 160
nb_validation_samples = 20
batch_size = 10
epochs = 4

model = applications.VGG19(weights="imagenet",
                           include_top=False,
                           input_shape=(img_width, img_height, 3))

# Freeze the layers which you don't want to train.
for layer in model.layers[:5]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

print("Starting at: " + str(time.time()))

# Train the model
model_final.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=2,
    callbacks=[tensorboard])
    #callbacks=[checkpoint, early])

print("Ended at: " + str(time.time()))

# Save the model
model_final.save(output_model_path)
print("Saved at: " + str(time.time()))
