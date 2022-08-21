import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Image Data generator with image augmentation for more robust in generalized model
# Use random rotations, width and height shifts and rescaling
# Note: using the FER 2013 dataset this may not be necessary for the current training images, but this will benefit for new ones later on
train_data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1.0/255
)

validation_data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1.0/255
)

# Gather and preprocess all test images (resize, define batchsize, grayscale)
train_generator = train_data_gen.flow_from_directory(
        directory='data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Gather and preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        directory='data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Build model structure
fer_model = Sequential()

# input layer
fer_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

# hidden layers
fer_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
fer_model.add(MaxPooling2D(pool_size=(2, 2)))
fer_model.add(Dropout(0.25))

fer_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
fer_model.add(MaxPooling2D(pool_size=(2, 2)))
fer_model.add(Dropout(0.25))

fer_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
fer_model.add(MaxPooling2D(pool_size=(2, 2)))
fer_model.add(Dropout(0.25))

# fully connected layer
fer_model.add(Flatten())
fer_model.add(Dense(1024, activation='relu'))
fer_model.add(Dropout(0.5))

# output layer
fer_model.add(Dense(7, activation='softmax')) # 7 fer classes

# disable GPU for training
cv2.ocl.setUseOpenCL(False)

# define optimizer and error metric
fer_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
# steps per epoch -> total number of training data points divided by batch size (defined in train generator)
fer_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in JSON file
model_json = fer_model.to_json()
with open("fer_model.json", "w") as json_file:
    json_file.write(model_json)

# creates HDF5 file
fer_model.save('fer_model.h5')

# save trained model weight in .h5 file
fer_model.save_weights('fer_model_weights.h5')