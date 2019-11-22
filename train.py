import matplotlib.pyplot as plt
import numpy as np
import os
import config
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report


def plot_training_history(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arrange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arrange(0,N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title('Training Loss and accuracy')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)



trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.TEST])

totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

valAug = ImageDataGenerator()


mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

#initializing the training generator
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode='categorical',
    target_size=(224,224),
    color_mode='rgb',
    shuffle=True,
    batch_size=config.BATCH_SIZE
)

valGen=valAug.flow_from_directory(
    valPath,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    target_size=(224,224),
    batch_size=config.BATCH_SIZE
)

testGen = valAug.flow_from_directory(
    testPath,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    target_size=(224,224),
    batch_size=config.BATCH_SIZE
)

#loading VGG16 model
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

#constructing the head of the model
headModel = baseModel.output
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation='softmax')(headModel)

#placing the head on top of base model
model = Model(inputs = baseModel.inputs, outputs = headModel)

for layer in model.layers:
    layer.trainable = False

#compiling the model
print('Compiling the model..........')
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('Training head........')
H= model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    epochs=50
)

print('evaluating after fine-tuning network head')
testGen.reset()
predIdxs = model.predict_generator(
    testGen,
    steps=(totalTest // config.BATCH_SIZE)+1
)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))
plot_training_history(H, 50, config.WARMUP_PLOT_PATH)

#unfreezing the final set of base model
trainGen.reset()
valGen.reset()

for layer in baseModel.layers[15:]:
    layer.trainable = True

# loop over the layers in the model and show which ones are trainable or not
for layer in baseModel.layers:
	print("{}: {}".format(layer, layer.trainable))

#retrainign the final set of laeyrs
print('Re-compiling the model')
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain//config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal//config.BATCH_SIZE,
    epochs=20
)

#make predictions on new model
print('Evaluating the fine-tuned model')
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))
plot_training_history(H, 20, config.UNFROZEN_PLOT_PATH)

#serailizing the model to disk
print('Serializing the network')
model.save(config.MODEL_PATH)