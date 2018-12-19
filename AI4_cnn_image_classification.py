
import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pydot
import graphviz
from keras.utils import plot_model
import datetime

# This is for quickly being able to change the values of the diffrent training parameters

nb_epoch = 100
nb_train_samples = 6920 #1100
nb_validation_samples = 980 #200

definedLR = 0.5
definedOptimizer = optimizers.SGD(lr=definedLR) # https://keras.io/optimizers/
                
img_width, img_height = 28, 28
filter_size = 5
stride = 1
pooling = 2
batch_size_train = 32
batch_size_val = 16

train_data_dir =        '/home/agrot12/Documents/3/AI4/Classification/datasets/mnist/train/' #mnist_reduced
validation_data_dir =   '/home/agrot12/Documents/3/AI4/Classification/datasets/mnist/test/'


amount_of_seeds = 4

variableChanged = "AmountOfImages"

for x1 in range (1,5):
        # LEARNING RATE
        if variableChanged == "LearningRate":
                definedLR = 0.2*x1 #10**(-x1)
                variableValue = str(definedLR)

        # SOLVER
        if variableChanged == "Solver":
                if x1 == 1:
                        definedOptimizer = optimizers.SGD(lr=definedLR)
                        variableValue = 'SGD'
                if x1 == 2:
                        definedOptimizer = optimizers.Adagrad(lr=definedLR)
                        variableValue = 'Adagrad'
                if x1 == 3:
                        definedOptimizer = optimizers.SGD(lr=definedLR)
                        variableValue = 'Adadelta'
                if x1 == 4:
                        definedOptimizer = optimizers.SGD(lr=definedLR)
                        variableValue = 'Adam'

        # DIFFERENT AMOUNT OF IMAGES
        if variableChanged == "AmountOfImages":
                nb_train_samples = 1700*x1 #1100
                nb_validation_samples = 200*x1 #200
                variableValue = "train" + str(nb_train_samples) + "val" +str(nb_validation_samples)


        print("variableValue: " + variableValue)
        for x in range(1, amount_of_seeds+1):

                print("variableValue: " + variableValue + " and seed: " + str(x))
                np.random.seed(x)

                trainingName = "results/"+variableChanged+"/"+variableValue+"_seed" + str(x)
                        
                # used to rescale the pixel values from [0, 255] to [0, 1] interval
                datagen = ImageDataGenerator(rescale=1./255)

                # automagically retrieve images and their classes for train and validation sets
                train_generator = datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size_train,
                        class_mode='categorical',
                        color_mode="grayscale")

                validation_generator = datagen.flow_from_directory(
                        validation_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size_val,
                        class_mode='categorical',
                        color_mode="grayscale")

                """
                This is the simple keras CNN model, CNN models often don't need more than 3 layers when working with small datasets. The focus here is to set alot of 
                filters on the layers, so the model have the possibility too find alot of patterns for the diffrent kinds of dogs and cats.
                """

                print("\n ------------------------------------------------------------------")
                print(" -------------------------- Sequential() --------------------------")
                print(" ------------------------------------------------------------------\n")
                model = Sequential()
                model.add(Conv2D(       filters=20,
                                        kernel_size=[5 , 5],
                                        strides=[1,1],
                                        padding = 'valid',
                                        input_shape=(img_width, img_height,1), 
                                        activation='relu')) 

                model.add(MaxPooling2D(pool_size=(2,2))) 

                model.add(Conv2D(       filters=50,
                                        kernel_size=[5 , 5],
                                        strides=[1,1],
                                        #padding = 'same',
                                        input_shape=(img_width, img_height,1), 
                                        activation='relu')) 

                model.add(MaxPooling2D(pool_size=(2,2))) 
                model.add(Flatten()) #Ready for Dense
                model.add(Dense(500, activation='relu'))
                model.add(Dense(10, activation='softmax'))

                print("\n ------------------------------------------------------------------")
                print(str(model.summary()))
                print("\n ------------------------------------------------------------------")



                model.compile(loss='mean_squared_error', optimizer=definedOptimizer, metrics=['accuracy'])



                #plot_model(model, to_file='model.png')
                #model.save('my_model.h5')

                print("\n ------------------------------------------------------------------")
                print(" -------------------------- fit_generator -------------------------")
                print(" ------------------------------------------------------------------\n")

                history = model.fit_generator(
                        train_generator,
                        samples_per_epoch=nb_train_samples,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_validation_samples)

                #model.save(trainingName+".h5")



                # Plot training & validation loss values
                with PdfPages(trainingName+".pdf") as pdf:

                        plt.plot(history.history['acc'])
                        plt.plot(history.history['val_acc'])
                        plt.title('Model accuracy')
                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Test'], loc='upper left')
                        #plt.savefig("AI_1_acc.pdf")
                        pdf.savefig()
                        plt.clf()
                
                        plt.plot(history.history['loss'])
                        plt.plot(history.history['val_loss'])
                        plt.title('Model loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Test'], loc='upper left')

                        pdf.savefig()
                        plt.clf()


                #acc val_acc loss val_loss time??
                with open(trainingName+".txt", "w") as file:
                        for i in range(nb_epoch):
                                file.write("%i\t%f\t%f\t%f\t%f\n" % (i,history.history['acc'][i],history.history['val_acc'][i],history.history['loss'][i],history.history['val_loss'][i]))

                print("\n ------------------------------------------------------------------")
                print(" ----------------------- evaluate_generator -----------------------")
                print(" ------------------------------------------------------------------\n")
                model_evaluate = model.evaluate_generator(validation_generator, nb_validation_samples)
                print(model_evaluate)


                '''

                Epoch 1/10
                1000/1000 [==============================] - 44s 44ms/step - loss: 0.3165 - acc: 0.8986 - val_loss: 0.2976 - val_acc: 0.8995
                batch/batches                              - time

                '''


                print("\n ------------------------------------------------------------------")
                print(" ----------------------- script finished at -----------------------")
                print(" ------------------------------------------------------------------\n")

                print(datetime.datetime.now())



        with open(trainingName+"_info.txt", "w") as file:
                file.write("\nTime \t\t\t\t\t" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                file.write("\n")
                file.write("\nnb_epoch \t\t\t\t" + str(nb_epoch))
                file.write("\nnb_train_samples \t\t" + str(nb_train_samples))
                file.write("\nnb_validation_samples \t" + str(nb_validation_samples))
                file.write("\n")
                file.write("\ndefinedLR \t\t\t\t" + str(definedLR))
                file.write("\ndefinedOptimizer \t\t" + str(definedOptimizer))
                file.write("\n")
                file.write("\nimg_width \t\t\t\t" + str(img_width))
                file.write("\nimg_height \t\t\t\t" + str(img_height))
                file.write("\nfilter_size \t\t\t" + str(filter_size))
                file.write("\nstride \t\t\t\t\t" + str(stride))
                file.write("\npooling \t\t\t\t" + str(pooling))
                file.write("\nbatch_size_train \t\t" + str(batch_size_train))
                file.write("\nbatch_size_val \t\t\t" + str(batch_size_val))
                file.write("\n")
                file.write("\ntrain_data_dir \t\t\t" + str(train_data_dir))
                file.write("\nvalidation_data_dir \t" + str(validation_data_dir))
                file.write("\n")
                file.write("\n")
                file.write("\nmodel.summary\n")
                file.write("\n")
                model.summary(print_fn=lambda x: file.write(x + '\n'))