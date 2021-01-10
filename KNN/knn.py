import numpy as np
import matplotlib.pyplot as plt
import time
from mnist import MNIST
from sklearn.neighbors import KDTree

#show_digit displays an image rendering of a 28x28 numpy array
def show_digit(img):
    plt.axis('off')
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

#vis_image displays a 28x28 image and its label given the image's index, a dataset, and a set of label
def vis_image(index, dataset, labels):
    show_digit(dataset[index,])
    label = labels[index]
    print('Label ' + str(label))
    return


#calcDistance computes and returns the L2 Euclidean distance between two images, p1 and p2
#Omit the square root as it doesn't change the behavior of KNN
def calc_distance(p1, p2):
    dist = np.sum(np.square(p1 - p2))
    return dist

#find_NN returns the closest image in the training set to the input image
def find_NN(img, train_data, train_labels):
    distances = [calc_distance(img, train_data[i,]) for i in range(len(train_labels))]
    nearestNeighbor = np.argmin(distances)
    return nearestNeighbor

#NN_classification returns the classification of a given input image according to our KNN model
def NN_classification(img, kd_tree, train_labels):
    #classification = train_labels[find_NN(img, train_data, train_labels)]
    classification = train_labels[np.squeeze(kd_tree.query(img, k=1, return_distance=False))]
    return classification

#view_menu visualizes an image from the test set given its index from the user
def view_menu(x_test, y_test):
    img_index = 10000
    while img_index > 9999 or img_index < 0:
        img_index = int(input('Enter the index of the image you want to view (0-9999): '))
    vis_image(img_index, x_test, y_test)


def classify_test_menu(kd_tree, y_train, x_test):
    img_index = 10000
    while img_index > 9999 or img_index < 0:
        img_index = int(input('Enter the index of the image you want to classify (0-9999): '))
    classification = NN_classification(x_test[img_index].reshape(1, -1), kd_tree, y_train)
    print("Test image " + str(img_index) + " is classified as " + str(classification))

def calc_accuracy(kd_tree, y_train, x_test, y_test):
    test_predictions = NN_classification(x_test, kd_tree, y_train)
    num_correct = np.sum(np.equal(test_predictions, y_test))
    accuracy = (float(num_correct) / len(y_test)) * 100
    print("The model accuracy over the test set is " + str(accuracy) + "%")

def open_menu():
    #load the train and test data
    mnist = MNIST('.\\MNIST dataset\\')
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)

    #create the KD tree structure from the training data
    kd_tree = KDTree(x_train)


    #User Menu - options: view test image(int in range), classify test image(int in rage), Test accuracy, classify custom image
    menu_option = 'c'
    while menu_option != 'q':
        menuMessage = ('KNN Menu:\n'
                       '(v) View an image from the test set\n'
                       '(t) Classify an image form the test set\n'
                       #'(c) Classify a custom image\n'
                       '(a) Calculate the accuracy of the classifier model\n'
                       '(q) Close the KNN menu\n')
        print(menuMessage)
        menu_option = input("Select an option: ")
        if menu_option == 'v':
            view_menu(x_test, y_test)
        elif menu_option == 't':
            classify_test_menu(kd_tree, y_train, x_test)
        #elif menu_option == 'c':
        #    print("")
        elif menu_option == 'a':
            calc_accuracy(kd_tree, y_train, x_test, y_test)
