import cv2
from scipy import misc
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from sklearn import datasets, svm, metrics, tree, ensemble, linear_model
import glob


'''
x = np.uint8(253)
y = np.uint8(255)
grad = np.int16(x) - np.int16(y)
gradMag = np.sqrt(grad **2)
gradMag = np.uint8(min(gradMag, 255))
print gradMag
exit(0)
'''
CELL_WIDTH = 8
CELL_HEIGHT = 8

rootdir_training = 'mnist_png/training'
rootdir_testing = 'mnist_png/testing'

# need to deal with underflow (and probably overflow as well)

'''
Make sure we don't access outside of array
'''
def boundIndex(index, maxIndex):
    if index < 0:
        return 0
    if index >= maxIndex:
        return maxIndex - 1
    return index


'''
Still need to work on the overflow errors
Now I'm passing all of them in
'''
def apply_kernel(pixels):
    rows = len(pixels)
    columns = len(pixels[0])
    gradient = np.array([[0 for x in range(columns)]for y in range(rows)], dtype=np.uint16)
    angles = np.array([[0 for x in range(columns)]for y in range(rows)], dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            max = np.uint8(0)
            angle = 0
            # still need to figure out if want to use separate array for angles
            for k in range(0,3):
                # i realize I only need to check 0 or max, not both
                rIndex = boundIndex(j+1, columns)
                lIndex = boundIndex(j-1, columns)
                tIndex = boundIndex(i-1, rows)
                bIndex = boundIndex(i+1, rows)
                leftValue = np.int16(pixels[i][lIndex][k])
                rightValue = np.int16(pixels[i][rIndex][k])
                topNeighbor = np.int16(pixels[tIndex][j][k])
                botNeighbor = np.int16(pixels[bIndex][j][k])
                gradientX = rightValue - leftValue
                gradientY = botNeighbor - topNeighbor
                gradMag = np.sqrt(gradientX ** 2 + gradientY ** 2)
                #gradMag = np.uint8(min(gradMag, 255))
                if gradMag > max:
                    max=gradMag
                    angle = np.arctan2(gradientY, gradientX)
                    angle = angle * 180 / np.pi
                    angle = np.uint8(abs(angle))
            gradient[i][j] = max
            angles[i][j] = angle

    return gradient, angles

def calc_hogs(gradients, angles):
    '''
    0 -> angle between 0 and 20
    1 -> angle between 20 and 40
    2 -> angle between 40 and 60
    3 -> angle between 60 and 80
    4 -> angle between 80 and 100
    5 -> angle between 100 and 120
    6 -> angle between 120 and 140
    7 -> angle between 140 and 160
    8 -> angle between 160 and 180
    '''
    hog = np.array([0 for x in range(9)])
    rows = len(gradients)
    columns = len(gradients[0])
    for i in range(0, rows):
        for j in range(0, columns):
            # terrible code to follow -> break out into own function later
            if angles[i][j] < 20:
                hog[0] += gradients[i][j] * (1 - angles[i][j] / 20)
                hog[1] += gradients[i][j] * (angles[i][j] / 20)
            elif angles[i][j] < 40:
                hog[1] += gradients[i][j] * (1-(angles[i][j]-20) / 20)
                hog[2] += gradients[i][j] * ((angles[i][j] - 20)/20)
            elif angles[i][j] < 60:
                hog[2] += gradients[i][j] * (1-(angles[i][j] - 40)/20)
                hog[3] += gradients[i][j] * ((angles[i][j] - 40)/20)
            elif angles[i][j] < 80:
                hog[3] += gradients[i][j] * (1-(angles[i][j] - 60)/20)
                hog[4] += gradients[i][j] * ((angles[i][j] - 60)/20)
            elif angles[i][j] < 100:
                hog[4] += gradients[i][j] * (1-(angles[i][j] - 80)/20)
                hog[5] += gradients[i][j] * ((angles[i][j]-80)/20)
            elif angles[i][j] < 120:
                hog[5] += gradients[i][j] * (1-(angles[i][j] - 100)/20)
                hog[6] += gradients[i][j] * ((angles[i][j] - 100)/20)
            elif angles[i][j] < 140:
                hog[6] += gradients[i][j] * (1-(angles[i][j] - 120)/20)
                hog[7] += gradients[i][j] * ((angles[i][j] - 120)/20)
            elif angles[i][j] < 160:
                hog[7] += gradients[i][j] * (1-(angles[i][j] - 140)/20)
                hog[8] += gradients[i][j] * ((angles[i][j] - 140)/20)
            elif angles[i][j] <= 180:
                hog[8] += gradients[i][j] * (1-(angles[i][j] - 160)/20)
                hog[0] += gradients[i][j] * ((angles[i][j] - 160)/20)

    return hog

def bin_gradients(gradients, angles):
    rows = len(gradients)
    columns = len(gradients[0])
    combined_hog = np.zeros((rows/CELL_HEIGHT, columns/CELL_WIDTH, 9))
    row_num = 0
    for i in range(0, rows, CELL_HEIGHT):
        col_num = 0
        for j in range(0, columns, CELL_WIDTH):
            cell_gradients = gradients[i:i+CELL_HEIGHT, j:j+CELL_WIDTH]
            cell_angles = angles[i:i+CELL_HEIGHT, j:j+CELL_WIDTH]
            hog = calc_hogs(cell_gradients, cell_angles)
            combined_hog[row_num][col_num] = hog
            col_num += 1
        row_num += 1
    return combined_hog


def normalize_hog(hogs):
    # need to combine the 4 groups of 9 into 1 group of 36 and normalize it
    # how many arrays of size 36 will we need?
    rows = len(hogs)
    columns = len(hogs[0])
    normalized_hog = np.zeros((rows - 1, columns - 1, 36))
    # need to make it 1 less in both directions
    for i in range(0, rows - 1):
        for j in range(0, columns - 1):
            # normalize it
            cell1 = hogs[i][j]
            cell2 = hogs[i][j+1]
            cell3 = hogs[i+1][j]
            cell4 = hogs[i+1][j+1]
            magnitude = 0
            # numpy has to have a better way of doing this, but for now..
            for k in range(0, 9):
                magnitude += (cell1[k] * cell1[k] + cell2[k] * cell2[k] + cell3[k] * cell3[k] + cell4[k] * cell4[k])
            magnitude = math.sqrt(magnitude)
            cells = np.concatenate([cell1, cell2])
            cells = np.concatenate([cells, cell3])
            cells = np.concatenate([cells, cell4])
            for k in range(0, 36):
                if magnitude > 0:
                    normalized_hog[i][j][k] = cells[k] / magnitude
                else:
                    normalized_hog[i][j][k] = cells[k]
    return normalized_hog

'''
Will concatenate a 3d array into 1d array
'''
def concatenate(array):
    rows = len(array)
    columns = len(array[0])
    concatenated_array = np.array((rows*columns*len(array[0][0])))
    concat_before = False
    for i in range(0, rows):
        for j in range(0, columns):
            if (not concat_before):
                concatenated_array = array[i][j]
                concat_before = True
            else:
                concatenated_array = np.concatenate([concatenated_array, array[i][j]])
    return concatenated_array

def main():

    test_image_path = os.path.abspath("mnist_png/testing/0/3.png")

    rgb = cv2.imread(test_image_path)
    '''
    rgb = cv2.resize(rgb, (0,0), fx=1, fy=2)
    plt.imshow(rgb)
    plt.show()
    exit(0)
    '''
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    # calculate gradients for each of r, g, b
    gradient, angles = apply_kernel(rgb)



    hog = bin_gradients(gradient, angles)

    hog = normalize_hog(hog)


    hog = concatenate(hog)


    winSize = (28, 28)
    blockSize = (14,14)
    blockStride = (7,7)
    cellSize = (7,7)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = True



    hog2 = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)#,derivAperture,winSigma,
                            #histogramNormType,L2HysThreshold,gammaCorrection)


    '''

    plt.imshow(rgb)
    plt.show()

    plt.imshow(gradient)
    plt.show()
    '''

    '''
    img_paths = glob.glob(rootdir_training + "/0/*.png")
    train_labels =[]
    training = []
    count = 0
    for img in img_paths[:]:
        if count > 100:
            break
        label = img.split("/")[-2]
        print label
        data = cv2.imread(img)
        #vec = hog2.compute(data)
        gradients, angles = apply_kernel(data)
        vec = bin_gradients(gradients, angles)
        vec = normalize_hog(vec)
        vec = concatenate(vec)
        train_labels.append(label)
        training.append(vec)
        count += 1
    img_paths = glob.glob(rootdir_testing + "/0/*.png")
    test_labels =[]
    testing = []
    count = 0
    for img in img_paths[:]:
        if count > 100:
            break
        count += 1
        label = img.split("/")[-2]
        print label
        data = cv2.imread(img)
        #vec = hog2.compute(data)
        gradients, angles = apply_kernel(data)
        vec = bin_gradients(gradients, angles)
        vec = normalize_hog(vec)
        vec = concatenate(vec)
        test_labels.append(label)
        testing.append(vec)
    '''
    train_labels =[]
    training = []
    test_labels =[]
    testing = []
    for x in range(0, 10):
        count = 0
        img_paths = glob.glob(rootdir_training + "/" + str(x) + "/*.png")
        for img in img_paths[:]:
            if count > 200:
                break
            label = img.split("/")[-2]
            print label
            data = cv2.imread(img)
            #vec = hog2.compute(data)
            data = cv2.resize(data, (0, 0), fx=2, fy=4)
            gradients, angles = apply_kernel(data)
            vec = bin_gradients(gradients, angles)
            vec = normalize_hog(vec)
            vec = concatenate(vec)
            train_labels.append(label)
            training.append(vec)
            count += 1
        img_paths = glob.glob(rootdir_testing + "/" + str(x) + "/*.png")
        count = 0
        for img in img_paths[:]:
            if count > 200:
                break
            count += 1
            label = img.split("/")[-2]
            print label
            data = cv2.imread(img)
            data = cv2.resize(data, (0, 0), fx=2, fy=4)
            #vec = hog2.compute(data)
            gradients, angles = apply_kernel(data)
            vec = bin_gradients(gradients, angles)
            vec = normalize_hog(vec)
            vec = concatenate(vec)
            test_labels.append(label)
            testing.append(vec)
    #nsamples, nx, ny = np.asarray(training).shape
    #training = np.asarray(training).reshape((nsamples,nx*ny))
    #nsamples, nx, ny = np.asarray(testing).shape
    #testing = np.asarray(testing).reshape((nsamples,nx*ny))

    clf = svm.SVC()
    clf.fit(training, train_labels)
    predict = clf.predict(testing)
    print metrics.accuracy_score(predict, test_labels)


if __name__ == "main":
    main()