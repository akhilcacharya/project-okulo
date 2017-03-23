import cv2
from scipy import misc
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from sklearn import datasets, svm, metrics, tree, ensemble, linear_model
import glob
import HoG


CELL_WIDTH = 8
CELL_HEIGHT = 8

rootdir_training = '/media/sf_Video_Dataset'
rootdir_testing = 'media/sf_Video_Dataset'

original_frame1 = cv2.imread("tennis492.jpg")
original_frame2 = cv2.imread("tennis493.jpg")

'''
Make sure we don't access outside of array
'''
def boundIndex(index, maxIndex):
    if index < 0:
        return 0
    if index >= maxIndex:
        return maxIndex - 1
    return index


def calcMatrixValues(curr_row, curr_col, prev_frame, curr_frame):
    lucas_kanade_matrix = np.zeros((2,2),dtype=np.float64)
    lucas_kanade_values = np.zeros((2,1),dtype=np.float64)
    lucas_kanade_optical_flow = np.zeros((2,1),dtype=np.float64)
    rows = len(prev_frame)
    columns = len(prev_frame[0])
    for i in range(curr_row-1, curr_row+2):
        for j in range(curr_col-1, curr_col+2):
            # dx of curr_frame
            rIndex = boundIndex(j + 1, columns)
            lIndex = boundIndex(j - 1, columns)
            tIndex = boundIndex(i - 1, rows)
            bIndex = boundIndex(i + 1, rows)
            dx = np.float64(curr_frame[i][rIndex] - curr_frame[i][lIndex])
            dy = np.float64(curr_frame[bIndex][j] - curr_frame[tIndex][j])
            dt = np.float64(curr_frame[i][j] - prev_frame[i][j])

            lucas_kanade_matrix[0][0] = dx * dx + lucas_kanade_matrix[0][0]
            lucas_kanade_matrix[0][1] = dx * dy + lucas_kanade_matrix[0][1]
            lucas_kanade_matrix[1][0] = dx * dy + lucas_kanade_matrix[1][0]
            lucas_kanade_matrix[1][1] = dy * dy + lucas_kanade_matrix[1][1]

            lucas_kanade_values[0][0] = dx * dt + lucas_kanade_values[0][0]
            lucas_kanade_values[1][0] = dy * dt + lucas_kanade_values[1][0]
    # now invert lucas_kanade_matrix
    try:
        #lucas_kanade_matrix = np.linalg.inv(lucas_kanade_matrix)
        lucas_kanade_values[0][0] = -1 * lucas_kanade_values[0][0]
        lucas_kanade_values[1][0] = -1 * lucas_kanade_values[1][0]
        a = lucas_kanade_matrix[0][0]
        b = lucas_kanade_matrix[0][1]
        c = lucas_kanade_matrix[1][0]
        d = lucas_kanade_matrix[1][1]
        det = a*d - b * c
        if (det == 0):
            lucas_kanade_optical_flow[0][0] = 0
            lucas_kanade_optical_flow[1][0] = 0
        else:
            lucas_kanade_matrix[0][0] = d / det
            lucas_kanade_matrix[1][1] = a / det
            lucas_kanade_matrix[0][1] = -1 * b / det
            lucas_kanade_matrix[1][0] = -1 * c / det
            lucas_kanade_optical_flow = np.dot(lucas_kanade_matrix, lucas_kanade_values)

    except np.linalg.linalg.LinAlgError:
        # I guess just set the flow equal to 0 if the matrix isn't invertible
        lucas_kanade_optical_flow[0][0] = 0
        lucas_kanade_optical_flow[1][0] = 0
    return lucas_kanade_optical_flow


def calcOpticalFlow(prev_frame, curr_frame):
    # u is x component, v is y component
    # lucas kanade takes a 5x5 patch around each point
    # I'll leave aside the edges for now
    rows = len(prev_frame)
    cols = len(prev_frame[0])
    # create matrix for storing flow u and v
    flows = np.zeros((rows,cols,2),dtype=np.float64)
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            # get the 25 points and calculate u and v for them
            curr_flow = calcMatrixValues(i,j,prev_frame, curr_frame)

            #angle = np.uint8(abs(angle))
            # opencv returns calculated new positions
            flows[i][j][0] = curr_flow[0][0]
            flows[i][j][1] = curr_flow[1][0]
    # now that we're done, return flows
    return flows


img_paths = glob.glob(rootdir_training)
train_labels =[]
test_labels = []
training = []
testing = []
count = 0
print "Made it here"
actions = ["walking", "jogging", "running"]
for x in range(len(actions)):
    count = 0
    print "Count " + str(count)
    img_paths = glob.glob(rootdir_training + "/" + str(actions[x]) + "/*.avi")
    for img in img_paths[:]:
        label = actions[x]
        descriptor = []
        cap = cv2.VideoCapture(img)
        num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # assume there is at least 1 frame
        has_more, old_frame = cap.read()
        curr_frame = 1
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        while True:
            has_more, new_frame = cap.read()
            curr_frame += 1
            # now perform the optical flow calculations
            # convert to grayscale
            if has_more:
                print "still got it"
            else:
                print "all out"
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            # now need to compute optical flow -> for now just focus on calculating u and v, and don't worry about gradients in x and y direction
            frame3 = np.zeros((len(old_frame), len(old_frame[0])), dtype=np.float64)
            frame4 = np.zeros((len(old_frame), len(old_frame[0])), dtype=np.float64)

            for i in range(0, len(old_frame)):
                for j in range(0, len(old_frame[0])):
                    frame3[i][j] = np.float64(old_frame[i][j] / 255.)
                    frame4[i][j] = np.float64(new_frame[i][j] / 255.)

            flow = calcOpticalFlow(frame3, frame4)
            # now perform the binning and everything else...
            #get magnitude and angles
            gradFlow = np.zeros((len(flow), len(flow[0])), dtype=np.uint64)
            angles = np.zeros((len(flow), len(flow[0])), dtype=np.uint64)
            # get magnitude and direction
            for i in range(1, len(flow) - 1):
                for j in range(1, len(flow[0]) - 1):
                    mag = flow[i][j][0] ** 2 + flow[i][j][1] ** 2
                    gradFlow[i][j] = np.sqrt(mag)
                    angle = np.arctan2(flow[i][j][0], flow[i][j][1])
                    # angle = angle * 180 / np.pi
                    # angle = np.uint8(abs(angle))
                    angles[i][j] = angle


            hoof = HoG.bin_gradients(gradFlow, angles)

            hoof = HoG.normalize_hog(hoof)

            hoof = HoG.concatenate(hoof)

            #now append it
            descriptor.append(hoof)
            #each one has 100 videos, so use 70 for training and rest for testing
            if not has_more or curr_frame >= num_frames / 5:
                break
            else:
                old_frame = new_frame
        if count > 70:
            test_labels.append(label)
            testing.append(descriptor)
        else:
            train_labels.append(label)
            training.append(descriptor)
        count += 1
        cap.release()

print "Finished"
clf = svm.SVC()
clf.fit(training, train_labels)
predict = clf.predict(testing)
print metrics.accuracy_score(predict, test_labels)
exit(0)

# convert to gray-scale
frame1 = cv2.cvtColor(original_frame1,cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(original_frame2,cv2.COLOR_BGR2GRAY)



'''
plt.subplot(1,2,1),plt.imshow(frame1, cmap="gray"),plt.title("Frame 1"),plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(frame2, cmap="gray"),plt.title("Frame 2"),plt.xticks([]), plt.yticks([])
plt.show()
'''

# now need to compute optical flow -> for now just focus on calculating u and v, and don't worry about gradients in x and y direction
frame3 = np.zeros((len(frame1), len(frame1[0])), dtype=np.float64)
frame4 = np.zeros((len(frame1), len(frame1[0])), dtype=np.float64)

for i in range(0, len(frame1)):
    for j in range(0, len(frame1[0])):
        frame3[i][j] = np.float64(frame1[i][j] / 255.)
        frame4[i][j] = np.float64(frame2[i][j] / 255.)

flow = calcOpticalFlow(frame3, frame4)

'''
gradFlow = np.zeros((len(flow), len(flow[0])), dtype=np.uint64)
# get magnitude and direction
for i in range(1, len(flow) - 1):
    for j in range(1, len(flow[0]) - 1):
        mag = flow[i][j][0] ** 2 + flow[i][j][1] ** 2
        gradFlow[i][j] = np.sqrt(mag)
        angle = np.arctan2(flow[i][j][0], flow[i][j][1])
        #angle = angle * 180 / np.pi
        #angle = np.uint8(abs(angle))
        #gradFlow[i][j][1] = angle
'''


#flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0)

prvs = frame1
next = frame2
initial_flow = None
pyr_scale = 0.5
levels = 3
winsize = 15
iterations = 3
poly_n = 5
poly_sigma = 1.1
flags = 0

their_flow = cv2.calcOpticalFlowFarneback(prvs, next, 0, 0, 3, 15, 3, 5, 1, 0)


'''
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

hsv = np.zeros_like(original_frame1)
hsv[...,1] = 255
hsv[...,0] = ang * 180 / np.pi / 2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

plt.subplot(1,1,1),plt.imshow(rgb),plt.title("RGB")
plt.show()
'''
st2 = 1 # variable to reduce density of the vector field

Uf = np.flipud(flow[...,0]) # horizontal optical flow
Vf = np.flipud(-flow[...,1]) # vertical optical flow

Uf2 = np.flipud(their_flow[...,0]) # horizontal optical flow
Vf2 = np.flipud(-their_flow[...,1]) # vertical optical flow
Q2 = plt.quiver(Uf2, Vf2)

Q = plt.quiver(Uf,Vf)

plt.subplot(1,2,1),plt.quiver(Uf, Vf),plt.title("My optical flow"),plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.quiver(Uf2, Vf2),plt.title("Their optical flow"),plt.xticks([]), plt.yticks([])

plt.show()


Uf_dense = np.flipud(flow[...,0]) # horizontal optical flow
Vf_dense = np.flipud(-flow[...,1]) # vertical optical flow
m,n = Uf_dense.shape
y,x = np.mgrid[0.:m, 0.:n]

pct = 97
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
mag_threshold = np.percentile(mag.ravel(), pct)

x[mag<mag_threshold] = np.nan
y[mag<mag_threshold] = np.nan

plt.subplot(1,2,1)
Q = plt.quiver(x,y,Uf_dense,Vf_dense,color='r')
plt.imshow(frame1, cmap="gray"),plt.title("Frame 1")
plt.subplot(1,2,2)
Q = plt.quiver(x,y,Uf_dense,Vf_dense,color='r')
plt.imshow(frame2, cmap="gray"),plt.title("Frame 2")
plt.show()