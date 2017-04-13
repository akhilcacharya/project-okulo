import os, cv2, sys
import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from sklearn import svm, metrics, tree, ensemble, linear_model
from sklearn.model_selection import cross_val_score

# The dimensions to resize video frames to before calculating HoF
FIXED_WIDTH = 128
FIXED_HEIGHT = 128

# Property useful for finding length (in frames) of a video.
# Renamed here because the way to access it can differ between machines, so we can change it easily
FRAME_LENGTH_PROP = cv2.CAP_PROP_FRAME_COUNT

# Threshold ratio for segmentation. Segments longer than this * length of smallest video are not discarded
SEG_THRESHOLD = 0.5

# Ratio of training data to testing data
#TRAIN_RATIO = 0.7

# level parameter for pyramiding HoF
LEVEL = 3

# Directory and files of the first action
ACT1_DIR = "./hof2/running/"
ACT1_FILES = os.listdir(ACT1_DIR)
ACT1_FILES = [ACT1_DIR + f for f in ACT1_FILES]

# Directory and files of the second action
ACT2_DIR = "./hof2/handclapping/"
ACT2_FILES = os.listdir(ACT2_DIR)
ACT2_FILES = [ACT2_DIR + f for f in ACT2_FILES]

# Directory and files of the third action
ACT3_DIR = "./hof2/handwaving/"
ACT3_FILES = os.listdir(ACT3_DIR)
ACT3_FILES = [ACT3_DIR + f for f in ACT3_FILES]

# Directory and files of the fourth action
ACT4_DIR = "./hof2/walking/"
ACT4_FILES = os.listdir(ACT4_DIR)
ACT4_FILES = [ACT4_DIR + f for f in ACT4_FILES]

# Directory and files of the fifth action
ACT5_DIR = "./hof2/boxing/"
ACT5_FILES = os.listdir(ACT5_DIR)
ACT5_FILES = [ACT5_DIR + f for f in ACT5_FILES]

# Use equal number of data from each class, setting a cap at a total of 464 files
nc = min(len(ACT1_FILES), len(ACT2_FILES), len(ACT3_FILES), len(ACT4_FILES), len(ACT5_FILES))
nc = nc if (nc < 30) else 30
print "nc:", nc
ACT1_FILES = ACT1_FILES[0:nc]
ACT2_FILES = ACT2_FILES[0:nc]
ACT3_FILES = ACT3_FILES[0:nc]
ACT4_FILES = ACT4_FILES[0:nc]
ACT5_FILES = ACT5_FILES[0:nc]

#offset = int(np.floor(nc*TRAIN_RATIO))
#print "offset:", offset

# Split test and training at ratio
train_files = ACT1_FILES[0:nc] + ACT2_FILES[0:nc] + ACT3_FILES[0:nc] + ACT4_FILES[0:nc] + ACT5_FILES[0:nc]
#test_files = ACT1_FILES[offset:nc] + ACT2_FILES[offset:nc] + ACT3_FILES[offset:nc] + ACT4_FILES[offset:nc] + ACT5_FILES[offset:nc]

# Put the labels in vectors
train_labels = np.zeros(nc*5, int)
train_labels[0:nc] = 1
train_labels[nc:nc*2] = 2
train_labels[nc*2:nc*3] = 3
train_labels[nc*3:nc*4] = 4
train_labels[nc*4:nc*5] = 5

#test_len = nc-offset
#test_labels = np.zeros(test_len*5, int)
#test_labels[0:test_len] = 1
#test_labels[test_len:test_len*2] = 2
#test_labels[test_len*2:test_len*3] = 3
#test_labels[test_len*3:test_len*4] = 4
#test_labels[test_len*4:test_len*5] = 5

print "train files:", len(train_files)
print "train labels:", len(train_labels)
#print "test files:", len(test_files)
#print "test labels:", len(test_labels)

def normalizeFrame(frame_original):
    frame_gray = cv2.cvtColor(frame_original,cv2.COLOR_BGR2GRAY)
    frame_gray_resized = cv2.resize(frame_gray, (FIXED_WIDTH, FIXED_HEIGHT))
    return frame_gray_resized

# Gets the optical flow [<dx,dy>] from two frames
def getOpticalFlow(imPrev, imNew):
    flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=9,
                                        iterations=1, poly_n=3, poly_sigma=1.1,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

# Compute the Histogram of Optical Flow (HoF) from the given optical flow
def hof(flow, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(4, 3), normalise=False, motion_threshold=1.):
    flow = np.atleast_2d(flow)

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")

    if normalise:
        flow = sqrt(flow)

    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])

    gx = flow[:,:,1]
    gy = flow[:,:,0]

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))
    n_cellsy = int(np.floor(sy // cy))

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]
    for i in range(orientations-1):
        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)

        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    return normalised_blocks.ravel()

# get the Histogram of Optical Flow from two images
def getHoF(frame1, frame2):
    flow = getOpticalFlow(frame1, frame2)
    return hof(flow, pixels_per_cell=(20,20), cells_per_block=(5,5))

# get the Histogram of Optical Flows of a video grouped sequentially in a 1D array
def getSequentialHoF(video_path):
    hofs = []
    cap = cv2.VideoCapture(video_path)
    ret1, frame1 = cap.read()
    frame1 = normalizeFrame(frame1)
    while(cap.isOpened()):
        ret2, frame2 = cap.read()
        if ret2 == True:
            frame2 = normalizeFrame(frame2)
            hof_array = getHoF(frame1, frame2)
            hofs = np.concatenate((hofs, hof_array),axis=0)
            frame1 = frame2
        else:
            break
    return hofs

# Find the length of the largest row in training set and testing set
def maxRow(train):#, train):
    return np.array([len(i) for i in train]).max()#,
#               np.array([len(i) for i in test]).max())

# Pad each row of the 2D array, with 0, to a specified width
def numpy_fillna(data, width):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(width) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out
def getFeatures_Baseline(train_files, train_labels):#, test_files, test_labels):
    # Make feature vectors
    train = [getSequentialHoF(p) for p in train_files]
    #test = [getSequentialHoF(p) for p in test_files]

    # Pad them to the max video width
    max_width = maxRow(train)#, train);
    train_pad = numpy_fillna(np.array(train), max_width)
    #test_pad = numpy_fillna(np.array(test), max_width)

    return train_pad, train_labels#, test_pad, test_labels


# Determine the length in frames of the shortest video in the provided dataset
def shortest(data_dir):
    # get list of files in the directory. directory should be flat with only video files in it
    files = os.listdir(data_dir)

    # Find the length of the shortest video (in frames)
    shortestLen = sys.maxint
    for i in range(len(files)):
        cap = cv2.VideoCapture(data_dir+files[i])
        length = int(cap.get(FRAME_LENGTH_PROP))
        if length < shortestLen:
            shortestLen = length

        cap.release()

    return shortestLen

# Get the Histogram of Optical Flows of a video grouped sequentially in a 1D array
# Use only the specified amount of frames, from the middle of the video
def getSequentialHoFMiddle(video_path, frames):
    hofs = []
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(FRAME_LENGTH_PROP))
    startIdx = ((length - 1) - frames)/2
    if startIdx < 1:
        startIdx = 0

    # skip through beginning unneeded frames
    frameNum = 0
    while (frameNum < startIdx):
        cap.grab()
        frameNum += 1
    frameNum = 0

    # Calculate HoF from necessary frames
    ret1, frame1 = cap.read()
    frame1 = normalizeFrame(frame1)
    while(frameNum < frames-2):
        ret2, frame2 = cap.read()
        if ret2 == True:
            frame2 = normalizeFrame(frame2)
            hof_array = getHoF(frame1, frame2)
            hofs = np.concatenate((hofs, hof_array),axis=0)
            frame1 = frame2
            frameNum += 1
        else:
            break

    cap.release()
    return hofs

def getFeatures_Trimmed(train_files, train_labels):#, test_files, test_labels):
    numFrames = min(shortest(ACT1_DIR), shortest(ACT2_DIR))
    train = [getSequentialHoFMiddle(p, numFrames) for p in train_files]
    #test = [getSequentialHoFMiddle(p, numFrames) for p in test_files]

    return np.array(train), train_labels#, np.array(test), test_labels

def getSequentialHoFSegments(video_path, label, frames):
    seg_hofs = []
    hofs = []
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(FRAME_LENGTH_PROP))

    ret1, frame1 = cap.read()
    frame1 = normalizeFrame(frame1)
    while(cap.isOpened()):
        hofs = []
        for i in range(frames-1):
            ret2, frame2 = cap.read()
            if ret2 == True:
                frame2 = normalizeFrame(frame2)
                hof_array = getHoF(frame1, frame2)
                hofs = np.concatenate((hofs, hof_array),axis=0)
                frame1 = frame2
            else:
                cap.release()
                break
        seg_hofs.append(hofs)

    seg_labels = np.full(len(seg_hofs), label)
    return seg_hofs, seg_labels

def getFeatures_Segmented(train_files, train_labels):#, test_files, test_labels):
    numFrames = min(shortest(ACT1_DIR), shortest(ACT2_DIR))
    train_result = [a for t,l in zip(train_files, train_labels) for a in getSequentialHoFSegments(t, l, numFrames)]
    #test_result = [a for t,l in zip(test_files, test_labels) for a in getSequentialHoFSegments(t, l, numFrames)]
    train = [y for x in train_result[::2] for y in x]
    new_train_labels = np.concatenate(train_result[1::2])
    #test = [y for x in test_result[::2] for y in x]
    #new_test_labels = np.concatenate(test_result[1::2])
    train_threshold = int(max([len(x) for x in train]) * SEG_THRESHOLD)
    #test_threshold = int(max([len(x) for x in test]) * SEG_THRESHOLD)
    train_trimmed = [x for x,y in zip(train,new_train_labels) if len(x) > train_threshold]
    train_labels_trimmed = [y for x,y in zip(train,new_train_labels) if len(x) > train_threshold]
    #test_trimmed = [x for x,y in zip(test,new_test_labels) if len(x) > test_threshold]
    #test_labels_trimmed = [y for x,y in zip(test,new_test_labels) if len(x) > test_threshold]
    max_width = np.array([len(i) for i in train_trimmed]).max()#max(np.array([len(i) for i in train_trimmed]).max(),np.array([len(i) for i in test_trimmed]).max())
    train_pad = numpy_fillna(np.array(train_trimmed), max_width)
    #test_pad = numpy_fillna(np.array(test_trimmed), max_width)

    return train_pad, train_labels_trimmed#, test_pad, test_labels_trimmed

def getPyramid(img, level):
    p = []
    p.append(img)
    for i in range(level):
        p.append(cv2.pyrDown(p[len(p)-1]))
    return p

def pyramidHof(frame1, frame2, level, getHofFunc, getPyramidFunc, getOpticalFlowFunc):
    if frame1.shape != frame2.shape:
        raise ValueError('frame1 and frame2 should have identical dimensions')
    pyramid1 = getPyramidFunc(frame1, level)
    pyramid2 = getPyramidFunc(frame2, level)
    pyramidHoF = []
    for i in range(level):
        subframe1 = pyramid1[i]
        subframe2 = pyramid2[i]
        subflow = getOpticalFlowFunc(subframe1, subframe2)
        subhof = getHofFunc(subflow)
        pyramidHoF = np.append(pyramidHoF, subhof)
    return pyramidHoF

def getPyramidHof(video_path, level, normalizeFrameFunc=normalizeFrame, getHofFunc=hof, getPyramidFunc=getPyramid, getOpticalFlowFunc=getOpticalFlow):
    cap = cv2.VideoCapture(video_path)
    vid_hof = []
    ret1, frame1 = cap.read()
    frame1 = normalizeFrameFunc(frame1)
    while(cap.isOpened()):
        ret2, frame2 = cap.read()
        if ret2 == True:
            frame2 = normalizeFrameFunc(frame2)
            biframe_hof = pyramidHof(frame1, frame2, level, getHofFunc, getPyramidFunc, getOpticalFlowFunc)
            vid_hof = np.append(vid_hof, biframe_hof)
            frame1 = frame2
        else:
            break
    cap.release()
    return vid_hof

def getFeatures_Pyramid_Baseline(train_files, train_labels):#, test_files, test_labels):
    # make feature vectors
    train = [getPyramidHof(p, LEVEL) for p in train_files]
    #test = [getPyramidHof(p, LEVEL) for p in test_files]

    # Pad them to the max video width
    max_width = maxRow(train)#, train);
    train_pad = numpy_fillna(np.array(train), max_width)
    #test_pad = numpy_fillna(np.array(test), max_width)

    return train_pad, train_labels#, test_pad, test_labels


def getPyramidHofMiddle(video_path, frames, level, normalizeFrameFunc=normalizeFrame, getHofFunc=hof, getPyramidFunc=getPyramid, getOpticalFlowFunc=getOpticalFlow):
    cap = cv2.VideoCapture(video_path)
    vid_hof = []

    length = int(cap.get(FRAME_LENGTH_PROP))
    startIdx = ((length - 1) - frames)/2
    if startIdx < 1:
        startIdx = 0
    # skip through beginning unneeded frames
    frameNum = 0
    while (frameNum < startIdx):
        cap.grab()
        frameNum += 1
    frameNum = 0

    ret1, frame1 = cap.read()
    frame1 = normalizeFrameFunc(frame1)
    while(frameNum < frames-2):
        ret2, frame2 = cap.read()
        if ret2 == True:
            frame2 = normalizeFrameFunc(frame2)
            biframe_hof = pyramidHof(frame1, frame2, level, getHofFunc, getPyramidFunc, getOpticalFlowFunc)
            vid_hof = np.append(vid_hof, biframe_hof)
            frame1 = frame2
            frameNum += 1
        else:
            break
    cap.release()
    return vid_hof

def getFeatures_Pyramid_Trimmed(train_files, train_labels):#, test_files, test_labels):
    numFrames = min(shortest(ACT1_DIR), shortest(ACT2_DIR))
    train = [getPyramidHofMiddle(p, numFrames, LEVEL) for p in train_files]
    #test = [getPyramidHofMiddle(p, numFrames, LEVEL) for p in test_files]

    return np.array(train), train_labels#, np.array(test), test_labels

def getPyramidHoFSegments(video_path, label, frames, level, normalizeFrameFunc=normalizeFrame, getHofFunc=hof, getPyramidFunc=getPyramid, getOpticalFlowFunc=getOpticalFlow):
    seg_hofs = []
    hofs = []
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(FRAME_LENGTH_PROP))

    ret1, frame1 = cap.read()
    frame1 = normalizeFrame(frame1)
    while(cap.isOpened()):
        hofs = []
        for i in range(frames-1):
            ret2, frame2 = cap.read()
            if ret2 == True:
                frame2 = normalizeFrameFunc(frame2)
                biframe_hof = pyramidHof(frame1, frame2, level, getHofFunc, getPyramidFunc, getOpticalFlowFunc)
                hofs = np.concatenate((hofs, biframe_hof),axis=0)
                frame1 = frame2
            else:
                cap.release()
                break
        seg_hofs.append(hofs)

    seg_labels = np.full(len(seg_hofs), label)
    return seg_hofs, seg_labels

def getFeatures_Pyramid_Segmented(train_files, train_labels):#, test_files, test_labels):
    numFrames = min(shortest(ACT1_DIR), shortest(ACT2_DIR))
    train_result = [a for t,l in zip(train_files, train_labels) for a in getPyramidHoFSegments(t, l, numFrames, LEVEL)]
    #test_result = [a for t,l in zip(test_files, test_labels) for a in getPyramidHoFSegments(t, l, numFrames, LEVEL)]
    train = [y for x in train_result[::2] for y in x]
    new_train_labels = np.concatenate(train_result[1::2])
    #test = [y for x in test_result[::2] for y in x]
    #new_test_labels = np.concatenate(test_result[1::2])
    train_threshold = int(max([len(x) for x in train]) * SEG_THRESHOLD)
    #test_threshold = int(max([len(x) for x in test]) * SEG_THRESHOLD)
    train_trimmed = [x for x,y in zip(train,new_train_labels) if len(x) > train_threshold]
    train_labels_trimmed = [y for x,y in zip(train,new_train_labels) if len(x) > train_threshold]
    #test_trimmed = [x for x,y in zip(test,new_test_labels) if len(x) > test_threshold]
    #test_labels_trimmed = [y for x,y in zip(test,new_test_labels) if len(x) > test_threshold]
    max_width = np.array([len(i) for i in train_trimmed]).max()#max(np.array([len(i) for i in train_trimmed]).max(),np.array([len(i) for i in test_trimmed]).max())
    train_pad = numpy_fillna(np.array(train_trimmed), max_width)
    #test_pad = numpy_fillna(np.array(test_trimmed), max_width)

    return train_pad, train_labels_trimmed#, test_pad, test_labels_trimmed

def eval_model(classifier, (data, labels)):
    clf = None
    name = ""
    if (classifier == 0):
        clf = svm.SVC()
        name = "SVM"
    elif (classifier == 1):
        clf = tree.DecisionTreeClassifier()
        name = "Decision Tree"
    elif (classifier == 2):
        clf = ensemble.RandomForestClassifier()
        name = "Random Forest"
    else:
        clf = linear_model.LogisticRegression()
        name = "Logistic Regression"

    #clf.fit(train_features, train_labels)
    #predict = clf.predict(test_features)
    cross = cross_val_score(clf, data, labels, cv=5)

    return cross, name

def run(data):
    for i in range(4):
        score, name = eval_model(i, data);
        print(name + ": %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

print "baseline"
data = getFeatures_Baseline(train_files, train_labels)#, test_files, test_labels)
run(data)

print "\n\ntrimmed"
data = getFeatures_Trimmed(train_files, train_labels)#, test_files, test_labels)
run(data)

print "\n\nsegmented"
data = getFeatures_Segmented(train_files, train_labels)#, test_files, test_labels)
run(data)

print "\n\npyramid"
data = getFeatures_Pyramid_Baseline(train_files, train_labels)#, test_files, test_labels)
run(data)

print "\n\np + t"
data = getFeatures_Pyramid_Trimmed(train_files, train_labels)#, test_files, test_labels)
run(data)

print "\n\np + s"
data = getFeatures_Pyramid_Segmented(train_files, train_labels)#, test_files, test_labels)
run(data)
