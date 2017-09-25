import numpy as np
import cv2



def classical_opencv(fname_video):
    cap = cv2.VideoCapture('vtest.webm')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
             break

    cap.release()
    cv2.destroyAllWindows()

def image_diff(imgfname,video):
    cap = cv2.VideoCapture(video)
    bgimg =  cv2.imread(imgfname)
    icap = 0 ## use the inmage at index 0 for the background - could use the average of first 100 frames for instance


    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01


    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    while(1):
        ret, frame = cap.read()
        if icap == 0:
            bgimg = frame
            bgimg = cv2.bilateralFilter(bgimg,9,75,75)
            #bgimg = cv2.GaussianBlur(bgimg,(5,5),0)
        icap +=1
        frame = cv2.bilateralFilter(frame,9,75,75)
        #frame = cv2.GaussianBlur(frame,(5,5),0)
        diff = frame - bgimg
        #diff = cv2.bilateralFilter(diff,9,75,75)

        #### TODO to improve the blob detection, do a black and white thresolding
        

        # see https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        # Detect blobs.
        keypoints = detector.detect(diff)

        print("found %s blobs",str(len(keypoints)))
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(diff, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        cv2.imshow('frame',im_with_keypoints)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
             break

    cap.release()
    cv2.destroyAllWindows()


fname_video = 'vtest.webm'
bg_img = 'bg.png'
image_diff(bg_img, fname_video)
#classical_opencv(fname_video)
