import numpy as np
import math
import cv2 as cv

# Init global variables
background = None
prev_centroid = None
waving = False
frame = 0
wait_time = 30

def main():
    # The main loop for video capture
    cap = cv.VideoCapture(0)

    while True:
        _, img = cap.read()
        if img is None:
            break
        
        # Flip the image to make life easier
        img = cv.flip(img, 1)

        # Crop the region of interest - the top left corner of the video
        cv.rectangle(img, (0, 0), (600, 600), (0, 255, 0), 2)
        crop_img = img[0:600, 0:600]

        # Detect gestures and display the text
        count_defects, drawing, waving = detect_gestures(crop_img)
        annotate_gesture(img, count_defects, waving)

        # Display the drawing in the original image
        img[0:600, 0:600] = drawing  
        cv.imshow('Gesture Detection', img)

        # Press 'q' to stop video capturing
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Convert the image to grayscale and blur it for better recognition
def img_preprocessing(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grey, (5, 5), 0)
    return blurred

# Find the centroid of the contour (hand)
def find_centroid(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        return None

# The function for detection of handshapes (by num of fingers) and waving
def detect_gestures(crop_img):
    global prev_centroid, waving

    # Preprocess
    roi = img_preprocessing(crop_img)
    
    # Using threshold to differentiate the ROI
    _, thresholded = cv.threshold(roi, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    count_defects = 0
    drawing = np.zeros(crop_img.shape, np.uint8)

    if contours:
        # Find the largest contour (the hand)
        count1 = max(contours, key=lambda x: cv.contourArea(x))
        if count1 is not None and len(count1) > 0:
            x, y, w, h = cv.boundingRect(count1)
            cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Computing the convexhull
            hull = cv.convexHull(count1)

            # Using the green line to highlight the contour (the hand)
            cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0)
        
            hull = cv.convexHull(count1, returnPoints=False)
            defects = cv.convexityDefects(count1, hull)

            centroid = find_centroid(count1)
            if prev_centroid is not None and centroid is not None:
                dx = centroid[0] - prev_centroid[0]  # Change in X
                dy = centroid[1] - prev_centroid[1]  # Change in Y

                # More movement in X than in Y - a wave
                if abs(dx) > abs(dy):
                    waving = True

            # Update the position of the centroid
            prev_centroid = centroid

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(count1[s][0])
                    end = tuple(count1[e][0])
                    far = tuple(count1[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # Defects detection
                    if angle <= 90:
                        count_defects += 1
                        cv.circle(drawing, far, 1, [0, 0, 255], -1)

                    cv.line(drawing, start, end, [0, 255, 0], 2)

    return count_defects, drawing, waving

# Annotating each hand shape and showing it on the screen
def annotate_gesture(img, count_defects, waving):
    text = ""
    if count_defects == 0:
        text = "Pointing"
    elif count_defects == 1:
        text = "Scissors"
    elif count_defects == 2:
        text = "3 fingers"
    elif count_defects == 3:
        text = "4 fingers"
    elif count_defects == 4:
        text = "Open Hand"
    elif waving == True:
        text = "Waving"

    if text:
        cv.putText(img, text, (620, 50), cv.FONT_HERSHEY_COMPLEX, 2,(0, 0, 255),2, cv.LINE_AA)


if __name__ == "__main__":
    main()
