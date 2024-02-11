import numpy as np
import math
import cv2 as cv

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
        count_defects, drawing, distance, circularity = detect_gestures(crop_img)
        annotate_gesture(img, count_defects, distance, circularity)

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

# Differentiate the hand from the background
def segment_hand(crop_img):
    # Convert to YCrCb color space for skin detection
    ycrcb = cv.cvtColor(crop_img, cv.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], np.uint8)
    upper_skin = np.array([255, 173, 127], np.uint8)
    mask = cv.inRange(ycrcb, lower_skin, upper_skin)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    mask = cv.dilate(mask, kernel, iterations=2)
    mask = cv.GaussianBlur(mask, (3, 3), 0)
    hand_img = cv.bitwise_and(crop_img, crop_img, mask=mask)
    return hand_img, mask

# The function for detection of handshapes (by num of fingers) and waving
def detect_gestures(crop_img):
    prev_centroid = None
    distance = 0

    hand_img, mask = segment_hand(crop_img)

    # Preprocess
    roi = img_preprocessing(crop_img)
    
    # Using threshold to differentiate the ROI
    _, thresholded = cv.threshold(roi, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    count_defects = 0
    drawing = np.zeros(crop_img.shape, np.uint8)

    if contours:
        # Find the largest contour (the hand)
        max_contour = max(contours, key=lambda x: cv.contourArea(x))
        contour_area = cv.contourArea(max_contour)
        perimeter = cv.arcLength(max_contour, True)

        if max_contour is not None and len(max_contour) > 0:
            x, y, w, h = cv.boundingRect(max_contour)
            cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Computing the convexhull
            hull = cv.convexHull(max_contour)
            hull_area = cv.contourArea(hull)

            # Using the green line to highlight the contour (the hand)
            cv.drawContours(drawing, [max_contour], 0, (0, 255, 0), 0)
        
            hull = cv.convexHull(max_contour, returnPoints=False)
            defects = cv.convexityDefects(max_contour, hull)

            circularity = 4 * np.pi * (contour_area / (perimeter ** 2))

            # Find the centroid of the hand and draw it in blue
            M = cv.moments(max_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                curr_centroid = (cx, cy)
                cv.circle(drawing, curr_centroid, 5, (255, 255, 0), -1)

                # Detect a waving hand
                if prev_centroid is not None:
                    distance = math.sqrt((curr_centroid[0] - prev_centroid[0]) ** 2 + (curr_centroid[1] - prev_centroid[1]) ** 2)
                prev_centroid = curr_centroid

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # Defects detection
                    if angle <= 90:
                        count_defects += 1
                        cv.circle(drawing, far, 5, [0, 0, 255], -1)

                    cv.line(drawing, start, end, [0, 255, 0], 2)

    return count_defects, drawing, distance, circularity

# Annotating each hand shape and showing it on the screen
def annotate_gesture(img, count_defects, distance, circularity):
    text = ""
    if distance > 3:
        text = "Waving"
    elif circularity > 0.7:
        text = "Fist"
    elif count_defects == 0:
        text = "Pointing"
    elif count_defects == 1:
        text = "Scissors"
    elif count_defects == 2:
        text = "3 fingers"
    elif count_defects == 3:
        text = "4 fingers"
    elif count_defects == 4:
        text = "Open Hand"

    if text:
        cv.putText(img, text, (620, 50), cv.FONT_HERSHEY_COMPLEX, 2,(0, 0, 255),2, cv.LINE_AA)


if __name__ == "__main__":
    main()
