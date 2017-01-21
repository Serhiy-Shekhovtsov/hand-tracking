import numpy as np
import cv2
import get_points

cap = cv2.VideoCapture(0)

# ###1 ALL THIS WILL BE REPLACED WITH CAFFE DETECTION
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    if not ret:
        print "Cannot capture frame device"
        exit()
    if cv2.waitKey(10) == ord('p'):
        break
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

cv2.destroyWindow("Image")
#    tlx  tly  brx  bry
# [(359, 233, 630, 408)]
points = get_points.run(img)

if not points:
    print "ERROR: No object to be tracked."
    exit()

# ### END 1

bbplus = 100
minx = points[0][0]
miny = points[0][1]
maxx = points[0][2]
maxy = points[0][3]

r, h, c, w = minx, miny, maxx-minx, maxy-miny  # simply hardcoded the values

# (xmin, ymin, xmax - xmin, ymax - ymin)
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = img[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either iterations or move by at least some pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    if ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        (c, r, w, h) = track_window
        img_hand = img[r-bbplus:r + h + bbplus, c-bbplus:c + w + bbplus]

        #!!! find contours
        gray = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2)
        ret2, thresh1 = cv2.threshold(blur, 70, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(img_hand.shape, np.uint8)
        if thresh1 is not None:
            cv2.imshow('hand', thresh1)

        # > extract the largest contour
        max_area = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                ci = i
        cnt = contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

        centr = (cx, cy)
        cv2.circle(img_hand, centr, 5, [0, 0, 255], 2)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if cnt.any() and hull.any():
            defects = cv2.convexityDefects(cnt, hull)
            mind = 0
            maxd = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    dist = cv2.pointPolygonTest(cnt, centr, True)
                    cv2.line(img_hand, start, end, [0, 255, 0], 2)

                    cv2.circle(img_hand, far, 5, [0, 0, 255], -1)
            i = 0
        #!!! find contours end
        # cv2.imshow('hand', img_hand)

        # Draw it on image - meanShift
        # x, y, w, h = track_window
        # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        # Draw it on image - CamShift
        img[r-bbplus:r + h + bbplus, c-bbplus:c + w + bbplus] = img_hand
        pts = cv2.cv.BoxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
        cv2.imshow('img2', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img)

    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()