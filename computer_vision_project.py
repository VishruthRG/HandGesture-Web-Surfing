import cv2

vid = cv2.VideoCapture(0)
frame_no = 0
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    _, img = vid.read()
    cv2.imwrite("frame%d.jpg" % frame_no, img)
    frame_no += 1
vid.release()
cv2.destroyAllWindows
