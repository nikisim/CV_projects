import cv2
scale_factor = 1.4
min_neighbors = 3
min_size = (210, 210)
webcam=False #if working with video file then make it 'False'
def detect(path):
    cascade = cv2.CascadeClassifier(path)
    if webcam:
        video_cap = cv2.VideoCapture(0) # use 0,1,2..depanding on your webcam
    else:
        video_cap = cv2.VideoCapture("spider.mp4")
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))
    out = cv2.VideoWriter('spider_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()
        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, 'Face', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0 , 255 , 0), 1)
            # Display the resulting frame
            out.write(img)
            cv2.imshow("Face video detector", img)
            #wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
    out.release()
    cv2.destroyAllWindows()
def main():
    cascadeFilePath="haarcascade_frontalface_default.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()