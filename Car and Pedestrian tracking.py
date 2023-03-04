import cv2

# video
video = cv2.VideoCapture('videoplayback.mp4')

#Pre-trained car and pedestrian classifier
classifier_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#creat classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker =cv2.CascadeClassifier(pedestrian_tracker_file)

# Run forever 
while True:
    
    # read the current frame 
    (read_successful, frame) = video.read()
    
    # safe coding
    if read_successful:
        
        #conver to grayscale (need for haar cascade)
       grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #detect cars adn pedestirians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    #Draw rectangles around the cars 
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0 , 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0 , 255), 2)
    #Draw rectangles around the pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255 , 255), 2)
  
    #Display the frame with car and pedestrian detector
    cv2.imshow('Car and Pedestirioan detector', frame)

    #Dont autoclose (Wait here in the code and listen for a key fress)
    key = cv2.waitKey(1)
    
    #stop if Q key is pressed 
    if key == 81 or key == 113:
        break
    
# Release the VideoCapture object
video.release()

# print out cars position
#print(cars)

