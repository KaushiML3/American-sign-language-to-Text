import cv2
import mediapipe as mp
import numpy as np
import json
#import asyncio

from app.src.sign_t_text_model import random_forest ,gradien_boost ,cat_boost,xgb_boost #CNN_model
#from app.src.Yolo8_model import yolo_model

def resize_with_padding(image, desired_size):
    # Load the image
    #image = cv2.imread(image_path)

    # Get the current dimensions
    h, w = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h    
    # Determine new dimensions keeping the aspect ratio
    if aspect_ratio > 1:  # width is greater than height
        new_w = desired_size[0]
        new_h = int(new_w / aspect_ratio)
    elif aspect_ratio == 1:
        new_w = desired_size[0]
        new_h = desired_size[0]
    else:  # height is greater than width
        new_h = desired_size[1]
        new_w = int(new_h * aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding to make the image the desired size
    delta_w = desired_size[0] - new_w
    delta_h = desired_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Pad the image
    color = [225,225,225]  # Padding color (black)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return padded_image

def get_fps_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def image_prediction(data=None,kill:int=0):
    #data= image decode data from client side
    #kill=destroy hte ll cv2 windows
    
    # Initialize MediaPipe Hands and drawing utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cutoff=30
    desired_size=(300,300,3)


    # Choose font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2


    #final sent create variable
    temp_charecter=[]
    final_charecter=[]
    words=[]
    charecter=''
    sentences=[]
    

    # Initialize the Hands model
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

 
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    try:
        if frame is not None:
            

            image_rgb  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Read an image
            #cv2.imshow(image_rgb)

            # Process the image to detect hand landmarks
            results = hands.process(image_rgb)

            # Check if any hands are detected
            if results.multi_hand_landmarks:
                alphabetical_character=" "
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the landmarks
                    landmarks = hand_landmarks.landmark
                    #print(landmarks)
                    h, w, _ = image_rgb.shape
                    #print(h,w)

                    for i, landmark in enumerate(landmarks):
                        x, y, z = int(landmark.x * w), int(landmark.y * h), landmark.z
                        #print(f"Landmark {i}: (x: {x}, y: {y}, z: {z})")

                    # Initialize bounding box coordinates
                    min_x, min_y = w, h
                    max_x, max_y = 0, 0

                    # Iterate through landmarks to find the bounding box
                    for landmark in landmarks:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        min_x, min_y = min(min_x, x), min(min_y, y)
                        max_x, max_y = max(max_x, x), max(max_y, y)



                    #Define X Y margine
                    if min_x-cutoff < 0:
                        minx=0
                    else:
                        minx=min_x-cutoff

                    if min_y-cutoff < 0:
                        miny=0
                    else:
                        miny=min_y-cutoff

                    if max_x+cutoff > w:
                        maxx=max_x
                    else:
                        maxx=max_x+cutoff

                    if max_y+cutoff > h:
                        maxy=max_y
                    else:
                        maxy=max_y+cutoff

                    # Draw the bounding box
                    #cv2.rectangle(image_rgb , (minx, miny), (maxx, maxy), (0, 255, 0), 2)
                    #cv2_imshow(image)

                    # Crop the image to the bounding box
                    cropped_image = image_rgb [miny:maxy,minx:maxx]
                    #cv2_imshow(cropped_image)
                    if cropped_image is not None and cropped_image.size != 0:
                        try:

                            #resize image
                            resize_image=resize_with_padding(cropped_image, desired_size)
                            # Process the image to detect hand landmarks
                            results2 = hands.process(resize_image)
                            #cv2.imshow("image_crop",resize_image)

                            # Check if any hands are detected
                            if results2.multi_hand_landmarks:
                                #print("ok")
                                for hand_landmarks2 in results2.multi_hand_landmarks:
                                    # Get the landmarks
                                    landmarks2 = hand_landmarks2.landmark
                                    #print(landmarks)
                                    h2, w2, _ = resize_image.shape
                                    #print(h2,w2)

                                    #define x y storage location
                                    x_lists = []
                                    y_lists = []
                                    z_lists = []

                                    # Initialize bounding box coordinates
                                    min_x2, min_y2 = w2, h2
                                    max_x2, max_y2 = 0, 0

                                    for i, landmark in enumerate(landmarks2):
                                        x, y, z = int(landmark.x * w2), int(landmark.y * h2), int(landmark.z*100)
                                        x_lists.append(x)
                                        y_lists.append(y)
                                        z_lists.append(z)
                                        #print(f"Landmark {i}: (x: {x}, y: {y}, z: {z})")

                                    # Iterate through landmarks to find the bounding box
                                    for landmark in landmarks:
                                        x, y = int(landmark.x * w2), int(landmark.y * h2)
                                        min_x2, min_y2 = min(min_x2, x), min(min_y2, y)
                                        max_x2, max_y2 = max(max_x2, x), max(max_y2, y)

                                    list_pred=[]
                                    list_pred.extend(x_lists)
                                    list_pred.extend(y_lists)
                                    list_pred.extend(z_lists)

                                    # Optional: Draw landmarks and connections
                                    mp_drawing.draw_landmarks(
                                        resize_image,hand_landmarks2, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                                    
                                    #get prediction
                                    charecter_rf=random_forest(list_pred)

                                    charecter=charecter_rf

                                    # Display the output image with bounding box
                                    cv2.imshow("sub frame_2",resize_image)
                                    if resize_image is None:
                                        print("rerize none")
                                    #cv2.waitKey(1)


                        except ValueError as e:
                            print(e)
                    else:
                        pass
                


                # Optional: Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                

            # Add the text
            present_charecter=f" present charecter :{charecter}"

            cv2.putText(frame,present_charecter, (10,90), font, font_scale, (0, 255, 0), font_thickness)
                    

            cv2.imshow("Main_frame",frame)
            # Listen to the keyboard for presses.
            keyboard_input = cv2.waitKey(1)

            # Yield the processed result as a JSON string
            #yield f"data: {json.dumps({'Present charecter': present_charecter,'Final charecters':final_charecters})}\n\n"
            return charecter
            # Add a small delay to simulate real-time processing
            #await asyncio.sleep(0.1)

        elif kill == 1:
            #destroy hte ll cv2 windows
            cv2.destroyAllWindows() # Close the OpenCV window when the connection is closed

        else:
            #cv2.destroyAllWindows()
            print("image is None")

    except Exception as e:
        print(f"sign_to_text_video error: {e}")
