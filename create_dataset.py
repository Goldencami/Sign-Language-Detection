import os
import pickle # library to save datasets
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

# variables that contain all information
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # to save (x,y) coordinates
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # image are currently in bgr -> turn it into rgb

        results = hands.process(img_rgb)
        # check if we detected at least one hand
        if results.multi_hand_landmarks:
            # show what the landmarks look like on the hand
            for hand_landmarks in results.multi_hand_landmarks: # iterating through all different results
                ## draw points in images
                # mp_drawing.draw_landmarks(
                #     img_rgb, # image to draw
                #     hand_landmarks, # model output
                #     mp_hands.HAND_CONNECTIONS, # hand connections
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
                # print the value of all landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_) # category of symbol letter

## DON'T uncomment this unless you always take the first image from each dataset
## There's over 200 pictures
#         plt.figure()
#         plt.imshow(img_rgb)
# plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()