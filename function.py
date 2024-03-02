#import dependency
import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):

  
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
      # Get the keypoint for a specific landmark (e.g., wrist)
      wrist_keypoint = rh[0:3]  # Assuming wrist landmark is at index 0
      # Determine the captured number based on specific logic (e.g., wrist position)
      # ... Replace this with your logic to identify numbers based on keypoints ...
      captured_number = 'fingers'
      # Save the keypoints with a filename including the number
      save_path = os.path.join(DATA_PATH, f"keypoints_{captured_number}.npy")
      np.save(save_path, rh)
      return rh
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['0', '1', '2', '3', '4', '5'])

no_sequences = 30

sequence_length = 30

