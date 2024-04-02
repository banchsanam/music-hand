import cv2
import pyaudio
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Game:
    def __init__(self):
       # TODO: Modify loop condition  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = cv2.flip(image, 1)

       # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(0)

        # box arrangement
        box_arr = []
        
def play_frequency(frequency, duration):
    # Parameters for the audio stream
    sample_rate = 44100  # Sample rate (samples per second)
    duration_sec = duration  # Duration of the tone in seconds

    # Generate the waveform
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)

    # Open the audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # Play the waveform
    stream.write(waveform.astype(np.float32).tostring())

    # Close the stream and terminate the PyAudio instance
    stream.close()
    p.terminate()

    def detect_hands(self, frame):
         # Convert the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)

            # Detect hands
            results = self.detector.process(image)

            return results

    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        if detection_result.multi_hand_landmarks:
            for hand_landmarks in detection_result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        while self.video.isOpened():
            # Get the current frame
            ret, frame = self.video.read()

            if not ret:
                break

            # detect hands
            results = self.detect_hands(frame)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)

            # Extract hand center
            hand_centers = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    cx, cy = 0, 0
                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        hand_centers.append((cx, cy))

            # Draw boxes based on hand centers
            for cx, cy in hand_centers:
                for box in self.box_arrangements:
                    x, y, w, h = box
                    if x < cx < x + w and y < cy < y + h:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 50, 50), 5)
                        if (x, y) == (0, 0):
                           play_frequency(440, 2)
            
            # Display the image
            cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":        
    g = Game()
    g.run()
