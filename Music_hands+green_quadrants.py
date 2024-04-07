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
box_arr = []

# Function to draw quadrants on the frame
def draw_quadrants(frame):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Draw vertical lines to divide the frame into quadrants
    for i in range(1, 4):
        cv2.line(frame, (width * i // 4, 0), (width * i // 4, height), (0, 255, 0), 1)

    # Draw horizontal lines to divide the frame into quadrants
    for j in range(1, 4):
        cv2.line(frame, (0, height * j // 4), (width, height * j // 4), (0, 255, 0), 1)
   
    # Draw vertical and horizontal lines to divide the frame into 4 quadrants
    #cv2.line(frame, (width // 4, 0), (width // 4, height), (0, 255, 0), 1)
    #cv2.line(frame, (0, height // 4), (width, height // 4), (0, 255, 0), 1)

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

class Game:
    def __init__(self):
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(0)

    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())
    
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # TODO: Modify loop condition  
        while self.video.isOpened():
            # Read a frame from the video capture
            ret, frame = self.video.read()

            if not ret:
                break

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            self.draw_landmarks_on_hand(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Hand Tracking", image)

            # Draw quadrants on the frame
            draw_quadrants(image)

            # Display the frame
            cv2.imshow('Video', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()
