import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, keyframes_list, video_path):
        self.keyframes_list = keyframes_list
        self.video_path = video_path

    def process_keyFrames(self):
        itr = 0
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # print()

        if len(self.keyframes_list) % 2:
            self.keyframes_list.append({'frame':total_frames, 'type': 'out'})
        
        print(self.keyframes_list)

        while itr < len(self.keyframes_list):
            print(itr)
            frame_in_index = self.keyframes_list[itr]['frame']
            frame_out_index = self.keyframes_list[itr + 1]['frame']
            self.process_video(frame_in_index, frame_out_index)
            itr = itr + 2

        return

    def process_video(self, starting_frame, ending_frame):
        # Add your video analysis logic here
        # You can use OpenCV or other libraries for video processing

        # Example: Display facial landmarks for each keyframe
        
        # Let's say we have a keyframe from frame_index -> 'starting_frame' to 'ending_frame'
        # Now we have to process these many frames
        # here we could introduce the concept of frames interpolation too
        print("Hiiiiiiiiiiiiiiii")
        print(starting_frame, ending_frame)
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        success, frame = cap.read()
        cap.release()
        return

        if success:
            # Implement your facial analysis logic here
            # Example: Display facial landmarks using dlib
            # (Make sure to install dlib: pip install dlib)
            # This is just a placeholder, you can replace it with your actual analysis code
            self.display_facial_landmarks(frame)
        else:
            print(f'Error processing frame {starting_frame} - Invalid frame index')

    def display_facial_landmarks(self, frame):
        # Placeholder method for displaying facial landmarks
        # Replace this with your actual facial analysis code
        print('Displaying facial landmarks for analysis...')
        # Your facial analysis code here
