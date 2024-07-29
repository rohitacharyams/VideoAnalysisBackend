import os
import subprocess
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from moviepy.editor import VideoFileClip
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioProcessor:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.separated_signals = {}
        self.rate = None
        self.audio_processed = False

        

    def extract_audio_from_video(self):
        video = VideoFileClip(self.video_path)
        self.audio_path = os.path.join(self.output_dir, 'extracted_audio.wav')
        if not os.path.exists(self.audio_path):
            video.audio.write_audiofile(self.audio_path)
            self.separate_audio_components()
        print("We extracted the audio from video")

    def separate_audio_components(self):
        components_dir = os.path.join(self.output_dir, 'htdemucs_6s', os.path.basename(self.audio_path).replace('.wav', ''))
        if not os.path.exists(components_dir):
            command = [
                'demucs',
                self.audio_path,
                '-o', self.output_dir,
                '-n', 'htdemucs_6s'
            ]
            subprocess.run(command, check=True)
        
        print("We have separated the components")
        
        components = ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']
        for component in components:
            component_path = os.path.join(self.output_dir, 'htdemucs_6s', os.path.basename(self.audio_path).replace('.wav', ''), f'{component}.wav')
            signal, rate = torchaudio.load(component_path)
            self.separated_signals[component] = signal.squeeze().numpy()
            self.rate = rate
        self.save_separated_components()



    def save_separated_components(self):
        for component, signal in self.separated_signals.items():
            output_path = os.path.join(self.output_dir, f'{component}.wav')
            sf.write(output_path, signal.T, self.rate)
        print("We have saved the separated components")
    
    def get_total_frames(self):
        video = VideoFileClip(self.video_path)
        print("Here we are getting the total frames")
        return int(video.fps * video.duration)

    @staticmethod
    def extract_features(signal, rate, hop_length=256):
        D = librosa.stft(signal, hop_length=hop_length)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        onset_env = librosa.onset.onset_strength(y=signal, sr=rate)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=rate, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=rate, hop_length=hop_length)
        
        pitches, magnitudes = librosa.piptrack(y=signal, sr=rate, hop_length=hop_length)
        pitch = [pitches[:, i].max() for i in range(pitches.shape[1])]
        pitch_values = [np.max(pitches[:, i]) for i in range(pitches.shape[1]) if np.max(pitches[:, i]) > 0]
        pitch_times = librosa.frames_to_time(np.arange(len(pitch_values)), sr=rate, hop_length=hop_length)
        
        harmony = librosa.effects.harmonic(signal)
        melody = librosa.effects.percussive(signal)
        harmony_times = librosa.frames_to_time(np.arange(len(harmony) // hop_length), sr=rate, hop_length=hop_length)
        melody_times = librosa.frames_to_time(np.arange(len(melody) // hop_length), sr=rate, hop_length=hop_length)

        stft = librosa.stft(signal)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        
        return {
            'stft': stft_magnitude,
            'tempo': tempo,
            'beats': beat_times,
            'pitch': pitch,
            'harmony': harmony,
            'melody': melody
        }

    @staticmethod
    def extract_onset_times(signal, rate, hop_length=256, backtrack=False):
        onset_times = []
        for i in range(signal.shape[0]):
            onset_env = librosa.onset.onset_strength(y=signal[i], sr=rate, hop_length=hop_length)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=rate, hop_length=hop_length, backtrack=backtrack, delta=0.1, wait=20)
            onset_times_channel = librosa.frames_to_time(onset_frames, sr=rate, hop_length=hop_length)
            onset_times.extend(onset_times_channel)
        onset_times = np.unique(np.sort(onset_times))
        return onset_times

    def get_note_timestamps(self, signal, rate, hop_length=256):
        onset_times = self.extract_onset_times(signal, rate, hop_length)
        note_timestamps = []
        for i in range(len(onset_times) - 1):
            start_time = onset_times[i]
            end_time = onset_times[i + 1]
            note_timestamps.append((start_time, end_time))
        if len(onset_times) > 0:
            note_timestamps.append((onset_times[-1], len(signal[0]) / rate))
        return note_timestamps

    def extract_detailed_features(self, signal, rate, start, end, hop_length=256):
        segment = signal[:, int(start * rate):int(end * rate)]
        if segment.ndim > 1:
            segment = np.mean(segment, axis=0)
        pitches, magnitudes = librosa.core.piptrack(y=segment, sr=rate, hop_length=hop_length)
        pitch_values = [np.max(pitches[:, i]) for i in range(pitches.shape[1]) if np.max(pitches[:, i]) > 0]
        harmony = librosa.effects.harmonic(segment)
        rms_energy = librosa.feature.rms(y=segment, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(len(rms_energy[0])), sr=rate, hop_length=hop_length)
        return {
            'times': times,
            'pitch': pitch_values,
            'harmony': harmony,
            'intensity': rms_energy[0]
        }

    # def plot_intensity_curve(self, times, intensities, title):
    #     plt.figure(figsize=(14, 5))
    #     plt.plot(times, intensities[:len(times)], label='Intensity (RMS Energy)', color='b')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Intensity')
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()

    def save_segments(self, signal, rate, note_timestamps, output_dir, component):
        segment_dir = os.path.join(output_dir, f'{component}_segments')
        os.makedirs(segment_dir, exist_ok=True)
        for i, (start, end) in enumerate(note_timestamps):
            segment = signal[:, int(start * rate):int(end * rate)]
            segment_path = os.path.join(segment_dir, f'segment_{i+1}_{start:.3f}_to_{end:.3f}.wav')
            sf.write(segment_path, segment.T, rate)
            print(f"Saved segment {i+1} for {component}: {start:.3f}s to {end:.3f}s at {segment_path}")

    # def plot_onsets(self, signal, rate, onset_times, title):
    #     plt.figure(figsize=(14, 5))
    #     librosa.display.waveshow(signal[0], sr=rate, alpha=0.6)
    #     plt.vlines(onset_times, ymin=-1, ymax=1, color='r', linestyle='--', alpha=0.8, label='Onsets')
    #     plt.title(title)
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Amplitude')
    #     plt.legend()
    #     plt.show()
    
    def get_probable_end_frames(self, start_frame, frame_rate, max_frames=100, window_size=4):
        total_frames = self.get_total_frames()
        probable_end_frames = {}
        for component, signal in self.separated_signals.items():
            rate = self.rate
            note_timestamps = self.get_note_timestamps(signal, rate)
            print("Control is coming here")
            component_end_frames = []
            count = 0
            for start, end in note_timestamps:
                end_frame_calc = int(end * frame_rate)
                if start_frame < end_frame_calc < start_frame + max_frames:
                    for frame in range(end_frame_calc - window_size, end_frame_calc + window_size + 1):
                        if 0 <= frame < total_frames:
                            component_end_frames.append(frame)
                    count += 1
                    if count == 3:  # Collect at least three instances
                        break
            probable_end_frames[component] = list(set(component_end_frames))
        return probable_end_frames

    def process(self):
        self.extract_audio_from_video()
        self.separate_audio_components()
        self.save_separated_components()

        print("the audio is separated now")

        note_times = {}
        for component, signal in self.separated_signals.items():
            note_timestamps = self.get_note_timestamps(signal, self.rate)
            note_times[component] = note_timestamps
            print(f"Note timestamps for {component}: {note_timestamps}")
            self.save_segments(signal, self.rate, note_timestamps, self.output_dir, component)
            # self.plot_onsets(signal, self.rate, self.extract_onset_times(signal, self.rate), f'Onsets for {component}')

        # for component, signal in self.separated_signals.items():
        #     for start, end in note_times[component]:
        #         features = self.extract_detailed_features(signal, self.rate, start, end)
        #         self.plot_intensity_curve(features['times'], features['intensity'], f'Intensity for {component} from {start:.3f}s to {end:.3f}s')


# Example usage:
if __name__ == "__main__":
    video_path = '/Users/rohitacharya/Downloads/loveYa.mp4'
    output_dir = '/Users/rohitacharya/mmpose/demo/step_segmentation/output1'
    processor = AudioProcessor(video_path, output_dir)
    processor.process()
