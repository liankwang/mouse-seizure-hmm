import os
from tqdm import tqdm
import numpy as np
import cv2
import torch

from .make_videos_utils import get_fps, extract_syllable_slices, get_state_clips, sample_clips

def generate_videos(train_dataset,
                    train_posteriors,
                    hmm_states,
                    num_states,
                    time_match, 
                    video_path,
                    save_path):
    
    # Get HMM ordered states
    hmm_usage = torch.bincount(hmm_states, minlength=num_states)
    hmm_order = torch.argsort(hmm_usage, descending=True)
    
    # Get average FPS
    fps = get_fps(time_match)
    
    # Create the output directory
    output_dir = f"{save_path}/grid_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Make videos for all states
    for state in tqdm(hmm_order, desc="Generating videos for states"):
        # print(f"Generating video for state {state.item()}...")
        slices = extract_syllable_slices(state, train_posteriors)
        clips = get_state_clips(slices, train_dataset, time_match, video_path)
        output_path= f"{output_dir}/state_{state.item()}.mp4"

        if len(clips) == 0:
            # print(f"No clips found for state {state.item()}. Skipping.")
            continue
        
        generate_grid_video(clips, grid_width=3, grid_length=3,
                            output_path=output_path, fps=fps,
                            duration=5)

def generate_grid_video(clips, grid_width, grid_length, output_path, fps, duration):
    target_num_frames = fps * duration
    
    # Randomly sample a subset of clips
    num_clips_to_sample = grid_width * grid_length
    selected_clips = sample_clips(clips, num_clips_to_sample, min_length=target_num_frames)
    
    # Compute height and width of grid
    sample_frame = clips[0][0]
    height, width = sample_frame.shape[:2]
    grid_width_len = grid_width * width
    grid_height_len = grid_length * height

    # Pad clips to equal length
    num_frames_per_clip = min(int(target_num_frames), max(len(clip) for clip in selected_clips))
    padded_clips = []
    for clip in selected_clips:
        if len(clip) < num_frames_per_clip:
            # Pad with zeros (black frames)
            padding = np.zeros((num_frames_per_clip - len(clip), height, width, 3), dtype=np.uint8)
            padded_clip = np.concatenate([clip, padding], axis=0)
        else:
            padded_clip = clip[:num_frames_per_clip]
        padded_clips.append(padded_clip)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(fps), (grid_width_len, grid_height_len))

    # Generate grid frames
    for i in tqdm(range(num_frames_per_clip), desc="Writing grid frames", leave=False):
        grid_image = np.zeros((grid_height_len, grid_width_len, 3), dtype=np.uint8)

        for clip_idx, clip in enumerate(padded_clips):
            row = clip_idx // grid_length
            col = clip_idx % grid_width
            y_start = row * height
            x_start = col * width

            grid_image[y_start:y_start+height, x_start:x_start+width] = clip[i]

        out.write(grid_image)

    out.release()

