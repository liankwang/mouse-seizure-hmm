from typing import List, Dict
from jaxtyping import Float, Int

import torch
import numpy as np
import numpy.random as npr
import cv2

from src.utils import from_t

def extract_syllable_slices(
    state_idx: Int[torch.Tensor, ""],
    posteriors: List[Dict[str, Float[torch.Tensor, "num_timesteps num_states"]]],
    pad: int = 30,
    num_instances: int = 50,
    min_duration: int = 5,
    max_duration: int = 45,
    seed: int = 0
    ) -> List[List[slice]]:
    """Extracts slices of data corresponding to occurrences of a specified state.

    This function identifies segments (syllables) where a given `state_idx`
    is the most probable state, based on posterior probabilities. It processes
    data for multiple "mice" (sessions), concatenates all found occurrences,
    filters them by duration and proximity to the start of the sequence (padding),
    and then selects a random subset of these occurrences. The final output is
    a list of lists, where the outer list corresponds to each mouse/session,
    and each inner list contains slice objects representing the selected
    syllable occurrences for that mouse.

    Args:
        state_idx: A scalar integer tensor representing the target state index
            for which to extract syllables.
        posteriors: A list of dictionaries, where each dictionary corresponds
            to a mouse/session. Each dictionary must contain the key
            "expected_states", which maps to a 2D tensor of shape
            (num_timesteps, num_states) representing the posterior probabilities
            of being in each state at each time step.
        pad: Minimum number of time steps from the beginning of a sequence
            for a syllable to be considered valid. Syllables starting before
            this padding are discarded. Defaults to 30.
        num_instances: The maximum number of syllable instances to randomly
            select across all mice after filtering. If fewer valid instances
            are found, all valid instances are returned. Defaults to 50.
        min_duration: The minimum duration (in time steps) for a syllable
            to be considered valid. Shorter syllables are discarded. Defaults to 5.
        max_duration: The maximum duration (in time steps) for a syllable
            to be considered valid. Longer syllables are discarded. Defaults to 45.
        seed: Seed for the random number generator used to select a subset
            of syllable instances. Defaults to 0.

    Returns:
        A list of lists of slice objects. The outer list has a length equal to
        the number of mice (i.e., `len(posteriors)`). Each inner list contains
        `slice(start_index, stop_index)` objects for the selected syllables
        of the corresponding mouse.
    """
    # Find all the start indices and durations of specified state
    state_idx = from_t(state_idx)
    all_mouse_inds = []
    all_starts = []
    all_durations = []
    for mouse, posterior in enumerate(posteriors):
        expected_states = from_t(posterior["expected_states"])
        states = np.argmax(expected_states, axis=1)
        states = np.concatenate([[-1], states, [-1]])
        starts = np.where((states[1:] == state_idx) \
                          & (states[:-1] != state_idx))[0]
        stops = np.where((states[:-1] == state_idx) \
                         & (states[1:] != state_idx))[0]
        durations = stops - starts
        assert np.all(durations >= 1)
        all_mouse_inds.append(mouse * np.ones(len(starts), dtype=int))
        all_starts.append(starts)
        all_durations.append(durations)

    all_mouse_inds = np.concatenate(all_mouse_inds)
    all_starts = np.concatenate(all_starts)
    all_durations = np.concatenate(all_durations)

    # Throw away ones that are too short or too close to start.
    # TODO: also throw away ones close to the end
    valid = (all_durations >= min_duration) \
            & (all_durations < max_duration) \
            & (all_starts > pad)

    num_valid = np.sum(valid)
    all_mouse_inds = all_mouse_inds[valid]
    all_starts = all_starts[valid]
    all_durations = all_durations[valid]

    # Choose a random subset to show
    rng = npr.RandomState(seed)
    subset = rng.choice(num_valid,
                        size=min(num_valid, num_instances),
                        replace=False)

    all_mouse_inds = all_mouse_inds[subset]
    all_starts = all_starts[subset]
    all_durations = all_durations[subset]

    # Extract slices for each mouse
    slices = []
    for mouse in range(len(posteriors)):
        is_mouse = (all_mouse_inds == mouse)
        slices.append([slice(start, start + dur) for start, dur in
                       zip(all_starts[is_mouse], all_durations[is_mouse])])

    return slices


def sample_clips(clips, num_clips_to_sample, min_length=None):
    # print("The longest clip is ", max(len(clip) for clip in clips), "frames long.")
    if min_length is not None:
        filtered_clips = [clip for clip in clips if len(clip) >= min_length]
    
        if len(filtered_clips) < num_clips_to_sample:
                # print("Not enough clips long enough. Returning longest clips.")
                return sorted(clips, key=len, reverse=True)[:num_clips_to_sample]
    else:
        filtered_clips = clips

    selected_indices = np.random.choice(len(filtered_clips), num_clips_to_sample, replace=False)
    return [filtered_clips[i] for i in selected_indices]

def get_fps(time_match):
    timestamps = (time_match.iloc[:, 1]*1000).round().astype(int).tolist()

    # Calculate average FPS
    if len(timestamps) < 2:
        fps = 20  # Default assumption
    else:
        durations = [timestamps[i+1]/1000 - timestamps[i]/1000 for i in range(len(timestamps)-1)]
        avg_duration = sum(durations) / len(durations)
        fps = 1.0 / avg_duration
    print("Average FPS: ", fps)
    return fps

def get_state_clips(slices, train_dataset, time_match, buffer=0.0):
    clips = []
    for s in slices[0]:
        time_range = train_dataset[0]['times'][s]
        start_time_stamp = time_range[0].item() - buffer
        end_time_stamp = time_range[-1].item() + 20.0 + buffer # Time stamps are beginning of chunk, so add 20 ms to include the last frame
        
        # Get chunk of time_match that falls within the time range
        time_match_chunk = time_match[(time_match['LocalTimestamp_sec'] >= start_time_stamp / 1000) & 
               (time_match['LocalTimestamp_sec'] <= end_time_stamp / 1000)]
        
        try:
            start_frame = time_match_chunk['FrameIndex'].iloc[0]
            end_frame = time_match_chunk['FrameIndex'].iloc[-1]
        except IndexError:
            print(f"Warning: No frames found for time range {start_time_stamp} to {end_time_stamp}. Skipping this slice.")
            continue

        frames = extract_frames(start_frame, end_frame, scale=0.25)

        if len(frames) >= 1: # To prevent empty clips
            clips.append(frames)
    
    return clips

def extract_frames(start_frame, 
                   end_frame, 
                   path="data/250509_seizure_I7_1.avi",
                   scale=0.1,
                   max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    # Validate frame range
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or end_frame >= total_frames:
        raise ValueError(f"Frame range must be within 0-{total_frames-1}.")
    
    # Get original and downsampled resolution
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(height * scale)
    new_width = int(width * scale)

    num_frames = end_frame - start_frame + 1
    #print(f"Attempting to extract {num_frames} frames...")
    if max_frames is not None:
        print(f"Limiting extraction to {max_frames} frames.")
        num_frames = min(num_frames, max_frames)

    # Pre-allocate array for smaller frames
    frames = np.empty((num_frames, new_height, new_width, 3), dtype=np.uint8)
    
    # Jump to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames efficiently
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break  # Handle unexpected EOF

        # Resize before storing
        frame_small = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frames[i] = frame_small
    
    cap.release()
    return frames[:i]
