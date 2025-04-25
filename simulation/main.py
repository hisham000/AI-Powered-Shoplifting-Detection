import random
import time
import os
import re
from pathlib import Path
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Added import for Kaggle dataset download
import kagglehub

# -------- CONFIG --------
SOURCE_DIR = Path("/data/raw")  # Folder with your video pool
DEST_DIR = Path("/CCTV")  # Folder where CCTV-style folders go
NUM_CAMERAS = 4
CHUNK_DURATION = 5  # seconds
# ------------------------


def download_and_prepare_dataset():
    """Download dataset from Kaggle and prepare the video files if data/raw is empty"""
    print("📥 Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mateohervas/dcsass-dataset")
    
    print("🔄 Processing downloaded videos...")
    # Get all video files from the downloaded dataset
    shoplifting_0_dir = Path(f"{path}/DCSASS Dataset/Shoplifting/0")
    shoplifting_1_dir = Path(f"{path}/DCSASS Dataset/Shoplifting/1")
    
    # Create raw directory if it doesn't exist
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store clips with the same ID
    video_groups = {}
    
    # Process videos from both directories
    for directory in [shoplifting_0_dir, shoplifting_1_dir]:
        if not directory.exists():
            continue
            
        for video_file in directory.glob("*.mp4"):
            # Extract the pattern Shoplifting{i:.3f}x264{j}.mp4
            match = re.match(r'Shoplifting(\d{3})x264\d+\.mp4', video_file.name)
            if match:
                shoplifting_id = match.group(1)
                if shoplifting_id not in video_groups:
                    video_groups[shoplifting_id] = []
                video_groups[shoplifting_id].append(video_file)
    
    # Concatenate videos with the same ID
    for shoplifting_id, video_files in video_groups.items():
        if not video_files:
            continue
        
        output_path = SOURCE_DIR / f"Shoplifting{shoplifting_id}.mp4"
        
        # Skip if already processed
        if output_path.exists():
            continue
            
        try:
            # Sort files to ensure proper ordering
            video_files.sort(key=lambda x: str(x))
            
            # Load clips
            clips = []
            for video_file in video_files:
                try:
                    clip = VideoFileClip(str(video_file))
                    clips.append(clip)
                except Exception as e:
                    print(f"Error loading {video_file}: {e}")
            
            if clips:
                # Concatenate clips
                final_clip = concatenate_videoclips(clips)
                # Save concatenated clip
                final_clip.write_videofile(str(output_path), codec="libx264", 
                                          audio_codec="aac", verbose=False, logger=None)
                # Close clips to free resources
                final_clip.close()
                for clip in clips:
                    clip.close()
                    
                print(f"✅ Created {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing Shoplifting{shoplifting_id}: {e}")
    
    print("✅ Dataset preparation complete")


def get_random_clip_segment(video_path, duration):
    try:
        clip = VideoFileClip(str(video_path))
        if clip.duration < duration:
            return None
        start = random.randint(0, int(clip.duration) - duration)
        return clip.subclip(start, start + duration)
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None


def save_chunk(chunk, cam_folder, base_filename):
    output_path = cam_folder / f"{base_filename}.mp4"
    chunk.write_videofile(
        str(output_path), codec="libx264", audio_codec="aac", verbose=False, logger=None
    )


def simulate_camera_feeds():
    # Check if source directory is empty and download dataset if needed
    if not SOURCE_DIR.exists() or not any(SOURCE_DIR.iterdir()):
        print("📁 Source directory is empty. Downloading dataset from Kaggle...")
        download_and_prepare_dataset()
    
    video_pool = list(SOURCE_DIR.glob("*.mp4"))
    if not video_pool:
        print("⚠️ No videos found in source folder even after download attempt.")
        return

    print("📹 Starting real-time simulation... (press Ctrl+C to stop)")

    try:
        while True:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            for cam in range(1, NUM_CAMERAS + 1):
                cam_name = f"cam{cam:02d}"
                cam_folder = DEST_DIR / cam_name / date_str / hour_str
                cam_folder.mkdir(parents=True, exist_ok=True)

                video_path = random.choice(video_pool)
                chunk = get_random_clip_segment(video_path, CHUNK_DURATION)

                if chunk:
                    base_filename = f"{timestamp}_{cam_name}"
                    save_chunk(chunk, cam_folder, base_filename)

            # Wait exactly CHUNK_DURATION seconds before next iteration
            time.sleep(CHUNK_DURATION)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


if __name__ == "__main__":
    simulate_camera_feeds()
