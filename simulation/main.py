import random
import re
import time
from datetime import datetime
from pathlib import Path

# Added import for Kaggle dataset download
import kagglehub
from moviepy.editor import VideoFileClip, concatenate_videoclips

# -------- CONFIG --------
SOURCE_DIR = Path("/data/raw")  # Folder with your video pool
DEST_DIR = Path("/CCTV")  # Folder where CCTV-style folders go
NUM_CAMERAS = 4
CHUNK_DURATION = 5  # seconds
# ------------------------


def download_and_prepare_dataset():
    # Download the dataset
    print("⬇️ Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

    # Create destination directory if it doesn't exist
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # Get the path to shoplifting videos
    shoplifting_path = Path(dataset_path) / "DCSASS Dataset" / "Shoplifting"

    if not shoplifting_path.exists():
        print(f"❌ Path not found: {shoplifting_path}")
        return

    print("🔄 Processing videos...")

    # Iterate through all the directories matching the pattern Shoplifting*_x264.mp4
    for dir_path in shoplifting_path.glob("Shoplifting*_x264.mp4"):
        # Extract the index from directory name
        match = re.search(
            r"Shoplifting(\d+(?:\.\d+)?)_x264\.mp4", Path(str(dir_path)).name
        )
        if not match:
            continue

        index = match.group(1)
        output_filename = f"Shoplifting{index}.mp4"

        # Get all the video files in this directory
        video_files = list(dir_path.glob(f"Shoplifting{index}_x264_*.mp4"))

        if not video_files:
            print(f"⚠️ No video files found in {dir_path}")
            continue

        print(f"🎬 Combining videos for {dir_path.name}...")

        try:
            # Sort videos by their index j
            video_files.sort(
                key=lambda x: int(re.search(r"_(\d+)\.mp4$", x.name).group(1))
            )

            # Load all video clips
            clips = []
            for video_file in video_files:
                clip = VideoFileClip(str(video_file))
                clips.append(clip)

            # Concatenate all clips
            final_clip = concatenate_videoclips(clips)

            # Write the concatenated clip to the output directory
            output_path = SOURCE_DIR / output_filename
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac" if final_clip.audio else None,
                verbose=False,
                logger=None,
            )

            # Close all clips to free up resources
            for clip in clips:
                clip.close()
            final_clip.close()

            print(f"✅ Created {output_path}")

        except Exception as e:
            print(f"❌ Error processing {dir_path}: {e}")

    print("🎉 Dataset preparation complete!")


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
