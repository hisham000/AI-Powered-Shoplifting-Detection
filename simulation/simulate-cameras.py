import random
import time
from pathlib import Path
from datetime import datetime
from moviepy.editor import VideoFileClip

# -------- CONFIG --------
SOURCE_DIR = Path("./data")  # Folder with your video pool
DEST_DIR = Path("./CCTV")  # Folder where CCTV-style folders go
NUM_CAMERAS = 4
CHUNK_DURATION = 5  # seconds
# ------------------------


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
    video_pool = list(SOURCE_DIR.glob("*.mp4"))
    if not video_pool:
        print("⚠️ No videos found in source folder.")
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
