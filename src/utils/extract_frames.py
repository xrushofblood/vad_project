import cv2
import os
import glob

def batch_extract(source_dir, output_dir, fps_target=1):
    print(f"\n--- In progress: {source_dir} -> {output_dir} ---")
    
    # Create output directory if it doesn't exist 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all mp4 files in the specified folder 
    videos = glob.glob(os.path.join(source_dir, "*.mp4"))
    
    if not videos:
        print(f"Warning: no video found in {source_dir}")
        return

    for video_path in videos:
        video_name = os.path.basename(video_path).split('.')[0]
        
        # --- NEW: Check if frames for this video already exist ---
        existing_frames = glob.glob(os.path.join(output_dir, f"{video_name}_f*.jpg"))
        if existing_frames:
            print(f"Skipping '{video_name}': frames already extracted.")
            continue
        # ---------------------------------------------------------
        
        vidcap = cv2.VideoCapture(video_path)
        fps_source = round(vidcap.get(cv2.CAP_PROP_FPS))
        
        # Calculate every how many frames to extract to get the fps_target
        interval = max(1, fps_source // fps_target)
        
        count = 0
        saved = 0
        
        while True:
            success, image = vidcap.read()
            if not success: 
                break
            
            if count % interval == 0:
                # Save the frame
                cv2.imwrite(os.path.join(output_dir, f"{video_name}_f{saved:04d}.jpg"), image)
                saved += 1
            count += 1
            
        vidcap.release()
        print(f"Video '{video_name}': {saved} frames extracted.")

# --- EXECUTION ---
# Dynamically resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")

raw_videos_dir = os.path.join(project_root, "data", "raw_videos")
extracted_frames_dir = os.path.join(project_root, "data", "extracted_frames")

# Execute extraction
batch_extract(raw_videos_dir, extracted_frames_dir)

print("\nSuccess!")