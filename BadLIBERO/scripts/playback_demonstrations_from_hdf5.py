import h5py
import cv2
import numpy as np
import os
import imageio
import argparse

def save_rollout_video(rollout_images, output_dir, demo_id):
    """Saves an MP4 replay of an episode."""
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, f"{demo_id}.mp4")

    writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    print(f"✅ Saved rollout video to: {mp4_path}")
    return mp4_path

def process_hdf5_file(file_path, hdf5_root_dir, output_root_dir):
    rel_path = os.path.relpath(file_path, hdf5_root_dir)
    rel_dir = os.path.dirname(rel_path)
    task_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"▶️ Processing file: {file_path}...")

    with h5py.File(file_path, "r") as f:
        demos = [d for d in f["data"].keys() if d.startswith("demo_")]
        print(f"   Found {len(demos)} demos.")

        for demo_id in demos:
            frames = f["data"][demo_id]["obs"]["agentview_rgb"][:]
            processed_frames = []

            for frame in frames:
                frame = np.flipud(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
                processed_frames.append(frame)

            output_dir = os.path.join(output_root_dir, rel_dir, task_name)
            save_rollout_video(processed_frames, output_dir, demo_id)

def main(args):
    
    input_path = args.input_path

    if os.path.isfile(input_path) and input_path.endswith(".hdf5"):
        hdf5_root_dir = os.path.dirname(input_path)
        if args.output_dir:
            output_root_dir = args.output_dir
        else:
            task_dir_name = os.path.basename(hdf5_root_dir.rstrip("/"))
            output_root_dir = os.path.join("tmp_demo", task_dir_name)
        process_hdf5_file(input_path, hdf5_root_dir, output_root_dir)

    elif os.path.isdir(input_path):
        hdf5_root_dir = input_path
        if args.output_dir:
            output_root_dir = args.output_dir
        else:
            task_dir_name = os.path.basename(hdf5_root_dir.rstrip("/"))
            output_root_dir = os.path.join("tmp_demo", task_dir_name)

        for root, _, files in os.walk(hdf5_root_dir):
            for filename in files:
                if filename.endswith(".hdf5"):
                    file_path = os.path.join(root, filename)
                    process_hdf5_file(file_path, hdf5_root_dir, output_root_dir)
    else:
        print("❌ 输入路径必须是一个 .hdf5 文件或包含 .hdf5 文件的文件夹。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 demos to mp4 videos, preserving directory structure.")
    parser.add_argument('--input_path', type=str, 
                        default="../new7/knife_checking3",
                        help='Path to a single .hdf5 file or a directory containing .hdf5 files')
    parser.add_argument('--output_dir', type=str, default="./demo",
                        help='Output root directory. Default: tmp_demo/<top_level_folder_of_input_path>')
    args = parser.parse_args()
    main(args)
