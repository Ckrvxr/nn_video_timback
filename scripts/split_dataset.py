import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--src", type=str, default="data/it2017", help="Source dataset directory")
    parser.add_argument("--val", type=str, default="data/it2017_val", help="Validation dataset directory")
    parser.add_argument("--num-val", type=int, default=5, help="Number of validation segments to split")
    args = parser.parse_args()

    src_dir = args.src
    val_dir = args.val
    num_val = args.num_val

    if not os.path.exists(src_dir):
        print(f"Error: source directory {src_dir} does not exist.")
        return

    os.makedirs(val_dir, exist_ok=True)

    # Get all subdirectories in src_dir
    dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    dirs.sort()

    print(f"Total directories found in {src_dir}: {len(dirs)}")

    if len(dirs) < num_val:
        # Check if they are already in val_dir (e.g., if already run)
        val_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if len(val_dirs) == num_val:
            print(f"Validation dataset already split and contains {num_val} segments:")
            for d in val_dirs:
                print(f" - {d}")
            return
        else:
            print(f"Error: Not enough directories to split. Found {len(dirs)} in src and {len(val_dirs)} in val.")
            return

    # Select num_val directories uniformly spaced out
    indices = [int(i * (len(dirs) - 1) / (num_val - 1)) for i in range(num_val)]
    print(f"Selected indices for validation: {indices}")
    selected_dirs = [dirs[i] for i in indices]

    for d in selected_dirs:
        src_path = os.path.join(src_dir, d)
        dst_path = os.path.join(val_dir, d)
        print(f"Moving {d} -> {val_dir}")
        shutil.move(src_path, dst_path)

    print("Split completed successfully!")

if __name__ == "__main__":
    main()
