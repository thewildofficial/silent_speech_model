import os
import sys
import glob
import pickle
from pathlib import Path
from typing import Optional

from absl import app, flags
from scipy.io import savemat
from tqdm import tqdm

# Ensure repo modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from read_emg import EMGDataset  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string("save_folder", None, "Destination for processed mat files")
flags.DEFINE_integer("print_every", 1000, "Progress print frequency (set 0 to silence)")


def build_partition(fname: str, dev: bool, test: bool, normalizers_file: Optional[str]):
    """Build and cache a dataset partition."""
    dataset = EMGDataset(
        dev=dev,
        test=test,
        returnRaw=True,
        normalizers_file=normalizers_file,
    )
    with open(fname, "wb") as f:
        pickle.dump(dataset, f)


def save_partition(partition, folder: str, save_folder: str, print_every: int):
    for i, x in enumerate(tqdm(partition)):
        session = str(x["session_ids"][0].item())
        file_id = str(x["file_label"])
        save_dir = Path(save_folder) / folder / f"session_{session}"
        save_fname = save_dir / f"{file_id}.mat"
        save_dir.mkdir(parents=True, exist_ok=True)
        if not save_fname.exists():
            x_np = {}
            for key, val in x.items():
                try:
                    x_np[key] = val.numpy()
                except Exception:
                    x_np[key] = val
            savemat(save_fname, x_np)
        if print_every and (i % print_every == 0):
            pass


def main(argv):
    del argv
    save_folder = FLAGS.save_folder
    print_every = FLAGS.print_every
    normalizers_file = os.environ.get("NORMALIZERS_FILE", "normalizers.pkl")

    if not save_folder:
        raise ValueError("--save_folder is required")

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    fnames = ["trainset2.pkl", "devset2.pkl", "testset2.pkl"]
    partitions = [(False, False), (True, False), (False, True)]  # (dev, test)

    # Build and cache pkl partitions
    for fname, (dev, test) in zip(fnames, partitions):
        print(f"Building {fname} (dev={dev}, test={test})")
        build_partition(fname, dev, test, normalizers_file)

    # Load pkl partitions
    with open("trainset2.pkl", "rb") as f:
        trainset = pickle.load(f)
    with open("devset2.pkl", "rb") as f:
        devset = pickle.load(f)
    with open("testset2.pkl", "rb") as f:
        testset = pickle.load(f)

    # Save mats
    for partition, folder in zip([trainset, devset, testset], ["train", "dev", "test"]):
        print(f"===== {folder} =====")
        save_partition(partition, folder, save_folder, print_every)

    total = len(glob.glob(os.path.join(save_folder, "*/*/*.mat")))
    print("Total mats:", total)


if __name__ == "__main__":
    app.run(main)

