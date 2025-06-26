from datasets import load_dataset, Image
import numpy as np
import os

def prepare_data():
    # ------------------------------------------------------------------
    # 1.  Load the dataset (train + test) ------------------------------
    # ------------------------------------------------------------------
    ds = load_dataset("uoft-cs/cifar10")          # splits: train / test

    # ------------------------------------------------------------------
    # 2.  Decode → NumPy ------------------------------------------------
    #     A. cast_column(..., Image(decode=True)) tells 🤗 to return PIL.Image
    #     B. with_format("numpy") turns every PIL image into np.ndarray
    # ------------------------------------------------------------------
    def split_to_numpy(dset_split):
        split = (
            dset_split
            .cast_column("img", Image(decode=True))   # ensure decoded PIL
            .with_format("numpy", columns=["img", "label"])
        )
        images = np.stack(split["img"])              # (N, 32, 32, 3) uint8
        labels = np.array(split["label"], dtype=np.int64)
        return images, labels

    train_imgs, train_labels = split_to_numpy(ds["train"])
    test_imgs,  test_labels  = split_to_numpy(ds["test"])

    print("train:", train_imgs.shape, train_labels.shape)
    print("test :",  test_imgs.shape,  test_labels.shape)
    # → train: (50000, 32, 32, 3) (50000,)   test: (10000, 32, 32, 3) (10000,)

    # ------------------------------------------------------------------
    # 3.  Save as a single NPZ -----------------------------------------
    #     • savez_compressed uses zlib; ~170 MB → ~65 MB on disk
    # ------------------------------------------------------------------
    out_path = "cifar10_uint8.npz"
    np.savez_compressed(
        out_path,
        train_images=train_imgs,
        train_labels=train_labels,
        test_images=test_imgs,
        test_labels=test_labels,
    )
    print(f"Wrote {out_path!s}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

# ------------------------------------------------------------------
# 4.  Loading back later -------------------------------------------
#     Everything comes back as NumPy, ready for JAX:
# ------------------------------------------------------------------
# data = np.load("cifar10_uint8.npz")
# X_train = data["train_images"]   # jnp.asarray(...) if you want JAX
# y_train = data["train_labels"]

def load_data():
    # train_images = jnp.asarray(data["train_images"])  # shape (50000,32,32,3), dtype uint8
    # train_labels = jnp.asarray(data["train_labels"])  # shape (50000,), dtype int64
    # test_images  = jnp.asarray(data["test_images"])   # shape (10000,32,32,3)
    # test_labels  = jnp.asarray(data["test_labels"])   # shape (10000,)
    data = np.load("cifar10_uint8.npz")

    return data

