#  HaMeR + MediaPipe — Hand Pose & TSV Extraction

Extract **3D hand pose**, **MANO parameters**, **meshes**, and **Task Space Vectors (TSVs)** from RGB images using [HaMeR](https://github.com/geopavlakos/hamer) and [MediaPipe Hands](https://mediapipe.dev/).

---

## Pipeline

1. **MediaPipe Hands** detects hands in the image and produces 2D bounding boxes + handedness (left/right).
2. **HaMeR** takes each cropped hand region and predicts:
   - MANO parametric hand model coefficients
   - 3D joint positions (21 keypoints)
   - 3D mesh vertices
3. **Task Space Vectors** are computed from the 3D joints — displacement vectors from the palm to each fingertip.

---

## Outputs

For each detected hand in an image, the script saves:

| Output | Filename | Description |
|---|---|---|
| MANO parameters | `{image}_{id}_mano.npy` | Dictionary with `global_orient`, `hand_pose`, `betas`, `cam_t`, `is_right` |
| Task Space Vectors | `{image}_{id}_tsv.npy` | 5 × 3 NumPy array (5 fingertip vectors relative to palm) |
| 3D Mesh | `{image}_{id}.obj` | Triangulated hand mesh *(optional, requires `--save_mesh`)* |
| Overlay | `{image}_overlay.png` | Predicted mesh rendered on top of the original RGB image |

---

## Task Space Vectors (TSVs)

The TSV representation is adopted from the **[DexMV](https://yzqin.github.io/dexmv/)** paper (*DexMV: Imitation Learning for Dexterous Manipulation from Human Videos*). TSVs are 3D displacement vectors from the palm (wrist joint) to each fingertip, producing a compact **5 × 3** matrix per hand that captures the finger configuration in a pose-invariant way. This is the same representation used in DexMV to transfer human hand poses to dexterous robot hands.

---

## Usage

```bash
python hamer_get_tsv.py \
    --img_folder path/to/images \
    --out_folder demo_out \
    --save_mesh \
    --rescale_factor 2.0
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--img_folder` | `str` | *(required)* | Path to an image file or folder of `.jpg`/`.png` images |
| `--out_folder` | `str` | `demo_out` | Directory where all outputs are saved |
| `--checkpoint` | `str` | HaMeR default | Path to a custom HaMeR model checkpoint |
| `--save_mesh` | flag | `False` | Save `.obj` mesh files for each detected hand |
| `--rescale_factor` | `float` | `2.0` | Bounding box rescale factor passed to ViTDetDataset |

---

## Requirements

- Python ≥ 3.8
- PyTorch
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- tqdm
- [HaMeR](https://github.com/geopavlakos/hamer) (must be installed and importable)

```bash
pip install torch opencv-python mediapipe numpy tqdm
# Install HaMeR following https://github.com/geopavlakos/hamer
```

---

## Example

```bash
# Single image
python hamer_get_tsv.py --img_folder photo.jpg --out_folder results

# Folder of images with mesh export
python hamer_get_tsv.py --img_folder ./hand_images/ --out_folder results --save_mesh
```

### Sample Result

![Example overlay output](Screenshot%20from%202026-03-07%2012-07-53.png)

Output structure:

```
results/
├── photo_0_mano.npy
├── photo_0_tsv.npy
├── photo_0.obj          # only with --save_mesh
└── photo_overlay.png
```

---

## License

MIT — see [LICENSE](../LICENSE) for details.

---

## References

- **[HaMeR](https://github.com/geopavlakos/hamer)** — *Reconstructing Hands in 3D with Transformers* (Pavlakos et al.)
- **[DexMV](https://yzqin.github.io/dexmv/)** — *DexMV: Imitation Learning for Dexterous Manipulation from Human Videos* (Qin et al., CVPR 2022)
