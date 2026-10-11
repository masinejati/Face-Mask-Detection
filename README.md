# Face Mask Detection — MobileNetV2 (Academic Project, 2024)

**Contributors:** Masoumeh Nejati Moharrami and Parsa Khazraei  
**Context:** University computer-vision project, Summer 2024

A binary **image classifier** to distinguish face images labeled `with_mask` and `without_mask` using transfer learning from ImageNet-pretrained MobileNetV2 and TensorFlow/Keras. The original project involved approximately 4,000 images according to the student's project account; the original dataset is not included here.

## Historical result (not independently reproduced)

The original course presentation reports approximately **98% validation accuracy**. This is a historical project claim, **not** a result verified by the code in this repository; the original trained model, notebook history, evaluation logs, and dataset have not been recovered. Any future result from `src/train.py` must be reported separately.

## What's original and what's new?

| Path | Provenance |
| --- | --- |
| `original/recovered_github_script.py` | Python script recovered from the contributor's GitHub and preserved **as pasted**, including its syntax error and Colab-specific dataset path. This script is archival and will not run as-is. |
| `src/train.py` | **Later corrected / enhanced implementation**, adapted from the recovered script. Not claimed to be the exact 2024 program. |
| `docs/FaceMaskDetection-Presentation-2024.pdf` | Original project presentation. |
| `docs/Technical-Explanation.pdf` | Original technical explanation with code fragments. |
| `docs/Architecture-Study-Flowchart.pdf` | Separate analysis of a five-convolution-layer CNN described in a paper, **not** the MobileNetV2 architecture implemented here. |

## Original approach

- Binary labels: `with_mask` → 0, `without_mask` → 1.
- Resize RGB images to 224 × 224 and apply MobileNetV2 `preprocess_input`.
- Stratified 80/20 training/evaluation split (`random_state=42`).
- Training augmentation: rotation, zoom, shifts, shear, and horizontal/vertical flips in the original script.
- Frozen MobileNetV2 base → global average pooling → sigmoid output.
- Adam optimizer and binary cross-entropy loss; original recovered code specifies 10 training epochs.

**Scope:** This is a classifier for face images. It does not by itself detect face bounding boxes in larger photographs. Some course materials discuss face detection, but that inference pipeline is not present in the recovered Python script.

## Running the later updated implementation

Requires a compatible Python environment with TensorFlow, scikit-learn, Pillow, Matplotlib, and NumPy. Dependency versions are not pinned to the original 2024 environment, so exact compatibility will vary.

```bash
pip install -r requirements.txt
python src/train.py --data path/to/dataset --epochs 10 --output results
```

Expected local dataset folders (not provided):

```text
dataset/
  with_mask/
    picture1.jpg
  without_mask/
    picture2.jpg
```

`results/` receives a local Keras model, `metrics.json`, and accuracy plot. Outputs are ignored by Git because they are generated artifacts. **This updated implementation has not been trained or validated during package preparation**; its correctness and results require a real dataset and a compatible TensorFlow installation.

### Evaluation caution

The updated example uses the 20% holdout as the `validation_data` during training and reports metrics on the same portion afterward. These are **holdout/validation metrics, not an independent untouched final test-set evaluation**. To make strong generalization claims, use a separate unseen test set and check dataset provenance, leakage, and class balance.

## Original coursework documentation

The presentation and technical explanation retain their historical wording, which may contain inconsistencies (including references to RGB/BGR preprocessing and different training epochs). The archived code and the later example are documented separately to avoid silently treating those differences as proven implementation details.

## Attribution and privacy

This was a group academic project. The repository does not assign individual coding contributions that cannot be established from the recovered materials. Only share datasets, faces, and third-party illustrations when you have the right to publish them. The source and docs are supplied for academic portfolio review, not as a production face-mask safety system.
