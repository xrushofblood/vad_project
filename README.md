# XR Video Anomaly Detection for Industrial Purposes

Master's thesis project (Computer Science, Sapienza University of Rome).
The goal is to monitor an assembly procedure recorded with an XR headset (egocentric video) and to detect anomalies such as wrong or missing steps. Mr Potato Head is used as a proxy for an industrial assembly: 10 pieces, a clear procedure, and fully controllable recordings.

## Pipeline overview

| Phase | Task | Status |
|---|---|---|
| 1. Perception | Recognize the pieces in each frame | Pipeline (`notebooks/phase_1_perception.ipynb`) and support-set study (`notebooks/phase_1_support_set_study.ipynb`) done; margin study next |
| 2. State tracking | Track which pieces are mounted and in which order | Planned |
| 3. Reasoning | Check the sequence against the assembly rules (`configs/dependency_graph.json`) with an LLM (Bonsai-27B) | Planned |

### Phase 1 — Perception

No model is trained. C-RADIOv4-SO400M is used as a frozen feature extractor: each frame becomes a grid of 16x16 patches, each described by a 1152-dimensional vector. Each class (10 pieces, background and hand) is represented by a prototype, the normalized mean of its annotated patches in 8 reference frames, chosen in the support-set study. Each patch of a new frame takes the class of the most similar prototype (cosine similarity), with a margin that rejects uncertain pieces to background. In the video pipeline, perception runs only when the scene is still.

The notebook documents every design choice, with observations, references and sensitivity checks (Appendix A). Its last section lists the current limitations and next steps.

Evaluation protocol. Frames are split by recording: reference frames come from sequences never used for evaluation. Validation (`assembly_2`, `1_disassembled_controlled_caos`, 8 annotated frames) is used to inspect errors and choose the method. Test (`assembly_1`) is held out until the end of all studies; it was inspected once, in the qualitative evaluation of `v1-baseline`. The pipeline notebook reads only validation frames.

## Repository structure

```
vad-project/
├── configs/
│   └── dependency_graph.json       # Assembly rules for the reasoning phase
├── data/                           # Not tracked
│   ├── raw_videos/                 # Videos recorded with the headset
│   └── extracted_frames/           # Extracted frames (.jpg) and labelme annotations (.json)
├── models/                         # Not tracked
│   ├── c_radio_weights/            # C-RADIOv4-SO400M (weights and model code, Hugging Face)
│   └── bonsai_27b_1bit/            # Bonsai-27B, 1-bit quantized (reasoning phase)
├── notebooks/
│   ├── phase_1_perception.ipynb          # Phase 1 pipeline
│   ├── phase_1_support_set_study.ipynb   # Which and how many reference frames
│   └── _archive/                         # Not tracked
├── outputs/                              # Not tracked: prototypes, similarities, figures, videos
│   ├── phase_1_perception/
│   ├── phase_1_support_set_study/
│   └── baseline/                         # v1-baseline prototypes and similarities, for regression checks
├── src/
│   ├── perception/                       # Shared code, imported by every Phase 1 notebook
│   │   ├── __init__.py
│   │   ├── classes.py                    # Class list and colors
│   │   ├── features.py                   # C-RADIOv4 feature extraction
│   │   ├── labels.py                     # labelme polygons -> patch labels
│   │   ├── prototypes.py                 # Prototypes and classification
│   │   ├── metrics.py                    # Confusion matrix, per-class IoU, mIoU
│   │   └── visualization.py              # Label maps and overlays
│   └── utils/
│       └── extract_frames.py             # Frame extraction from the videos
├── archive/                        # Not tracked
├── requirements.txt
└── README.md
```

`archive/` and the `_archive/` folders contain old material (notebooks, outputs, logs, documents). It is kept only locally, so the repository shows only the active work.

## Setup

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.1), Transformers 5.17.0, on an NVIDIA GeForce RTX 4060 Laptop GPU.

```bash
python -m venv venv
venv\Scripts\activate            # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

### Models

- **C-RADIOv4-SO400M**: [nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M), downloaded into `models/c_radio_weights/`. The model is loaded with `trust_remote_code=True`: the model code comes with the weights.
- **Bonsai-27B (1-bit, GGUF)**: [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf), downloaded into `models/bonsai_27b_1bit/`.

### Data

The Mr Potato Head dataset was recorded for this project with an XR headset (1024x1024, 30 fps). Videos and frames are not in the repository yet.

Frames are extracted with:

```bash
python src/utils/extract_frames.py
```

The script reads every `.mp4` in `data/raw_videos/` and saves **one frame per second** in `data/extracted_frames/`, named `<video>_f<index>.jpg`: the index is the second of the video (e.g. `assembly_1_f0016.jpg` is at 16 s). Videos whose frames already exist are skipped. The prefix before `_f` identifies the recording, ...which is what the notebooks use to split reference, validation and test frames by sequence.

`frame_1.jpeg` is an independent frame taken with an iPhone (wide angle), used only as a qualitative check of generalization to a different camera (pipeline, Section 10.4).

Reference and validation frames are annotated with [labelme](https://github.com/wkentaro/labelme); the `.json` annotations are saved next to the frames.

## Usage

Open `notebooks/phase_1_perception.ipynb` and run all cells (Restart & Run All). The notebook reads data from `data/` and the model from `models/`, and writes prototypes, validation similarities, figures and the annotated video to `outputs/phase_1_perception/`.

The support-set study (`notebooks/phase_1_support_set_study.ipynb`) uses the same shared code and writes to `outputs/phase_1_support_set_study/`. It never reads test frames.


## Versions

- `v1-baseline`: first clean, documented version of the Phase 1 pipeline.
- `v1.1-shared-module`: shared code moved to `src/perception/`, verified bitwise identical to `v1-baseline`.
- `v1.2-support-set`: support set extended to 8 frames (5_assembled_f0019 added), as selected in the support-set study.

## Next steps

- Margin study: the background margin, measured on the annotated validation frames.
- Final evaluation on the test sequence (`assembly_1`), once, after all studies.
- Continuous perception (latency and resolution).
- Phase 2: state tracking.