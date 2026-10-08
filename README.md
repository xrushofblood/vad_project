# XR Video Anomaly Detection for Industrial Purposes

Master's thesis project (Computer Science, Sapienza University of Rome).
The goal is to monitor an assembly procedure recorded with an XR headset (egocentric video) and to detect anomalies such as wrong or missing steps. Mr Potato Head is used as a proxy for an industrial assembly: 10 pieces, a clear procedure, and fully controllable recordings.

## Pipeline overview

| Phase | Task | Status |
|---|---|---|
| 1. Perception | Recognize the pieces in each frame | Pipeline and four studies done (support set, resolution, multi-prototype, margin). Decided configuration: V2 with margin 0. Next: new support-set study on V2, then pipeline update |
| 2. State tracking | Track the state of each piece (on the table, in hand, mounted) and the order of picks and mounts | Planned |
| 3. Reasoning | Check picks and mounts against the assembly rules (`configs/dependency_graph.json`) with an LLM (Bonsai-27B): a wrong pick can be flagged before the mistake is made, a wrong mount when it happens | Planned |

### Phase 1 — Perception

No model is trained. C-RADIOv4-SO400M is used as a frozen feature extractor: each frame becomes a grid of 16x16 patches, each described by a 1152-dimensional vector. The classes are the 10 pieces, background and hand. Prototypes come from the annotated patches of a few reference frames (the support set). Each patch of a new frame takes the class of its most similar prototype (cosine similarity). A background margin can then send to background a patch whose best class beats background by less than the margin; it decides only between background and the other classes, never between two pieces. Frames enter the model at their native resolution, 1024x1024.

Two classifiers are compared in the studies. V0 has one prototype per class, the normalized mean of its annotated patches. V2 uses every annotated patch as a prototype (nearest neighbour); the similarity of a class is the highest similarity among its prototypes.

Current state. The pipeline notebook (`phase_1_perception.ipynb`) still runs V0 with 8 reference frames and margin 0.03; in its video pipeline perception runs only when the scene is still (about 400 ms per frame on the laptop GPU, almost all in C-RADIO). The studies chose V2, with a support set of 10 frames and margin 0 (`outputs/phase_1_margin_study/decision.json`). The pipeline will be updated after the next support-set study, since the margin must be chosen again for a new support set.

What the studies found:

- Support set (`phase_1_support_set_study.ipynb`, V0): candidate frames were added one at a time to the 7 initial reference frames and kept only if they improved validation. Only 5_assembled_f0019 entered (mIoU with the arms merged from 0.667 to 0.709). A frame aimed at a rare appearance brings few patches and barely moves a mean prototype.
- Resolution (`phase_1_resolution_study.ipynb`): the input stays at 1024. At 512 the prototypes of the small pieces rest on one or two patches, even perfect recognition on the 512 grid would not exceed what the method reaches at 1024, and the mIoU drops by about a quarter. Perception would be about four times faster (about 100 ms per frame instead of about 410).
- Multi-prototype (`phase_1_multi_prototype_study.ipynb`): two frames were added by a coverage rule (the loose mouth seen from the side, the body on the base with the empty mouth slot), and V0, V1 (one prototype per class and aspect, mounted or loose) and V2 were compared on the same 10 frames, at margin 0.03. V2 was chosen: mIoU with the arms merged 0.861 against 0.725 for V0 (+0.135, interval [0.093, 0.172]); V1 was worse than V0 (−0.064). V2 removes the mouth predicted on the empty slot. The Addendum runs perception on every frame of `assembly_2` for the first time.
- Margin (`phase_1_margin_study.ipynb`): the margin is chosen on the reference frames alone, by cross-validation leaving out one sequence at a time; the validation frames only check the choice. V0 needs 0.155 (mIoU from 0.725 to 0.824), V2 needs none (0). At these margins V2 is still better than V0, by +0.040 (interval [0.005, 0.069]). No margin recovers the loose pieces that V2 misses in the video.
- Exploratory test (`scratch_support_swap.ipynb`): with the two frames of `1_disassembled_controlled_caos` added to the support set of V2, the loose arm_2 is seen on the table before its pick in 44% of the frames instead of 12%, and the hat in 67% instead of 54%. For the loose pieces, the lever is which frames are in the support set. These frames are validation frames, so the test is optimistic.

Known limits of V2 at margin 0, from the video of `assembly_2`: the loose left arm (arm_2), swapped with the other arm in part of the video and lost while the hands work next to it; the hat on the table and in the hand; the loose mouth seen from the side, whose peg is predicted as nose (the mouth is recognized when its front is visible); a few hat patches on the cavity of the base alone on the table, and false positives on the watch at the wrist; pieces predicted on objects never annotated. Mounted pieces and the complete figure are recognized.

Evaluation protocol. Frames are split by recording: reference frames come from sequences never used for evaluation. Validation (`assembly_2`, `1_disassembled_controlled_caos`, 8 annotated frames) is used to inspect errors and check choices; the video of `assembly_2` is used only for exploratory analyses, against an event log annotated on the raw video (`configs/assembly_2_events.json`). Test (`assembly_1`) is held out until the end of Phase 1; it was inspected once, in the qualitative evaluation of `v1-baseline`. No study notebook reads test frames.
The v1-baseline file outputs/baseline/test_similarities_v1-baseline.npz also contains the test frames from that inspection; the support-set study loads only its validation entries.

## Repository structure

```
vad-project/
├── configs/
│   ├── dependency_graph.json       # Assembly rules for the reasoning phase
│   └── assembly_2_events.json      # Event log of assembly_2 (pick, mount, hands off), annotated on the raw video
├── data/                           # Not tracked
│   ├── raw_videos/                 # Videos recorded with the headset
│   └── extracted_frames/           # Extracted frames (.jpg) and labelme annotations (.json)
├── models/                         # Not tracked
│   ├── c_radio_weights/            # C-RADIOv4-SO400M (weights and model code, Hugging Face)
│   └── bonsai_27b_1bit/            # Bonsai-27B, 1-bit quantized (reasoning phase)
├── notebooks/
│   ├── phase_1_perception.ipynb              # Phase 1 pipeline (V0, margin 0.03; to be updated)
│   ├── phase_1_support_set_study.ipynb       # Which and how many reference frames (V0)
│   ├── phase_1_resolution_study.ipynb        # Input resolution: 1024 vs 512
│   ├── phase_1_multi_prototype_study.ipynb   # V0, V1, V2 on a support set of 10 frames
│   ├── phase_1_margin_study.ipynb            # Background margin for V0 and V2
│   ├── scratch_support_swap.ipynb            # Exploratory: two more frames in the support set of V2
│   └── _archive/                             # Not tracked
├── outputs/                                  # Not tracked: prototypes, similarities, figures, videos
│   ├── phase_1_perception/
│   ├── phase_1_support_set_study/
│   ├── phase_1_resolution_study/
│   ├── phase_1_multi_prototype_study/
│   ├── phase_1_margin_study/                 # decision.json, per-frame gaps and labels of assembly_2, video
│   ├── scratch/                              # Exploratory runs (continuous perception, support swap)
│   └── baseline/                             # v1-baseline prototypes and similarities, for regression checks
├── src/
│   ├── perception/                           # Shared code, imported by every Phase 1 notebook
│   │   ├── __init__.py
│   │   ├── classes.py                        # Class list, indices and colors
│   │   ├── features.py                       # C-RADIOv4 feature extraction (optional input resolution)
│   │   ├── labels.py                         # labelme polygons -> patch labels
│   │   ├── prototypes.py                     # Prototypes (mean, per aspect, per patch), background margin, CPU and GPU similarities
│   │   ├── metrics.py                        # Confusion matrix, per-class IoU, mIoU, boundary ring, arms merged, bootstrap
│   │   └── visualization.py                  # Label maps and overlays
│   └── utils/
│       ├── extract_frames.py                 # Frame extraction from the videos
│       └── continuous_analysis.py            # Measures of continuous perception against the event log (exploratory)
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

`requirements.txt` installs PyTorch with CUDA 12.1 from the PyTorch package index (first line of the file). pywinpty is installed only on Windows.

V2 runs on the GPU (`class_similarities_gpu`), in about 70 to 95 ms per frame depending on the support set, but only when the GPU memory is free: with other notebooks or kernels open, Windows moves data to system memory and V2 slows down to several seconds per frame.

### Models

- **C-RADIOv4-SO400M**: [nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M), commit c0457f5, downloaded into `models/c_radio_weights/`. The model is loaded with `trust_remote_code=True`: the model code comes with the weights.
- **Bonsai-27B (1-bit, GGUF)**: [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf), downloaded into `models/bonsai_27b_1bit/`.

### Data

The Mr Potato Head dataset was recorded for this project with an XR headset (1024x1024, 30 fps). Videos and frames are not in the repository yet.

Frames are extracted with:

```bash
python src/utils/extract_frames.py
```

The script reads every `.mp4` in `data/raw_videos/` and saves **one frame per second** in `data/extracted_frames/`, named `<video>_f<index>.jpg`. The script keeps one frame every round(fps) frames, so at 30 fps `<video>_f<index>.jpg` is frame index × 30 of the video, at about second `index` (e.g. `assembly_1_f0016.jpg` is frame 480, at about 16 s). Videos whose frames already exist are skipped. The prefix before `_f` identifies the recording, which is what the notebooks use to split reference, validation and test frames by sequence.

`frame_1.jpeg` is an independent frame taken with an iPhone (wide angle), used only as a qualitative check of generalization to a different camera (pipeline, Section 10.4; multi-prototype study, Section 6.3).

Reference and validation frames are annotated with [labelme](https://github.com/wkentaro/labelme); the `.json` annotations are saved next to the frames.

## Usage

Every notebook is run with Restart & Run All, reads data from `data/` and the model from `models/`, and writes to its own folder in `outputs/`. No notebook reads test frames.

- `phase_1_perception.ipynb`: builds the prototypes, classifies the validation frames, and writes prototypes, similarities, figures and the annotated video to `outputs/phase_1_perception/`.
- `phase_1_support_set_study.ipynb`: selection of the reference frames for V0. Section 7 adds diagnostic analyses of the selected support set; they do not change the selection.
- `phase_1_resolution_study.ipynb`: 1024 and 512 on the validation frames (recognition and latency); 1024 is kept.
- `phase_1_multi_prototype_study.ipynb`: V0, V1 and V2 on the support set of 10 frames. The Addendum runs continuous perception on `assembly_2` (exploratory).
- `phase_1_margin_study.ipynb`: chooses the margin of V0 and V2 by cross-validation on the reference frames and checks it on the validation frames. Section 8 runs continuous perception on `assembly_2` (about 20 to 25 minutes with a free GPU) and analyses it with `src/utils/continuous_analysis.py`. The decision is saved in `outputs/phase_1_margin_study/decision.json`.
- `scratch_support_swap.ipynb`: exploratory test outside any study, V2 with the two frames of `1_disassembled_controlled_caos` added to the support set.

`src/utils/continuous_analysis.py` can also be run on its own on saved labels (see its docstring). Its windows and thresholds are provisional.

## Versions

- `v1-baseline`: first clean, documented version of the Phase 1 pipeline.
- `v1.1-shared-module`: shared code moved to `src/perception/`, verified bitwise identical to `v1-baseline`.
- `v1.2-support-set`: support set extended to 8 frames (5_assembled_f0019 added), as selected in the support-set study.
- `v1.3-review`: notebooks reviewed (markdowns rewritten, references verified against the papers), diagnostic analyses added to the support-set study (Section 7). The pipeline outputs are unchanged.
- `v1.4-resolution`: input resolution studied (1024 vs 512), 1024 kept. The feature extractor accepts an optional input resolution; arm merging, bootstrap and label upsampling moved to `src/perception/metrics.py`. Results at 1024 are bitwise unchanged.
- `v1.5-multi-prototype`: support set of 10 frames (coverage rule); V0, V1 and V2 compared, V2 chosen. Per-aspect and per-patch prototypes and GPU similarities added to `src/perception/prototypes.py`. Addendum on continuous perception.
- `v1.6-margin`: background margin studied for V0 and V2; V2 with margin 0 decided. `assign_labels` accepts negative margins (labels unchanged for margins of zero or above, checked). Event log of `assembly_2` and `src/utils/continuous_analysis.py` added. Exploratory test on the support set of V2.

## Next steps

- Rebuild the validation set by annotating more frames of `assembly_2`, so that it stays independent of a support set that includes `1_disassembled_controlled_caos`.
- New support-set study on V2 with one criterion: one frame for each state of each piece, as seen in use (including the mouth with its front visible, both loose arms, the base alone seen from above), removing redundant frames by cross-validation on the reference frames, without the validation frames.
- Choose the margin again with the procedure of the margin study, then update the pipeline to V2 with a regression check.
- Formal study of continuous perception, with windows and thresholds justified from the event log, for the three moments that matter in Phase 2: pieces still on the table, picks and mounts.
- Robustness set recorded with an iPhone in another environment, reported separately.
- Phase 2: state tracking of each piece (on the table, in hand, mounted) from the per-frame labels.
- Final evaluation on the test sequence (`assembly_1`), once, at the end of Phase 1.

Open questions: arm identity decided at object level, from the hand and the shape of the arm; several nearest neighbours (k > 1) instead of one, if a single atypical prototype still misleads V2 after the new support set; objects never seen (open set).