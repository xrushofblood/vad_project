# XR Video Anomaly Detection for Industrial Purposes

Master's thesis project (Computer Science, Sapienza University of Rome).
The goal is to monitor an assembly procedure recorded with an XR headset (egocentric video) and to detect anomalies such as wrong or missing steps. Mr Potato Head is used as a proxy for an industrial assembly: 10 pieces, a clear procedure, and fully controllable recordings.

## Pipeline overview

| Phase | Task | Status |
|---|---|---|
| 1. Perception | Recognize the pieces in each frame | Baseline done (`notebooks/phase_1_perception.ipynb`) |
| 2. State tracking | Track which pieces are mounted and in which order | Planned |
| 3. Reasoning | Check the sequence against the assembly rules (`configs/dependency_graph.json`) with an LLM (Bonsai-27B) | Planned |

### Phase 1 — Perception

No model is trained. C-RADIOv4-SO400M is used as a frozen feature extractor: each frame becomes a grid of 16x16 patches, each described by a 1152-dimensional vector. Each class (10 pieces, background and hand) is represented by a prototype, the normalized mean of its annotated patches in a few reference frames. Each patch of a new frame takes the class of the most similar prototype (cosine similarity), with a margin that rejects uncertain pieces to background. In the video pipeline, perception runs only when the scene is still.

The notebook documents every design choice, with observations, references and sensitivity checks (Appendix A). Its last section lists the current limitations and next steps.

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
│   ├── phase_1_perception.ipynb    # Phase 1 pipeline
│   └── _archive/                   # Not tracked
├── outputs/                        # Not tracked: prototypes, similarities, figures, videos
│   ├── phase_1_perception/
│   └── baseline/                   # Figures of the v1-baseline version, for comparisons
├── logs/                           # Not tracked
├── src/
│   └── perception/
│       └── extract_frames.py       # Frame extraction from the videos
├── archive/                        # Not tracked
├── requirements.txt
├── trials_and_errors_log.md        # Experiment diary: what was tried, what failed, what was kept
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
python src/perception/extract_frames.py
```

The script reads every `.mp4` in `data/raw_videos/` and saves **one frame per second** in `data/extracted_frames/`, named `<video>_f<index>.jpg`: the index is the second of the video (e.g. `assembly_1_f0016.jpg` is at 16 s). Videos whose frames already exist are skipped. The prefix before `_f` identifies the recording, which is what the notebook uses to split reference and test frames by sequence.

Reference frames are annotated with [labelme](https://github.com/wkentaro/labelme); the `.json` annotations are saved next to the frames.

## Usage

Open `notebooks/phase_1_perception.ipynb` and run all cells (Restart & Run All). The notebook reads data from `data/` and the model from `models/`, and writes prototypes, test similarities, figures and the annotated video to `outputs/phase_1_perception/`.

## Versions

- `v1-baseline`: first clean, documented version of the Phase 1 pipeline.

## Next steps

- Annotate the 17 test frames, for a quantitative evaluation (per-class IoU).
- Support-set study: which and how many reference frames.
- Margin study: the background margin and a confidence criterion between the first and second class.
- Continuous perception (latency and resolution).