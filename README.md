# XR Video Anomaly Detection for Industrial Purposes

Master's thesis project (Computer Science, Sapienza University of Rome).
The goal is to monitor an assembly procedure recorded with an XR headset (egocentric video) and to detect anomalies such as wrong or missing steps. Mr Potato Head is used as a proxy for an industrial assembly: 10 pieces, a clear procedure, and fully controllable recordings.

## Pipeline overview

| Phase | Task | Status |
|---|---|---|
| 1. Perception | Recognize the pieces in each frame | Done. V2 with a support set of 10 frames (A) and margin 0. Test on `assembly_1`, run once: mIoU over 10 pieces 0.765 (95% interval 0.719-0.796), against 0.779 in validation. A support set grown by condensation (G) was studied and not adopted |
| 2. State tracking | Track the state of each piece (on the table, in hand, mounted) and the order of picks and mounts; its output is what the LLM reads | Next |
| 3. Reasoning | Check picks and mounts against the assembly rules (`configs/dependency_graph.json`) with an LLM (Bonsai-27B): a wrong pick can be flagged before the mistake is made, a wrong mount when it happens | Planned |

### Phase 1 — Perception

No model is trained. C-RADIOv4-SO400M is used as a frozen feature extractor: each frame becomes a grid of 16x16 patches, each described by a 1152-dimensional vector. The classes are the 10 pieces, background and hand. Prototypes come from the annotated patches of a few reference frames (the support set). Each patch of a new frame takes the class of its most similar prototype (cosine similarity). A background margin can then send to background a patch whose best class beats background by less than the margin; it decides only between background and the other classes, never between two pieces. Frames enter the model at their native resolution, 1024x1024.

Two classifiers are compared in the studies. V0 has one prototype per class, the normalized mean of its annotated patches. V2 uses every annotated patch as a prototype (nearest neighbour); the similarity of a class is the highest similarity among its prototypes.

Current state. The pipeline notebook (`phase_1_perception.ipynb`) runs V2 with the support set of 10 frames and margin 0, and reproduces the support-set study on V2 on the 32 validation frames (0.779; 0.818 at the table, 0.777 at the pick, 0.731 at the mounting; 0.836 with the arms merged). It saves the V2 prototypes (`outputs/phase_1_perception/patch_prototypes.npz`, with the class of each prototype and the metadata) for the next phases. In its video pipeline perception runs only when the scene is still, about 460 ms per frame on the laptop GPU (up to about 650 ms in other executions).

Test. `phase_1_test.ipynb` measured the fixed method once on 31 frames of `assembly_1`, chosen by rules from its event log before annotation, with the measurement plan committed before the run. mIoU over 10 pieces 0.765 (95% interval 0.719-0.796, block bootstrap); 0.835 at the table, 0.706 at the pick, 0.782 at the mounting; 0.819 with the arms merged. The validation value of every measure and of every piece lies inside its test interval, so the validation estimate was only slightly optimistic for a new recording of the same procedure. Base, hat, ear and body reach an IoU between 0.86 and 0.91; arm_2 is the weakest piece (0.416). On the whole video, run frame by frame offline (0.54 s per frame), the mounted pieces keep their class and the complete figure is labelled correctly.

What the studies found:

- Support set (`phase_1_support_set_study.ipynb`, V0): candidate frames were added one at a time to the 7 initial reference frames and kept only if they improved validation. Only 5_assembled_f0019 entered (mIoU with the arms merged from 0.667 to 0.709). A frame aimed at a rare appearance brings few patches and barely moves a mean prototype.
- Resolution (`phase_1_resolution_study.ipynb`): the input stays at 1024. At 512 the prototypes of the small pieces rest on one or two patches, even perfect recognition on the 512 grid would not exceed what the method reaches at 1024, and the mIoU drops by about a quarter. Perception would be about four times faster (about 100 ms per frame instead of about 410).
- Multi-prototype (`phase_1_multi_prototype_study.ipynb`): two frames were added by a coverage rule (the loose mouth seen from the side, the body on the base with the empty mouth slot), and V0, V1 (one prototype per class and aspect, mounted or loose) and V2 were compared on the same 10 frames, at margin 0.03. V2 was chosen: mIoU with the arms merged 0.861 against 0.725 for V0 (+0.135, interval [0.093, 0.172]); V1 was worse than V0 (−0.064). V2 removes the mouth predicted on the empty slot. The Addendum runs perception on every frame of `assembly_2` for the first time.
- Margin (`phase_1_margin_study.ipynb`): the margin is chosen on the reference frames alone, by cross-validation leaving out one sequence at a time; the validation frames only check the choice. V0 needs 0.155 (mIoU from 0.725 to 0.824), V2 needs none (0). At these margins V2 is still better than V0, by +0.040 (interval [0.005, 0.069]). No margin recovers the loose pieces that V2 misses in the video.
- Exploratory test (`scratch_support_swap.ipynb`): with the two frames of `1_disassembled_controlled_caos` added to the support set of V2, the loose arm_2 is seen on the table before its pick in 44% of the frames instead of 12%, and the hat in 67% instead of 54%. For the loose pieces, the lever is which frames are in the support set. These frames are validation frames, so the test is optimistic.
- Support set on V2 (`phase_1_support_set_v2_study.ipynb`): a new validation set of 32 frames of `assembly_2` (10 at the table, 11 at the pick, 11 at the mounting) was chosen by rules fixed from the event log. A decremental selection of objects among 13 candidate frames (Hart 1968, Gates 1972, García et al. 2012), judged by cross-validation with the one-standard-error rule (Hastie et al. 2009), halved the prototypes but was worse in use: mIoU over 10 pieces 0.717 against 0.779 for the current support set; a second selection protecting the largest example of each piece gave 0.744. The current support set is kept (0.818 at the table, 0.777 at the pick, 0.731 at the mounting; 0.836 with the arms merged). It recognizes arm_1 mostly through one example seen at working distance. A piece is recognized mostly through the examples of what it is confused with (the hat through the base seen from above, the arms through each other, body and arms through the hands). No support set tells the loose arms apart.
- Fine details (`scratch_fine_details.ipynb`, exploratory): a larger input (1152, 2048) and a second look at each annotated object enlarged three times were compared on the arm and mouth frames of the validation set, with criteria written before running. None passed: the input stays at 1024. A larger input does not separate the arms and at 2048 costs about ten times more. The second look helps where a good example at working distance exists (arm_1 in hand, the mouth against the nose), not for arm_2; adding arm examples only moves the confusion from one arm to the other. A smaller patch needs a retrained model (FlexiViT), and a shifted grid gives no finer detail (FCN). The identity of similar pieces as whole objects, kept over time, is left to Phase 2.
- Test (`phase_1_test.ipynb`): see above. Of the hypotheses written before the run, three hold (arm_2 weakest and confused with arm_1, mouth partly taken for nose, total lower than validation), and two do not: an arm just picked is found but arm_2 is taken for arm_1 (an error of identity, not of detection), and the small pieces of the figure seen from farther away are not lost.
- Working distance (`scratch_working_distance.ipynb`, exploratory): adding to the support set every annotated example at least as large as the typical size of its piece in use (23 examples) improves most pieces (nose, hat, moustache, base; arms merged from 0.836 to 0.855) but swaps the arms again (arm_1 from 0.741 to 0.363), so the mIoU over 10 pieces falls from 0.779 to 0.759. Without the two mouth frames the mIoU falls to 0.754 and the mouth from 0.662 to 0.514, so they are kept. The current support set is kept; the rule is a recommendation for a new assembly.
- New scene and camera (`scratch_iphone_scene.ipynb`, exploratory): clips recorded with a phone on the head, outside the laboratory. V2 with A recognizes most pieces; the new errors come from coloured objects in the background. arm_2 remains the problem: its closed hand seen from above is labelled base, and the side where an arm is mounted does not decide its class.
- Support set by condensation (`phase_1_support_set_hart.ipynb`): starting from A, the condensed nearest neighbour rule (Hart 1968) added the patches of the 10 pieces of the 32 frames of `assembly_2` that the support set labelled wrongly, and the reduction of Gates (1972) removed those not needed: G is A plus 248 patches. Both rules were fixed before any result. On `assembly_1`, G labels arm_2 correctly in 91 patches against 35 where it is loose or in hand (13 of 16 against 0 just picked), loses no event and raises the mIoU over 10 pieces from 0.765 to 0.807 (interval [0.014, 0.077]); arm_1 is unchanged (0.727 against 0.733). The rule kept A, because a loss of at most 0.014 could not be shown for every piece with 31 frames. A blind comparison by eye on a new phone recording kept A as well: G lost no cell and gained two, but a score of 0 or 1 did not separate the arms. Half of the errors of A on `assembly_2` come from 20 of its 39,125 prototypes, on the boundaries between similar pieces; removing some of them (A') moves the errors instead of removing them. A stays the support set; G is saved as a candidate (`outputs/phase_1_support_set_hart/patch_prototypes_G.npz`).

Known limits of V2 with the current support set (pipeline, Section 10, and test): the two arms are told apart only once mounted; loose, just picked or being mounted, arm_2 is labelled arm_1, except for its closed hand (arm_2 IoU 0.380 in validation, 0.416 in test). The mouth is partly taken for nose and covers only about 3.5 patches in use. The pick is the weakest moment. The hat is taken for body when held, and appears in the holes of the base when the base is alone. Pieces under the hands disappear, and labels flicker from one frame to the next, so a state must be decided over several frames. The errors of A are concentrated on few prototypes that lie on the boundaries between similar pieces (support set by condensation, Section 8). Objects never seen in the recordings are taken for pieces, since there is no "unknown" answer. The empty slots of mouth, nose and moustache stay body: the earlier hypothesis that the empty slots of nose and moustache are taken for mouth is not supported on the test video, where the mouth shrinks only because the moustache covers it. Mounted pieces, the complete figure and the hands are recognized.

Indications for a new assembly, from the studies and still to be verified on new data. Each piece needs at least one example of every appearance it takes in the assembly, loose and mounted, and a new example only when it looks clearly different (coverage rule; multi-prototype study, Section 1). Each piece needs at least one example at the scale of use, seen at working distance (support-set study on V2, Observation 8.5; working-distance exploration). Examples of look-alike pieces should be added together, for instance by condensation from a recording of the real procedure, which corrects both pieces at once; examples of one piece alone move the confusion to its look-alike (support set by condensation, Section 7). A piece is recognized by being separated from what it is confused with, so the support set needs balanced examples of the piece and of its look-alikes and neighbours (hypothesis; support-set study on V2, Observation 10.3). The hands must be in the support set as they appear in use (hypothesis, same study). Reference frames come from recordings never used for evaluation. An automatic selection judged on candidates unlike the use can remove the examples that matter (support-set study on V2, Section 11). All background prototypes come from the laboratory; on a phone in a new scene, coloured objects in the background are the new source of errors (iPhone exploration). A rule of decision must be designed for the precision of the data it uses: with about 30 frames a margin smaller than the natural variation of a piece cannot be shown, and a score of 0 or 1 by eye saturates (support set by condensation, Section 10).

Evaluation protocol. Frames are split by recording: reference frames come from sequences never used for evaluation. Validation is 32 annotated frames of `assembly_2`, chosen by rules from the event log (support-set study on V2, Section 3); `1_disassembled_controlled_caos` is now a reference sequence. After the support-set study on V2 the validation frames are a development set: they were used after its decision, so their values are optimistic. Test (`assembly_1`) was held out until the end of Phase 1 and measured once in `phase_1_test.ipynb` (event log `configs/assembly_1_events.json`, 31 annotated frames); before that it had been inspected only once, in the qualitative evaluation of `v1-baseline`. After the test, `assembly_1` became development material for Phase 2, like `assembly_2`. Phase 2 will be tested on new recordings with assembly errors, made with a phone mounted to simulate the headset, in a scene outside the laboratory. The phone recording of a complete assembly used for the blind comparison of the support sets (`decision.MOV`) is development material.
The v1-baseline file outputs/baseline/test_similarities_v1-baseline.npz also contains the test frames from that inspection; the support-set study loads only its validation entries.

## Repository structure

```
vad-project/
├── configs/
│   ├── dependency_graph.json       # Assembly rules for the reasoning phase
│   ├── assembly_2_events.json      # Event log of assembly_2 (touch, pick, insert, mount, hands off), annotated on the raw video
│   └── assembly_1_events.json      # Event log of assembly_1 (test), same fields; first_seen when a piece is picked outside the frame
├── data/                           # Not tracked, except the labelme annotations (.json) in extracted_frames/
│   ├── raw_videos/                 # Videos recorded with the headset; phone recordings in iphone_scene/
│   └── extracted_frames/           # Extracted frames (.jpg) and labelme annotations (.json)
│   └── annotation_versions/
│       └── margin_study/           # Annotations as used by the margin study (caos_f0211), for the regression check
├── models/                         # Not tracked
│   ├── c_radio_weights/            # C-RADIOv4-SO400M (weights and model code, Hugging Face)
│   └── bonsai_27b_1bit/            # Bonsai-27B, 1-bit quantized (reasoning phase)
├── notebooks/
│   ├── phase_1/                                  # Phase 1, closed
│   │   ├── phase_1_perception.ipynb              # Phase 1 pipeline: V2, support set of 10 frames, margin 0
│   │   ├── phase_1_support_set_study.ipynb       # Which and how many reference frames (V0)
│   │   ├── phase_1_resolution_study.ipynb        # Input resolution: 1024 vs 512
│   │   ├── phase_1_multi_prototype_study.ipynb   # V0, V1, V2 on a support set of 10 frames
│   │   ├── phase_1_margin_study.ipynb            # Background margin for V0 and V2
│   │   ├── phase_1_support_set_v2_study.ipynb    # Support set of V2: new validation set, selection of objects, decision
│   │   ├── phase_1_test.ipynb                    # Test on assembly_1, run once; video of assembly_1; conclusions of Phase 1
│   │   ├── phase_1_support_set_hart.ipynb        # Support set by condensation (Hart, Gates); decision on assembly_1; blind comparison on the phone
│   │   ├── scratch_support_swap.ipynb            # Exploratory: two more frames in the support set of V2
│   │   ├── scratch_fine_details.ipynb            # Exploratory: larger input and second look on the arms and the mouth
│   │   ├── scratch_working_distance.ipynb        # Exploratory: examples at working distance; the two mouth frames
│   │   └── scratch_iphone_scene.ipynb            # Exploratory: new scene and camera (phone)
│   └── _archive/                                 # Not tracked
├── outputs/                                  # Not tracked: prototypes, similarities, figures, videos
│   ├── phase_1_perception/
│   ├── phase_1_support_set_study/
│   ├── phase_1_resolution_study/
│   ├── phase_1_multi_prototype_study/
│   ├── phase_1_margin_study/                 # decision.json, per-frame gaps and labels of assembly_2, video
│   ├── phase_1_support_set_v2_study/         # Video and labels of assembly_2, current and protected support sets
│   ├── phase_1_test/                         # Test predictions, overlays, videos and continuous labels of assembly_1
│   ├── phase_1_support_set_hart/             # Support sets H and G, decision, patch counts by piece and state, video and labels of assembly_1, blind comparison on the phone
│   ├── scratch/                              # Exploratory runs (continuous perception, support swap)
│   └── baseline/                             # v1-baseline prototypes and similarities, for regression checks
├── src/
│   ├── perception/                           # Shared code, imported by every Phase 1 notebook
│   │   ├── __init__.py
│   │   ├── classes.py                        # Class list, indices and colors
│   │   ├── features.py                       # C-RADIOv4 feature extraction (optional input resolution)
│   │   ├── labels.py                         # labelme polygons -> patch labels
│   │   ├── prototypes.py                     # Prototypes (mean, per aspect, per patch), background margin, CPU and GPU similarities, saving of the V2 prototypes similarities
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

V2 runs on the GPU (`class_similarities_gpu`), in about 80 ms per frame with the current support set (up to about 120 ms in some executions), but only when the GPU memory is free: with other notebooks or kernels open, Windows moves data to system memory and V2 slows down to several seconds per frame.

### Models

- **C-RADIOv4-SO400M**: [nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M), commit c0457f5, downloaded into `models/c_radio_weights/`. The model is loaded with `trust_remote_code=True`: the model code comes with the weights.
- **Bonsai-27B (1-bit, GGUF)**: [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf), downloaded into `models/bonsai_27b_1bit/`.

### Data

The Mr Potato Head dataset was recorded for this project with an XR headset (1024x1024, 30 fps). Videos and frames are not in the repository; the labelme annotations (`.json`) are.

Frames are extracted with:

```bash
python src/utils/extract_frames.py
```

The script reads every `.mp4` in `data/raw_videos/` and saves **one frame per second** in `data/extracted_frames/`, named `<video>_f<index>.jpg`. The script keeps one frame every round(fps) frames, so at 30 fps `<video>_f<index>.jpg` is frame index × 30 of the video, at about second `index` (e.g. `assembly_1_f0016.jpg` is frame 480, at about 16 s). Videos whose frames already exist are skipped. The prefix before `_f` identifies the recording, which is what the notebooks use to split reference, validation and test frames by sequence.

`frame_1.jpeg` is an independent frame taken with an iPhone (wide angle), used only as a qualitative check of generalization to a different camera (pipeline, Section 10.5; multi-prototype study, Section 6.3).

Reference, validation and test frames are annotated with [labelme](https://github.com/wkentaro/labelme); the `.json` annotations are saved next to the frames and tracked by git, so the annotations used by each measure are fixed in the history.

## Usage

Every notebook is run with Restart & Run All, reads data from `data/` and the model from `models/`, and writes to its own folder in `outputs/`. Each notebook finds the project folder by itself (the first folder above it that contains `src/`), so it can be run from any folder inside the project. Only `phase_1_test.ipynb` read `assembly_1` as test frames; later notebooks use it as development material.

- `phase_1_perception.ipynb`: the Phase 1 pipeline with the configuration chosen by the studies (V2, 10 frames, margin 0). Builds and saves the V2 prototypes, evaluates the 32 validation frames with a regression check against the support-set study on V2, and writes similarities, figures and the annotated video of `assembly_2` to `outputs/phase_1_perception/`.
- `phase_1_support_set_study.ipynb`: selection of the reference frames for V0. Section 7 adds diagnostic analyses of the selected support set; they do not change the selection.
- `phase_1_resolution_study.ipynb`: 1024 and 512 on the validation frames (recognition and latency); 1024 is kept.
- `phase_1_multi_prototype_study.ipynb`: V0, V1 and V2 on the support set of 10 frames. The Addendum runs continuous perception on `assembly_2` (exploratory).
- `phase_1_margin_study.ipynb`: chooses the margin of V0 and V2 by cross-validation on the reference frames and checks it on the validation frames. Section 8 runs continuous perception on `assembly_2` (about 20 to 25 minutes with a free GPU) and analyses it with `src/utils/continuous_analysis.py`. The decision is saved in `outputs/phase_1_margin_study/decision.json`.
- `phase_1_support_set_v2_study.ipynb`: builds the new validation set, selects objects among 13 candidate frames by cross-validation, chooses their margin, and applies a decision rule on the validation frames; the current support set is kept. Section 10 runs continuous perception on `assembly_2` (about 25 minutes with a free GPU) and measures the influence of each example on each piece (exploratory).
- `phase_1_test.ipynb`: the test of Phase 1 on `assembly_1`, run once (Section 7); the frames are fixed by rules from the event log and the measurement plan is written before the run. Sections 8 and 9 add the video of `assembly_1` (with motion gating and frame by frame, about 21 minutes) and the conclusions of Phase 1. Section 7 must not be run again: later analyses read `outputs/phase_1_test/predictions.npz`.
- `scratch_support_swap.ipynb`: exploratory test outside any study, V2 with the two frames of `1_disassembled_controlled_caos` added to the support set.
- `scratch_fine_details.ipynb`: exploratory test outside any study, input size and second look on the frames of the arms and of the mouth.
- `scratch_working_distance.ipynb`: exploratory test outside any study, examples at working distance added to the support set, and the support set without the two mouth frames.
- `scratch_iphone_scene.ipynb`: exploratory test outside any study, V2 on phone clips of a new scene (central square of the frame, resized to 1024).
- `phase_1_support_set_hart.ipynb`: Hart and Gates on `assembly_2` (Sections 5-6), decision on `assembly_1` with a rule fixed beforehand (Section 7), descriptions and the video of `assembly_1` with A and G (Section 8, about 25 minutes), blind comparison of A, G and H on a phone recording (Section 9: the video takes about 30 minutes, the assignment of the panels is drawn once and the reveal in Section 9.4 is run only after the scoring sheet is filled), conclusions (Section 10).

`src/utils/continuous_analysis.py` can also be run on its own on saved labels (see its docstring). Its windows and thresholds are provisional.

## Versions

- `v1-baseline`: first clean, documented version of the Phase 1 pipeline.
- `v1.1-shared-module`: shared code moved to `src/perception/`, verified bitwise identical to `v1-baseline`.
- `v1.2-support-set`: support set extended to 8 frames (5_assembled_f0019 added), as selected in the support-set study.
- `v1.3-review`: notebooks reviewed (markdowns rewritten, references verified against the papers), diagnostic analyses added to the support-set study (Section 7). The pipeline outputs are unchanged.
- `v1.4-resolution`: input resolution studied (1024 vs 512), 1024 kept. The feature extractor accepts an optional input resolution; arm merging, bootstrap and label upsampling moved to `src/perception/metrics.py`. Results at 1024 are bitwise unchanged.
- `v1.5-multi-prototype`: support set of 10 frames (coverage rule); V0, V1 and V2 compared, V2 chosen. Per-aspect and per-patch prototypes and GPU similarities added to `src/perception/prototypes.py`. Addendum on continuous perception.
- `v1.6-margin`: background margin studied for V0 and V2; V2 with margin 0 decided. `assign_labels` accepts negative margins (labels unchanged for margins of zero or above, checked). Event log of `assembly_2` and `src/utils/continuous_analysis.py` added. Exploratory test on the support set of V2.
- `v1.7-support-set-v2`: new validation set of 32 frames of `assembly_2`; support set of V2 studied by decremental selection of objects; the current support set of 10 frames is kept. Event log revised (touch and insert added). Annotation of `caos_f0211` completed, with the old one archived; new annotated frame `caos_f0278`.
- `v1.8-fine-details`: fine details explored (larger input, second look on annotated objects); none passed the criteria written before running, the input stays at 1024. No change to `src/perception/`.
- `v1.9-pipeline`: pipeline updated to V2 with the support set of 10 frames and margin 0, on the 32 validation frames, with a regression check against the support-set study on V2. Saving and loading of the V2 prototypes added to `src/perception/prototypes.py` (new functions only). Exploration of working distance.
- `v1.10-test`: test on `assembly_1`, run once (mIoU 0.765, 95% interval 0.719-0.796); event log of `assembly_1`; labelme annotations tracked by git; video of `assembly_1` frame by frame; conclusions of Phase 1.
- `v1.11-support-set-hart`: support set by condensation (Hart, Gates) studied on `assembly_2`, judged on `assembly_1` and blind on a phone recording; A kept, G saved as a candidate. Exploration of a new scene with a phone. No change to `src/perception/`.

## Next steps

Phase 2 is planned from the literature before any code (O-VAD, PREGO, Follow the Rules, VERA, in the project files):

- What Phase 2 must output: the state of each piece (on the table, in hand, mounted) and the order of picks and mounts, in words, in a form the LLM of Phase 3 can check every two or three steps.
- How to follow the pieces over time: from the per-patch labels to objects with one identity each, kept over time (a piece does not change identity, only state). Phase 2 is where the identity of the arms, the hidden mouth and unfamiliar objects can be checked against the history of each piece, and against the procedure itself.
- When to run perception: C-RADIO only when it is needed (for instance when the scene changes), measured rather than assumed.
- How to evaluate it: in windows around the events of the log (Ward et al. 2011).
- A realistic calendar until the graduation in March 2027.
- Then the anomalies to record and the dependency graph the LLM checks against.

Open questions: several nearest neighbours (k > 1) instead of one; objects never seen (open set); background examples for a new scene; grouping patches into objects (e.g. class-agnostic masks); G instead of A if the identity of the arms limits Phase 2.
