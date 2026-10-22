# Anamorph

An experimental Python library for facial anonymization and artistic video
transformations. The first milestone provides a working CPU pipeline with
reusable face analysis, tracking, configurable masks, and spatial and causal
temporal effects. There is no GUI.

This is research software. Face detection can miss faces and transformations
do not provide a measured anonymity guarantee. Inspect outputs before relying
on them for privacy. Audio is **not anonymized or copied**.

## Install

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this repository's checkout:

```bash
uv sync --locked
```

This installs the library in editable mode and development tools in `.venv`.
Use `uv sync --locked --no-dev` for runtime dependencies only. For an existing
environment, `pip install .` also works. Anamorph is not yet published to PyPI.

NumPy, SciPy, and `opencv-python-headless` are the runtime dependencies. OpenCV
provides codecs and a bundled frontal Haar cascade, so the default detector
requires no separate model download. Avoid installing another OpenCV wheel
alongside the headless package because they share the `cv2` namespace.

## Python API

```python
from anamorph import GaussianBlur, LandmarkFaceMask, process_video

report = process_video(
    "local_inputs/clip.mp4",
    "outputs/blurred.mp4",
    effect=GaussianBlur(sigma=16),
    mask=LandmarkFaceMask(feather=0.08),
)
print(report.frame_count, report.output)
```

All output is **video-only, constant frame rate**. Audio, subtitles, original
timestamps, and container metadata are not preserved. The API emits a warning
at export to make this behavior explicit. Use constant-frame-rate inputs for
timing fidelity. Existing destinations require `overwrite=True`. Input and
output must be different files.

### Analyze once, render many times

```python
from pathlib import Path
from anamorph import (
    Compose,
    GaussianBlur,
    IoUTracker,
    Pixelate,
    TemporalBlend,
    VideoAnalysis,
    analyze_video,
    render_video,
)

Path("outputs").mkdir(exist_ok=True)
analysis = analyze_video(
    "local_inputs/clip.mp4",
    tracker=IoUTracker(max_missing=5, smoothing=0.7),
    detection_interval=1,
)
analysis.save("outputs/analysis.json")
analysis = VideoAnalysis.load("outputs/analysis.json")

for name, effect in {
    "blur": GaussianBlur(sigma=16),
    "pixels": Pixelate(cells=6),
    "ghost": Compose(Pixelate(cells=6), TemporalBlend(current_weight=0.35)),
}.items():
    render_video(
        "local_inputs/clip.mp4",
        f"outputs/{name}.mp4",
        analysis=analysis,
        effect=effect,
    )
```

Analysis stores geometry and provenance in versioned JSON, with no image data.
A SHA-256 digest binds it to the exact source bytes. Rendering performs no
face inference. It decodes frames, creates masked face crops, applies effects,
and composites them into the frame. Decoded frames are streamed, while analysis
records occupy memory proportional to frame count and face count.

Raw detections are retained separately from smoothed track boxes. To explore
tracking parameters without decoding or inference, use
`analysis.retrack(tracker=IoUTracker(smoothing=1, max_missing=8))`.
`FrameAnalysis.detections` is `None` on skipped inference frames and an empty
tuple when inference ran without finding faces.

`GaussianBlur.sigma` is measured in crop pixels. `Pixelate.cells` sets the
maximum number of cells along each crop axis. `Compose(a, b)` applies `a`, then
`b`, and composites once. `TemporalBlend` retains per-track history for the
current render and demonstrates a causal temporal effect.

### Neural detection and landmarks

`YuNetFaceDetector` uses OpenCV's CPU neural detector and returns five named
points: `right_eye`, `left_eye`, `nose_tip`, `right_mouth_corner`, and
`left_mouth_corner`. Left and right refer to the subject's anatomical sides.
No additional Python dependency is required. Download the pinned OpenCV 4.x
model and its MIT license explicitly:

```bash
uv run anamorph download-yunet local_inputs/models/yunet/face_detection_yunet_2023mar.onnx
```

```python
from anamorph import YuNetFaceDetector, analyze_video, download_yunet_model

model = download_yunet_model("local_inputs/models/yunet/face_detection_yunet_2023mar.onnx")
detector = YuNetFaceDetector(model, score_threshold=0.9, max_size=640)
analysis = analyze_video("local_inputs/clip.mp4", detector=detector)

for frame in analysis.frames:
    for face in frame.faces:
        if face.landmarks is not None:
            print(frame.index, face.track_id, face.landmarks["nose_tip"])
```

The download helper verifies the model and license against pinned byte counts
and SHA-256 hashes. Matching files are reused offline and different existing
files are preserved. Detector construction and inference never download files.
The default Haar detector remains available without a neural model.

YuNet resizes the longest input dimension to at most `max_size`, preserves
aspect ratio, and never upscales. Set `max_size=None` for native resolution.
All returned boxes and landmarks use original-frame pixel coordinates,
including separate x/y corrections for rounded resize dimensions. Scores are
model outputs and are not calibrated probabilities. Thresholds, resizing,
landmark layout, and the model digest are recorded in analysis provenance.
Lowering `score_threshold` can improve recall while admitting more false
positives. Native-resolution inference can help with small faces and costs
more computation. Choose settings against representative annotated footage.
For example, `YuNetFaceDetector(model, score_threshold=0.7, max_size=None)`
provides a configuration to explore for higher recall on small faces. In the
CLI the corresponding options are `--score-threshold 0.7 --max-size native`.

Raw `Detection.landmarks` contain measured points. `Face.landmarks` map those
points into the track's smoothed box while preserving their box-relative
positions. Missing detections hold both box and points, with `observed=False`.
Without optional motion, these held points are stale estimates. Landmark shape is not independently
smoothed. Effects can access `context.face.landmarks` and subtract the crop
origin in `context.crop_bounds` to obtain crop-local coordinates.

The immutable `Landmarks` representation pairs explicit `names` and `(x, y)`
`points`. Analysis schema 2 stores them for raw detections and tracked faces.
Existing schema-1 files load with absent landmarks and are upgraded in memory.
Saving writes schema 2. The default mask uses these five landmarks when
available, with an upright box-based fallback. Optional semantic parsing adds
dense face boundaries through a separate cache, described below.

For the CLI, select YuNet during analysis and reuse the resulting file normally:

```bash
uv run anamorph analyze local_inputs/clip.mp4 outputs/analysis.json --detector yunet --model local_inputs/models/yunet/face_detection_yunet_2023mar.onnx
uv run python examples/process_video.py local_inputs/clip.mp4 outputs/comparison --model local_inputs/models/yunet/face_detection_yunet_2023mar.onnx
```

### Masks and tracking

`LandmarkFaceMask` is the default for Python rendering and CLI `--shape face`.
It targets the full face, including forehead, cheeks, and chin, with a broad
upper contour and tapered jaw. The eye line guides roll. Landmark spacing
constrains size, while the detector box anchors coverage when looking down or
turning sideways. Missing or implausible points produce an upright fallback.
This is an inspectable geometric estimate. Five points cannot locate the exact
hairline, ears, or silhouette, so some overlap or exposed skin remains possible.

`scale_x` and `scale_y` multiply estimated face dimensions. `offset_y` shifts
along the face's downward axis by a fraction of box height. `feather=0.08`
extends a smooth boundary **outside** the fully opaque core, by a fraction of
the shape radius. Zero gives a hard edge. Increasing feather never weakens
opaque core coverage. Clipping at frame edges preserves the original geometry.
`chin_extension=0.10` extends the lower radius toward the jaw without lifting
the top boundary into the hair. Defaults use `scale_x=1.12` and `scale_y=1.10`.

Each renderer owns a `MaskSession`. It filters box-relative center, size, and
roll per track with `stabilization=0.65` as the current observation weight.
Position and scale follow the tracked box without extra filtering lag. The
maximum of current and filtered masks retains current-frame coverage. Set
`stabilization=1` to disable filtering. Absent tracks, frame gaps, resolution
changes, invalid landmarks, and large box jumps reset history. Held detections
freeze the filter and retain stale geometry, so moving faces can escape them.

For array experiments, `mask.geometry(box, landmarks)` exposes the estimated
center, radii, roll, and fallback flag. `mask.build(box, width, height,
landmarks=points)` builds an independent mask. `mask.alpha_at(geometry, x, y)`
samples continuous full-frame coordinates for numerical inspection.
`MaskSession(mask).build(
frame_analysis, width, height)` returns track-ID keyed `MaskRegion` objects
with full-frame bounds and local `float32` alpha arrays. Sessions require
increasing frame indices. No decoding or inference is needed for these calls.

The original `FaceMask` remains available with unchanged `ellipse` and
`rectangle` behavior, including its **inward** feather convention and box-axis
offset. Explicit `FaceMask(...)` experiments retain their previous output.
Omitting `mask` now selects `LandmarkFaceMask()`. Old analysis caches work via
the upright fallback.

Inspect boundaries on your footage before tuning effects:

```bash
uv run python examples/inspect_masks.py local_inputs/clip.mp4 outputs/analysis.json outputs/mask_review --start 0 --frames 200
```

This reuses analysis and produces a verified comparison video, still images,
and a JSON settings report. Cyan contours mark the opaque core and white
contours mark feather support. These translucent inspection previews reveal
the original faces and **are not anonymized**. The report's tracked-landmark
coverage of visible points is a diagnostic, not an independent face-area or
privacy measurement. Offscreen landmarks are counted separately.

For independent point annotations, `examples/evaluate_mask_coverage.py` compares
legacy and full-face masks from cached analysis. Provide a JSONL file with one
object per frame containing `frame_index` and `landmarks` as an `N x 2` list of
full-frame coordinates. Use a single annotated subject per clip. The evaluation
samples the containing pixel, measures the union of masks, and reports opaque
point coverage and frames with exposed points. Point annotations do not define
the entire face silhouette or establish performance on unseen footage.

```bash
uv run python examples/evaluate_mask_coverage.py outputs/analysis.json local_inputs/landmarks.jsonl outputs/mask_coverage.json
```

`IoUTracker` uses global one-to-one box association and optional exponential
box smoothing. IDs persist through brief missed detections using static held
boxes by default. `max_missing` counts decoded frames, including skipped inference frames.
It must be at least `detection_interval - 1`. A frame record exposes `observed`
and `missing_frames` for each face. IDs restart for each video.

### Motion through short detection gaps

`SparseOpticalFlow` optionally predicts existing boxes from face-interior image
features before association. It checks forward/backward tracking agreement,
photometric error, inlier fraction, spatial support, and maximum displacement.
Only translation is estimated. Predictions do not count as detections or reset
the missing age. They are bounded to three consecutive missing frames by default.
Rejected estimates use the normal tracker hold/expiry policy.

```python
from anamorph import SparseOpticalFlow, analyze_video, track_video

analysis = analyze_video(
    "local_inputs/clip.mp4",
    detector=detector,
    motion=SparseOpticalFlow(),
)
# Alternatively, reuse raw detections from an existing VideoAnalysis.
refined = track_video(
    "local_inputs/clip.mp4",
    analysis=analysis,
    motion=SparseOpticalFlow(),
)
```

`track_video` decodes images but performs no detection inference. The new
analysis has different tracked geometry, so parse it again before using semantic
masks. Mask and effect changes can then reuse that new parsing normally.
`analysis.retrack(...)` remains the faster box-only option and drops motion
diagnostics because it does not replay images. Detected scene-cut boundaries
are retained. A coarse RGB-change heuristic clears tracks on detected cuts,
preserving unused IDs, and forces detection when analyzing fresh footage.
During cached replay, it uses only available cached observations. The heuristic
can miss cuts or mistake lighting changes for cuts.

Motion settings, per-frame acceptance/rejection reasons, and cut indices are
saved in analysis provenance. A successful motion estimate is not an identity
or occlusion guarantee. Optical flow cannot discover a face that was never
detected, reliably follow complete occlusion, or solve crossings.

```bash
uv run python examples/compare_tracking.py local_inputs/clip.mp4 outputs/analysis.json outputs/tracking_review
```

The example writes static/flow analyses and a verified mask comparison preview.
Use `--detection-interval 3` for an explicit experiment skipping cached observations.
With normalized box annotations, `--annotations local_inputs/boxes.jsonl` reports
one-to-one box matching, adjacent ID switches, and consecutive miss runs.
Unannotated detection counts are diagnostics. Preview faces remain visible.

### Semantic face boundaries

`BiSeNetFaceParser` adds pixel-level facial labels using the existing OpenCV CPU
runtime. `ParsedFaceMask` includes skin, eyebrows, eyes, glasses, nose, mouth,
and lips. It excludes predicted hair, ears, earrings, neck, necklace, clothing,
and hats by default. It is optional because it requires external weights and
adds inference per observed face. Geometric masks remain the default without it.

Download the pinned ResNet34 model (about 94 MB) and its upstream license:

```bash
uv run anamorph download-face-parser local_inputs/models/face_parsing/resnet34.onnx
```

Reuse detection analysis, parse once, then reuse the labels for mask experiments:

```python
from anamorph import (
    BiSeNetFaceParser,
    GaussianBlur,
    ParsedFaceMask,
    VideoAnalysis,
    VideoParsing,
    parse_video,
    render_video,
)

analysis = VideoAnalysis.load("outputs/analysis.json")
parsing = parse_video(
    "local_inputs/clip.mp4",
    analysis=analysis,
    parser=BiSeNetFaceParser("local_inputs/models/face_parsing/resnet34.onnx"),
)
parsing.save("outputs/parsing.npz")
parsing = VideoParsing.load("outputs/parsing.npz")

render_video(
    "local_inputs/clip.mp4",
    "outputs/parsed_blur.mp4",
    analysis=analysis,
    effect=GaussianBlur(16),
    mask=ParsedFaceMask(parsing, feather=2, fallback="geometry"),
)
```

`crop_scale=1.8` in `parse_video` expands each tracked box to a square with
context. Crops use bilinear sampling and replicated borders. The parser uses
512 x 512 RGB inputs, upstream channel normalization, and the primary 19-class
output. It never downloads weights during construction or inference.

The NPZ cache stores immutable label grids and JSON provenance, without pickle
or RGB images. It is bound to the source bytes and the exact detection/track
analysis. Retracking requires parsing again because it changes crop geometry.
Changing `ParsedFaceMask.included_labels`, boundary edits, feathering, or
stabilization reuses the same parsing. Extend the default `included_labels`
tuple to add hair or ears.
Full-face masks require all default facial classes to remain included. For
other semantic selections, `FaceParsing.labels_at(x, y)` exposes full-frame labels.
`save` refuses existing paths unless `overwrite=True`.

Each face has a separate temporal session. Connected components are selected
using that face's landmarks, or box overlap when landmarks are absent, to avoid
including a disconnected neighbor. Enclosed background holes are filled while
explicitly excluded classes remain excluded by default. Accepted facial pixels
are fully opaque before deliberate boundary edits. `feather` is measured in
**full-frame pixels** and extends into predicted background by default.
Class exclusions are applied after temporal filtering, so
skin/hair edges may remain hard. Touching faces and incorrect predictions can
still defeat component separation and boundary accuracy.

Tiny regions, limited face extent, and uncovered visible feature points fail
a coverage check. When raw detections are available, checks use their measured
points instead of points remapped into smoothed boxes. Matching reconstructs
measurement links using gated one-to-one box association. Without raw detections,
tracked points remain estimates. By default, failed checks use a geometric
fallback anchored to the measured box when available, which can include hair and
ears. Set `fallback="error"` to refuse such frames. These checks are heuristics,
not a complete silhouette assessment. `mask.create_session().diagnostics`
reports `parsed`, `held`, `fallback_empty`, `fallback_incomplete`, or
`fallback_missing` after `build(frame_analysis, width, height)`. Held tracks
reuse their previous accepted labels and remain stale without motion prediction.

`expansion` grows or shrinks the opaque core by a signed distance in source
pixels. Positive values grow into background while retaining predicted hair,
ear, and neck exclusions. Negative values deliberately remove facial coverage,
then apply the outward feather. Temporal history cannot undo deliberate shrinkage.
Edits stop at the parsing crop boundary. Coverage checks run before these edits.
The CLI exposes this control as `--expansion` with `--shape parsed`.

For a known boundary error, `boundary_labels=("background", "neck")` permits
growth and feathering a limited distance into predicted neck, without including
the entire neck class. The default is `("background",)`. Adding `"hair"` or
ears explicitly allows coverage on those classes near the selected contour.
This trades semantic exclusion for coverage and needs visual review. It does
not correct the underlying model. Disconnected facial components remain excluded.

Use `PerFaceMask` to give selected cached track IDs different configurations:

```python
from dataclasses import replace
from anamorph import LandmarkFaceMask, ParsedFaceMask, PerFaceMask

base = ParsedFaceMask(parsing, expansion=1, feather=2)
mask = PerFaceMask(
    default=base,
    overrides={
        7: replace(base, expansion=3, feather=4),
        12: replace(base, fallback_mask=LandmarkFaceMask(scale_y=1.3)),
    },
)
# Pass mask to render_video, or inspect mask.create_session().build(...) first.
```

Choose IDs from your analysis. They are tracker IDs within that clip and can
switch during crossings. Overrides are immutable and sessions isolate temporal
state per configuration and track. Semantic providers validate the original
frame before processing the selected IDs, so strict fallback errors apply only
to faces using that configuration. Custom providers can optionally implement
`build_selected(frame, width, height, track_ids)` on their sessions. Otherwise
they receive and calculate the complete frame. No inference repeats when tuning
mask settings. `fallback_mask` controls the instantaneous geometric shape,
including scale, position, chin extension, and feathering. Its `stabilization`
field is unused because fallback geometry is built independently per frame.

Equivalent CLI commands and a comparison overlay:

```bash
uv run anamorph parse local_inputs/clip.mp4 outputs/analysis.json outputs/parsing.npz --model local_inputs/models/face_parsing/resnet34.onnx
uv run anamorph render local_inputs/clip.mp4 outputs/analysis.json outputs/parsed_blur.mp4 --shape parsed --parsing outputs/parsing.npz
uv run python examples/inspect_masks.py local_inputs/clip.mp4 outputs/analysis.json outputs/semantic_review --parsing outputs/parsing.npz
```

The overlay compares geometric and semantic masks. Orange means a geometric
fallback. Its report counts fallbacks and checks for nonzero alpha on excluded
labels in accepted parses. That checks an implementation invariant, not model
accuracy. Preview faces remain visible. `parse` also accepts `--start` and
`--stop` for a half-open frame interval. Partial caches work for inspection,
while full video rendering requires a complete cache.

The model comes from [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing),
whose code has an MIT license. Its weights were trained on
[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), whose published
terms restrict the dataset and derived data to non-commercial research.
External assets are separate from Anamorph's MIT code license. The downloader
retains the upstream license beside the model and verifies both assets by hash.

### CLI and examples

The CLI is a lightweight wrapper over the same Python functions:

```bash
mkdir -p outputs
uv run anamorph analyze local_inputs/clip.mp4 outputs/analysis.json
uv run anamorph render local_inputs/clip.mp4 outputs/analysis.json outputs/pixels.mp4 --effect pixelate --cells 6
uv run python examples/process_video.py local_inputs/clip.mp4 outputs/comparison
```

`python -m anamorph` provides the same commands. Use `--help` for mask, codec,
and tracking options. `--codec FFV1` with an `.avi` output is useful for lossless
experiments when supported. The default `mp4v` codec is intended for `.mp4`.
YAML configuration is deferred until experiments demonstrate a need for it.

For direct array processing, use `GaussianBlur(...).apply(rgb)`, or create a
`FrameRenderer(effect, mask)` and call `render(rgb, frame_analysis,
timestamp=seconds)`. A renderer owns one sequential session and requires
strictly increasing frame indices. Arrays use `uint8` RGB throughout the public
API. See [the architecture and extension guide](docs/architecture.md) and
[a custom spatial effect](examples/custom_effect.py).

## Development and validation

```bash
uv sync --locked
uv run pytest --cov=anamorph --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests cover numerical behavior, coordinate clipping, mask boundaries, tracking
and missing detections, temporal state isolation, cache validation, and real
encoding/decoding through the pipeline. Controlled detections make the video
tests deterministic. Lossless output is compared against direct array rendering
pixel for pixel. MP4 export and the bundled detector are exercised separately.
The FFV1 tests skip explicitly if that encoder is unavailable.

YuNet unit tests cover original-coordinate mapping, RGB/BGR conversion,
landmark naming, smoothed and held tracks, cache migration, and verified model
downloads using controlled inputs. Run the optional real-model integration
test after downloading the weights:

```bash
uv run pytest tests --yunet-model=local_inputs/models/yunet/face_detection_yunet_2023mar.onnx
```

Add `--face-parser-model=local_inputs/models/face_parsing/resnet34.onnx` to run
the optional real BiSeNet test as well. Without model options, those integration
tests skip. Test execution never downloads models or footage. Semantic tests
cover class exclusions, multi-face component selection, temporal vetoes,
fallbacks, crop coordinates, source/analysis binding, NPZ round trips, and exact
lossless rendering without repeated inference.

No natural video footage or external neural model is bundled. Tests do not
measure real-world detector recall, tracking accuracy, or re-identification
resistance. Those require representative, appropriately licensed footage and
an evaluation protocol.

For silhouette evaluation, [the mask evaluation guide](docs/mask_evaluation.md)
defines local polygon annotations, explicit hair/ear exclusions, uncertain
boundaries, area and boundary measurements, and a temporal flicker proxy.
`examples/evaluate_masks.py` replays cached masks against those references
without inference and writes per-face reports and review images. Keep
annotations separate from model predictions and report their provenance.

## Current scope and next steps

- The bundled detector targets frontal faces. Profiles, occlusion, small faces,
  and scene cuts can cause failures. Held boxes can miss moving faces, box
  smoothing can lag motion, and crossing faces can exchange IDs.
- YuNet exposes five anatomical landmarks. Geometric masks estimate tilted
  full-face coverage. Optional semantic masks exclude predicted hair and ears,
  but small, dim, or occluded faces can require geometric fallbacks. Exact
  anatomical exclusion and independent silhouette accuracy are not guaranteed.
  Optional sparse optical flow handles bounded translation through short gaps.
  It can reject low-texture or occluded faces and cannot recover unseen faces.
- Temporal blending resizes the previous crop into current crop coordinates.
  It can create ghosting and is not a tracking improvement.
- Video export requires even dimensions and an available codec/container pair.
  Corrupt-stream decoding can stop early because OpenCV does not reliably
  distinguish decode failure from end-of-file. Export is decoded again to
  verify frame count, dimensions, and frame rate before publication.
- Processing is CPU-only. Effects operate on RGB crops of the input resolution,
  and composition uses encoded RGB values rather than linear-light color.
- Current priority: semantic mask coverage, boundaries, and stability on broader
  annotated footage, especially small faces and occlusion in crowded scenes.
  New effects are deferred until mask quality is satisfactory.
  Sequence-level optimization and audio/VFR preservation
  need dedicated interfaces beyond the current causal streaming sessions.

Anamorph's code is MIT licensed. Dependency and bundled cascade licenses remain
with their upstream distributions. See [LICENSE](LICENSE).
