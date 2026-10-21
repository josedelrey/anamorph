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
from anamorph import FaceMask, GaussianBlur, process_video

report = process_video(
    "local_inputs/clip.mp4",
    "outputs/blurred.mp4",
    effect=GaussianBlur(sigma=16),
    mask=FaceMask(scale_x=1.35, scale_y=1.6, feather=0.12),
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

### Masks and tracking

`FaceMask` supports `ellipse` and `rectangle`. `scale_x` and `scale_y` multiply
the detector box dimensions. `offset_y` shifts the center by a fraction of the
original height. Positive offsets move downward. `feather` specifies the inward
soft boundary as a fraction of the shape radius. Zero gives a hard edge.
Expanded masks can include forehead, hair, and jaw areas beyond the detector
box. They remain aligned when clipped at image edges.

`IoUTracker` uses global one-to-one box association and optional exponential
box smoothing. IDs persist through brief missed detections using static held
boxes. `max_missing` counts decoded frames, including skipped inference frames.
It must be at least `detection_interval - 1`. A frame record exposes `observed`
and `missing_frames` for each face. IDs restart for each video.

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

No natural video footage or external neural model is bundled. Tests do not
measure real-world detector recall, tracking accuracy, or re-identification
resistance. Those require representative, appropriately licensed footage and
an evaluation protocol.

## Current scope and next steps

- The bundled detector targets frontal faces. Profiles, occlusion, small faces,
  and scene cuts can cause failures. Held boxes can miss moving faces, box
  smoothing can lag motion, and crossing faces can exchange IDs.
- Masks are based on axis-aligned boxes. Landmarks, segmentation, optical flow,
  and motion compensation are not implemented.
- Temporal blending resizes the previous crop into current crop coordinates.
  It can create ghosting and is not a tracking improvement.
- Video export requires even dimensions and an available codec/container pair.
  Corrupt-stream decoding can stop early because OpenCV does not reliably
  distinguish decode failure from end-of-file. Export is decoded again to
  verify frame count, dimensions, and frame rate before publication.
- Processing is CPU-only. Effects operate on RGB crops of the input resolution,
  and composition uses encoded RGB values rather than linear-light color.
- Promising next work: a landmark-capable neural detector adapter, a small
  annotated evaluation set, motion-aware tracking, and spectral or low-rank
  spatial operators. Sequence-level optimization and audio/VFR preservation
  need dedicated interfaces beyond the current causal streaming sessions.

Anamorph's code is MIT licensed. Dependency and bundled cascade licenses remain
with their upstream distributions. See [LICENSE](LICENSE).
