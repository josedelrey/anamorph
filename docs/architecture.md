# Architecture and extension contracts

## Data flow

```text
source video -> RGB decode -> detector -> IoU tracker -> VideoAnalysis JSON
      |                                                    |
      +------> RGB decode -> mask/crop -> effect session <--+
                                 |              |
                                 +-> composite -> encode -> verify -> output
```

Analysis is independent of effects and mask settings. It records one
`FrameAnalysis` per decoded frame, source dimensions and FPS, the source SHA-256,
detector settings and model digest, tracker parameters, inference interval, and
library versions. Rendering validates the digest, metadata, and decoded frame
count. Schema version 1 rejects unsupported versions and malformed core data.
Raw detections are retained separately from tracked boxes so `analysis.retrack`
can change association, smoothing, and hold parameters without repeating
decoding or inference. Skipped inference is distinguishable from no detections.
Images are streamed and all geometry is retained in memory. Long recordings
may eventually justify an indexed, incremental analysis store.

## Images, coordinates, and composition

Public images are nonempty `H x W x 3` NumPy arrays in RGB order with `uint8`
values. OpenCV's BGR conversion happens at video and detector boundaries.
The origin is the top-left, x increases rightward, and y increases downward.
`Box` uses continuous full-frame pixel edges with half-open intervals
`[x1, x2) x [y1, y2)`. Coordinates can extend outside the image. Integer crops
use floor for the minimum edge and ceil for the maximum, clipped to the frame.

`FaceMask` expands and shifts the original box in continuous coordinates, then
samples its shape at pixel centers. Clipping never recenters the shape. It
returns a crop region and a local `float32` alpha array in `[0, 1]`. Feathering
decreases alpha linearly inward from the shape boundary and has compact support.

Each effect receives a private copy of the original RGB crop. Effects never
read another face's transformed pixels. Results are blended in increasing track
ID order. Later IDs cover earlier IDs where their masks are opaque. This is a
deterministic overlap policy and does not infer anatomical depth. Blending uses
encoded RGB values with round-to-nearest conversion to `uint8`. Zero-alpha
pixels are unchanged before encoding. Lossy codecs can subsequently change any
pixel.

## Detection and identity

`FaceDetector` is a structural protocol with `detect(rgb) -> list[Detection]`
and `metadata() -> dict`. Return full-frame boxes, do not mutate images, and
make metadata JSON-serializable. Include model and preprocessing provenance.
A detector is called serially and should be reusable across videos. Perform
internal resizing as needed and map detections back to original coordinates.
The Haar adapter records the cascade digest and returns no confidence score
because its output is not a calibrated probability.

`IoUTracker` gates box-pair IoU, then uses SciPy's global linear assignment to
maximize valid match count and summed IoU. Gating precedes assignment so invalid
edges cannot steal valid matches. New IDs follow sorted detection coordinates.
The current observation has weight `smoothing` in the updated box. Unmatched
tracks retain their last box for at most `max_missing` decoded frames.
Reacquired tracks keep their IDs and expired IDs are never reused in a clip.
`observed=False` distinguishes held boxes from measurements.

Track IDs are local associations within a clip, not biometric identities.
There is no motion prediction, recognition embedding, scene-cut detection, or
backward interpolation. Larger detection intervals increase the risk of
uncovered movement and late entries. These tradeoffs remain explicit in the
analysis records.

## Effects and extension points

`SpatialEffect.apply(rgb) -> rgb` is an independent stateless image operator.
Subclass it for convolution, Fourier/DCT filtering, wavelet thresholding, SVD,
or crop-space geometric remapping. Transform internally to floating-point,
frequency coefficients, or displacement fields as appropriate. Return clipped,
same-shape `uint8` RGB at the public boundary.

`Compose` performs ordered image-operator composition. Intermediate `uint8`
quantization means it does not imply linearity, commutativity, or coefficient
fusion. For fused numerical experiments, implement one operator that keeps
intermediate values in the desired numeric representation.

`Effect.create_session() -> EffectSession` creates fresh temporal state per
render. `EffectSession.apply(rgb, context)` receives frame index, timestamp,
the tracked face, and full-frame integer crop bounds. Its output follows the
same crop contract. `end_frame(active_track_ids)` runs after every frame,
including empty frames, so sessions can release absent track state. Handle
changing crop sizes and skipped indices explicitly. After an exception in a
stateful effect, create a fresh renderer.

`TemporalBlend` demonstrates per-track exponential history in floating point.
It resizes history on crop-size changes, resets on index gaps, and drops absent
tracks at frame end. `Compose` creates independent sessions for its children.
Effects may mutate their private input crop. Every composition stage is
validated for dtype and shape before the next stage runs.

Frequency operators fit the spatial interface when they reconstruct the same
crop. Geometric effects can inverse-map local crop coordinates. Landmark-driven
meshes, effects moving pixels outside mask support, and full-frame physical
simulations will need richer geometry or frame interfaces. Those interfaces
are deferred until a concrete algorithm establishes their requirements.

Bidirectional filtering, eigenface fitting across a clip, and sequence
optimization require future frames or an explicit fit stage. They should
consume reusable analysis in a separate sequence processor or prepare fitted
parameters for a render session. Current sessions support causal algorithms
only. Arbitrary sequence algorithms do not fit a streaming unary operator.

## Backend and reproducibility

OpenCV headless supplies CPU image operations, bundled detection, and common
codecs without a GUI dependency. NumPy supplies inspectable arrays. SciPy
supplies tested global assignment and a foundation for later signal and
spectral experiments. A `src/` package separates imports from the checkout and
`uv.lock` pins runtime and development dependencies.

Analysis and rendering each decode the source. SHA-256 is checked before and
after both stages. Export writes to a temporary file beside the destination,
closes the encoder, decodes the result, and checks count, dimensions, and FPS.
It replaces an explicitly overwritable destination or publishes with an
exclusive hard link. Exceptions remove temporary output and preserve existing
destinations. Parent directories may remain after a failed render. Publication
with `overwrite=False` requires same-directory hard-link support.

Timestamps are `index / fps`. Variable-rate timing, audio, container metadata,
odd dimensions, and codec-independent bitwise reproducibility are outside this
milestone. Odd dimensions are rejected because OpenCV can truncate them.
Decoder failures can resemble EOF, so source corruption is not fully detectable.
Revalidation catches mismatches between analysis, rendering, and output.
`RenderReport` records output path, frame count, face instances, codec, and
dimensions/FPS. Record effect and mask constructors in experiment scripts
alongside analysis JSON for reproducibility.

## Upstream references

- [OpenCV cascade classification](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [OpenCV headless packaging and bundled cascades](https://pypi.org/project/opencv-python-headless/)
- [Video writer contracts and codec limitations](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
- [SciPy linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
- [uv build backend](https://docs.astral.sh/uv/configuration/build-backend/)
