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
count. Schema version 2 includes named landmarks. The loader accepts schema 1,
upgrades it in memory with absent landmarks, and rejects unsupported versions.
Saving always writes schema 2.
Raw detections are retained separately from tracked boxes so `analysis.retrack`
can change association, smoothing, and hold parameters without repeating
decoding or inference. Skipped inference is distinguishable from no detections.
Images are streamed and all geometry is retained in memory. Long recordings
may eventually justify an indexed, incremental analysis store.

Optional dense analysis adds a separate path:

```text
VideoAnalysis + RGB decode -> per-face parser -> VideoParsing NPZ
                                                 |
                               ParsedFaceMask <--+
```

`parse_video` uses existing tracked observations and never repeats detection.
`VideoParsing` stores one immutable `uint8` label grid per observed face, crop
bounds, frame and analysis digests, source digest, and parser provenance. NPZ
schema 1 contains compressed arrays and JSON metadata and loads with pickle
disabled. Detection JSON remains schema 2 without dense image arrays. Changing
mask configuration reuses labels. Changing tracked geometry invalidates them.
Cache saves publish atomically and refuse existing files without explicit
overwrite. `start`/`stop` allow partial analysis for experiments, while video
export requires the complete interval. Dense grids remain in memory during
analysis, so long recordings may need a future incremental cache writer.

## Images, coordinates, and composition

Public images are nonempty `H x W x 3` NumPy arrays in RGB order with `uint8`
values. OpenCV's BGR conversion happens at video and detector boundaries.
The origin is the top-left, x increases rightward, and y increases downward.
`Box` uses continuous full-frame pixel edges with half-open intervals
`[x1, x2) x [y1, y2)`. Coordinates can extend outside the image. Integer crops
use floor for the minimum edge and ceil for the maximum, clipped to the frame.

`LandmarkFaceMask` is the default. Its inspectable `FaceMaskGeometry` contains
full-frame center, radii, roll in image coordinates, and a landmark-guided flag.
The subject's eye line estimates roll within 60 degrees. Eye and mouth spacing
constrain size within box-relative limits. The center remains anchored near the
box center so foreshortened eye/mouth spacing cannot discard the forehead.
Missing names, degenerate eyes, inverted nose/mouth ordering, implausible
coordinates, or excessive roll produce an upright box-based fallback. Constants
in this model are geometric heuristics, not learned anatomical measurements.

The core is an asymmetric superellipse with exponent 2.6, a configurable lower
radius extension (10 percent by default), and lower-face width taper up to
8 percent. Pixel-center sampling produces
`float32` alpha in `[0, 1]`. The core stays opaque and compact smoothstep
feathering extends outward in normalized radial distance. Feather width is
relative to shape radius, so it is not a constant pixel-distance border.
Clipping never recenters or rescales geometry. These five landmarks cannot
resolve exact hairlines, ears, or occlusion boundaries.

`MaskSession.build(frame_analysis, width, height)` returns regions keyed by
track ID. A box-relative EMA filters geometry without adding translation or
scale lag to tracked boxes. Its output is combined with the current unfiltered
mask using pointwise maximum, so temporal stabilization cannot reduce current
mask coverage. This can modestly broaden the boundary during changes. State
is private to a render and track. Gaps, absent IDs, dimension changes, fallback
geometry, and box IoU below 0.1 reset history. Held observations freeze the
filter. Stale held geometry and tracking lag can still leave moving faces exposed.

`FaceMask` retains the original axis-aligned ellipse/rectangle implementation
for existing experiments, with its inward linear feather convention. Both
implementations return integer crop bounds and local alpha arrays. Mask
sessions use only cached geometry and never run inference or alter analysis.

`MaskProvider` is the extension contract for masks with independent analysis.
It supplies `create_session()` with a frame-level `build` method and
`validate_analysis(analysis)`. `MaskSession` delegates to providers. Video export
validates a provider before encoding, while parsed frame sessions also check
frame digests and dimensions for direct array use. The supplied RGB arrays must
still correspond to those frames, which cannot be verified from geometry alone.

`ParsedFaceMask` selects facial classes from BiSeNet's 19-class layout. Every
face is parsed separately using a context-expanded square crop, with default
side length 1.8 times the larger box dimension. Replicated padding and
pixel-center sampling preserve crop coordinates at frame edges. Label lookup
uses the un-clipped crop bounds, so clipping does not stretch a face. The
OpenCV CPU adapter selects the named primary `output` head, uses upstream RGB
normalization, and records the model digest. The pinned downloader supplies the
ResNet34 model and upstream license. Neither inference nor rendering downloads.

Components that contain tracked facial points are selected. Without valid
seeds, the component with the greatest overlap with the track box is selected.
This excludes disconnected neighboring faces, but touching components and bad
tracks can remain ambiguous. Enclosed background holes are filled without
filling excluded semantic classes. Accepted core pixels remain opaque.
Outward smoothstep feathering is measured in source pixels and may enter only
predicted background. A box-relative resampling of prior alpha stabilizes each
track. Current coverage is retained, and excluded labels are vetoed last.
Consequently, an old skin prediction cannot paint over current hair labels.

A parse is rejected if its selected area is below 15 percent of the visible
box area, its extent does not reach the expected upper face, chin, and sides,
or a visible eye, nose, or mouth point lacks core coverage. Extent must span
the box's central 30 percent horizontally, reach its upper 35 percent, and
extend through 80 percent of its height. Raw observation landmarks are used
when available because remapped tracked points can lag real image features.
Existing caches lack explicit measurement links, so these are reconstructed
with gated one-to-one IoU association. Tracked points are used without raw data.
These thresholds are heuristic and do not measure silhouette accuracy. The
default `fallback="hold"` retains the last accepted semantic alpha and transports
it using the tracked box's translation and scale. Every carried frame samples
the accepted anchor, avoiding accumulated interpolation blur. Unusable current
labels do not replace that anchor or veto its shape. A fresh accepted parse
restores the usual current-label exclusions. The anchor can become stale as
hair, expression, pose, or occlusion changes.

The hold budget is measured from the last accepted parse, with a default of 15
decoded frames. Unusable parses and missing observations cannot refresh it.
Without an accepted anchor, or beyond the budget, hold mode raises and export
does not publish. It never inserts a geometric mask. Frame gaps, absent IDs,
and anchor/current box IoU below 0.1 clear the available history. Tracks must
remain active, and movement during detection gaps requires motion-aware boxes.
Diagnostics distinguish `held_missing`, `held_empty`, `held_incomplete`, and
`held_recovery` from a fresh `parsed` shape.

Explicit `fallback="geometry"` uses measured boxes when available, favors
coverage, and can include hair or ears. Unusable parses clear its accepted-label
history. Missing observations can reuse the last accepted labels. The
`fallback="error"` policy refuses any unusable parse immediately. An optional
per-track recovery counter requires consecutive usable observations before
accepting a fresh shape after a failure. Hold mode carries the anchor during
recovery. Geometric mode retains geometry. Invalid or missing observations
interrupt the counter, and recovery never extends the hold budget. Default
recovery is one observation. There is no crossfade into a partially opaque face.

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

`YuNetFaceDetector` is the optional neural adapter. It uses the existing
OpenCV DNN CPU backend with a local YuNet ONNX model and returns five points
alongside boxes and model scores. A pinned OpenCV 4.x-compatible model is
available through the explicit `download_yunet_model` helper and CLI command.
Both the weights and the upstream license sidecar are verified by SHA-256.
There is no network access during detector initialization or inference.

Inference inputs are aspect-preserving area resizes bounded by `max_size`.
The adapter uses the actual rounded width and height to map x and y coordinates
back independently, then returns deterministic box-sorted results. It converts
public RGB input to the backend's BGR format and updates input dimensions on
every call, allowing serial reuse across resolutions. Model digest, resize
policy, thresholds, and point layout are recorded in provenance. Instances
contain mutable backend state and should not be shared across concurrent calls.

`Landmarks` stores immutable named `(x, y)` pairs in full-frame coordinates.
YuNet names follow OpenCV's anatomical right/left convention, ordered as right
eye, left eye, nose tip, right mouth corner, left mouth corner. Coordinates are
finite but may extend outside the frame or detector box. Other detectors can
return different explicit layouts. Layout conversion must be an explicit
operation, because different landmark schemes need not describe the same points.

`IoUTracker` gates box-pair IoU, then uses SciPy's global linear assignment to
maximize valid match count and summed IoU. Gating precedes assignment so invalid
edges cannot steal valid matches. New IDs follow sorted detection coordinates.
The current observation has weight `smoothing` in the updated box. Unmatched
tracks retain their last box for at most `max_missing` decoded frames unless
external motion predictions move that box.
Reacquired tracks keep their IDs and expired IDs are never reused in a clip.
`observed=False` distinguishes held boxes from measurements.

Raw detections preserve their measured landmarks. Each tracked observation
maps those points affinely from its measurement box into the smoothed track
box, preserving box-relative geometry. Held tracks keep both box and points
unchanged by default and report their missing-frame age. Optional image-derived
box predictions remap held landmarks without changing observation age. A new
observation without landmarks clears old points. There is no independent
temporal landmark filter. This keeps tracking geometry consistent while
allowing the original measurements to be inspected and retracked from cache.
Effects receive tracked landmarks through `EffectContext.face`. Subtracting
the crop bounds' `(x1, y1)` converts a full-frame point to crop-local pixels.

Track IDs are local associations within a clip, not biometric identities.
There is no recognition embedding or backward interpolation. Larger detection intervals increase the risk of
uncovered movement and late entries. These tradeoffs remain explicit in the
analysis records.

Optional `SparseOpticalFlow` estimates median translation from interior
Shi-Tomasi corners using pyramidal Lucas-Kanade flow in both directions.
Forward/backward disagreement, photometric error, insufficient inliers or
spatial support, excessive displacement, and long observation gaps reject an
estimate. Existing track boxes are predicted before global association and
smoothing. No detections or identities are invented by motion estimation.
The coarse RGB scene-change guard clears active tracks while retaining unused
IDs. It is a heuristic, including possible missed cuts and false alarms.

`track_video` replays existing raw detections while decoding the source for
motion. Settings, per-frame status, cuts, and the parent analysis digest enter
provenance. New tracked geometry invalidates old parsing caches. Box-only
`retrack` strips motion diagnostics and respects recorded cuts. Analysis schema
2 remains unchanged because predictions use existing box/landmark fields and
the additional diagnostics are JSON provenance. No frames enter the cache.

Geometric masks optionally protect protruding feature points using a bounded
convex hull, only when a canonical point would otherwise lack opaque coverage.
The relative-box gate rejects remote outliers and a small source-pixel margin
covers each accepted feature pixel. Low-level analytic shape methods remain
separate from this guard. Semantic fallback uses the same geometric builder.
Accepted semantic masks never apply an unsegmented feature hull over their
class vetoes.

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

Mask configuration can be selected by cached track ID using `PerFaceMask`.
Each configuration object owns a separate session. Complete original frames
are preserved for cache validation. Sessions with `build_selected` can scope
processing after validation, which prevents strict semantic fallback errors
from unrelated faces. Other custom providers calculate complete frames and the
wrapper selects their output regions. Configuration changes do not repeat
inference. Track IDs remain subject to association errors.

Semantic core editing uses signed source-pixel distances, followed by outward
feathering and temporal filtering. The final class veto retains exclusions,
apart from explicitly allowed `boundary_labels`. Negative expansion clips
history to the edited mask's feather support. Quality gates inspect the raw
semantic core before deliberate edits. Geometric fallback shape has its own
configuration and remains an instantaneous estimate.

`anamorph.evaluation` measures masks against reference polygons independently
of model labels. The replayable evaluation example binds annotations to video,
decoded frame, and analysis hashes, and records per-face fallback status. The
reference format and metric limitations are in [mask evaluation](mask_evaluation.md).

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
- [OpenCV YuNet API and point layout](https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html)
- [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [OpenCV Zoo YuNet model and license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [BiSeNet parser, preprocessing, and model releases](https://github.com/yakhyo/face-parsing)
- [CelebAMask-HQ labels and dataset agreement](https://github.com/switchablenorms/CelebAMask-HQ)
- [OpenCV headless packaging and bundled cascades](https://pypi.org/project/opencv-python-headless/)
- [Video writer contracts and codec limitations](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
- [SciPy linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
- [uv build backend](https://docs.astral.sh/uv/configuration/build-backend/)
