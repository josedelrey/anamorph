# Evaluating masks against reference outlines

Use local reference outlines to evaluate coverage before selecting effects.
Keep dataset images and their derived annotations under ignored `local_inputs/`.
Keep reports and review images under ignored `outputs/`. Input dataset terms
still apply. Anamorph's license does not license the footage.

```bash
uv run python examples/evaluate_masks.py local_inputs/clip.mp4 outputs/analysis.json local_inputs/references.json outputs/mask_review --parsing outputs/parsing.npz
```

Omit `--parsing` to evaluate the default geometric mask. Supply multiple
non-overlapping partial parsing caches to evaluate selected intervals without
inferring the entire video. The example warms temporal sessions from each
cache's start. Missing tracks produce zero coverage, rather than disappearing
from the report. Reuse the same annotations and caches for parameter comparisons.

Use `--config local_inputs/mask_settings.json` to compare explicit configurations:

```json
{
  "default": {"kind": "parsed", "expansion": 1, "feather": 2},
  "overrides": {
    "7": {"kind": "parsed", "expansion": 3, "feather": 4},
    "12": {"kind": "face", "scale_y": 1.3}
  }
}
```

Each override is a complete configuration with constructor defaults for omitted
parameters. It does not inherit omitted parameters from `default`. `kind` is
`parsed` or `face`. Parsed settings accept a nested `fallback_mask` object with
`LandmarkFaceMask` shape parameters, and `boundary_labels` for explicit boundary
tolerance. The JSON adapter is local to this example. The primary library API
is `PerFaceMask`, `ParsedFaceMask`, and `LandmarkFaceMask`. Reports record the
configuration and implementation digests. They retain per-track diagnostics.

Reference JSON has the following format. Replace the example digests with
actual SHA-256 hashes. Coordinates describe full-frame continuous pixel edges.
Polygons must be simple, with at least three vertices, and may be concave.
The polygon list represents a union, which supports separated visible regions.

```json
{
  "schema_version": 1,
  "source_sha256": "hash of the original video bytes",
  "analysis_sha256": "hash of the analysis JSON bytes",
  "width": 640,
  "height": 480,
  "annotation_provenance": "Who traced these outlines, how, and with what review",
  "frames": [{
    "index": 10,
    "rgb_sha256": "hash of decoded uint8 RGB frame bytes in row-major order",
    "faces": [{
      "reference_id": "person_a",
      "track_id": 0,
      "tags": ["closeup", "frontal"],
      "face": [[[100, 80], [200, 80], [210, 200], [110, 210]]],
      "excluded": {
        "hair": [[[100, 30], [200, 30], [200, 75], [100, 75]]],
        "ears": [[[85, 110], [95, 110], [95, 150], [85, 150]]]
      },
      "ignore": [],
      "uncertainty_pixels": 2
    }]
  }]
}
```

Trace visible facial tissue including eyebrows, eyes, glasses, lips, and chin,
excluding hair, ears, neck, and occluders. Do not infer hidden anatomy. Use
`ignore` polygons for ambiguous regions. `uncertainty_pixels` ignores an area
around the traced facial boundary in area measurements. Explicit excluded
regions cannot overlap evaluable facial tissue. For reproducible per-person
evaluation, assign the corresponding cached `track_id`, or `null` for a missed
face. Keep `reference_id` stable across frames even if the tracker changes IDs.
The analysis digest prevents silently reusing assignments after retracking.

Trace images without mask overlays, then review the reference overlay and
correct obvious mistakes before comparing predictions. Record whether the
annotator is a human or an AI agent, annotation uncertainty, dataset selection,
and the extent of independent review. AI-traced development annotations are
not independent human ground truth. Do not claim generalization from footage
used to tune masks. A readiness decision needs held-out, independently reviewed
examples of the conditions the library is expected to handle.

## Measurements

`anamorph.evaluation` exposes `polygon_mask`, `evaluate_mask`, and
`temporal_error_change` for direct array-based experiments. No inference runs.

- **Missed opaque face fraction** counts evaluable facial pixels with alpha
  below 0.999. **Mean face alpha deficit** measures the average of `1 - alpha`.
- **Opaque IoU** compares the opaque mask with the reference silhouette.
  **Extra alpha pixels** sums alpha outside the reference, ignoring uncertainty.
- **Exclusion coverage** reports mean alpha and opaque coverage separately for
  each manually annotated hair/ear region. Predicted model labels are not used
  as their own correctness reference.
- **Boundary p95** is the 95th percentile of pooled, bidirectional distances
  between opaque and reference boundary pixels. Units are source pixels. Raw
  outlines include uncertain borders. Empty predictions yield `null`, alongside
  complete missed coverage. They are never reported as zero boundary error.
- **Temporal error change** compares signed mask errors in adjacent frames of
  the same reference identity, after aligning reference bounding boxes. This
  removes ideal box translation and scaling, while still responding to pose
  changes and annotation noise. A context region extends 25 percent of the
  reference width/height beyond each side to capture outward boundary flicker.
  Unknown pixels outside either frame or in ignored regions do not contribute.
  `context` configures this fraction in the Python measurement function.
  It is a flicker proxy, not a perceptual score.
  Sparse non-adjacent frames do not produce temporal measurements.

Reports retain per-face results, tags, parsing/fallback status, and file hashes.
The overall missed-face rate is weighted by annotated facial area. Inspect
per-face and condition-specific failures because a large close-up can dominate
that average. A small development set cannot establish anonymity or exhaustive
detection accuracy. Unannotated faces do not contribute to these measurements.

## Additional pose and motion footage

[VidTIMIT](https://conradsanderson.id.au/vidtimit/) provides face recordings
with left/right and up/down head rotations across three sessions. It is useful
for profile and temporal stress testing. It does not provide ground-truth face
segmentation, so detection counts and model-landmark coverage on these videos
are diagnostics, not independently measured mask accuracy.

```bash
uv run python examples/prepare_vidtimit.py --download
```

This explicit acquisition downloads two pinned subject archives serially and
retains the original documentation with its usage conditions. It prepares all
six rotation sequences under `local_inputs/face_evaluation/vidtimit/prepared/`.
Every FFV1 video is decoded and compared pixel-for-pixel with the original JPEG
decodes. The manifest records frame mappings, SHA-256 hashes, and the documented
25 fps timing. Existing prepared data is preserved. Omit `--download` to work
offline with already acquired archives. Neither footage nor derived videos are
versioned or included in Anamorph's MIT license.

The upstream terms permit research usage with the notice retained and require
citation of C. Sanderson and B.C. Lovell, *Multi-Region Probabilistic Histograms
for Robust and Scalable Identity Inference*, LNCS 5558, 199–208, 2009, in
resulting publications. Consult the retained original documentation for the
complete conditions. Add independently reviewed facial outlines separately
before making face-area accuracy claims.
