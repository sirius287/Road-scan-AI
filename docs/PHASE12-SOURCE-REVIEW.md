# Publisher/source review — 18 September 2026

This is a review of primary publisher material, NOT a local byte audit of the datasets.
Full-archive transfer was unavailable in the authoring runtime. The shipped code
performs the byte/class audit on the user's laptop; all operational counts come from
that execution. No training data or metric is fabricated in this package.

## First audit candidate: UAV-PDD2023 v4

Authors: HaoHui Yan and JunFei Zhang. Exact Zenodo record 8429208, version v4.
The official record declares CC BY 4.0, VOC box annotations, UAV capture at 30 m,
six source categories (four crack types, pothole, repair), approximately 2.1 GB of
files, 2,440 images and 11,158 instances. These counts are publisher-reported and
include source classes excluded or merged by our project, not measured candidate counts.

https://zenodo.org/records/8429208
https://zenodo.org/records/8429208/preview/UAV-PDD2023.zip?include_deleted=0

The publicly readable preview contains quadrant-suffixed filenames, such as
`lr_00001_top_left.xml` and `lr_00001_bottom_right.xml`. The preview says it does NOT
show every file; it cannot establish full per-class coverage. Crop-parent grouping
is therefore a necessary precaution, not proof of road/flight independence.

## Alternatives considered, not downloaded or selected

RDD2022: the maintainer offers a China_Drone ZIP and labeled VOC training data, but
the overall RDD class list does not establish the per-class coverage of that subset.
The README states CC BY-SA 4.0 for images, distinct from its MIT repository code
license. Exact release licensing and local subset coverage should be reviewed before
mixing it into a future dataset. Unlabeled challenge test images are not negatives.

https://github.com/sekilab/RoadDamageDetector

HighRPD v1: the official Mendeley record declares CC BY 4.0, YOLO boxes and classes
line/block/pit. Its images are 640-pixel crops from higher-resolution UAV imagery.
It is a plausible aerial extension; its class semantics and parent-image grouping
need their own audit rather than an unverified direct merge with pothole/crack.

https://data.mendeley.com/datasets/sywswj7djj/1

## Domain fit and limitations

The original UAV-PDD2023 paper describes UAV imagery across different road categories
and weather conditions. This supports evaluating it ahead of a ground-view-only
source; it does not establish ESP32-CAM resolution, motion blur or detection quality.

https://pubmed.ncbi.nlm.nih.gov/38020429/

The project must later evaluate real ESP32-CAM images at the actual operating
altitude, speed and encoding settings. A high-quality aerial crop is not equivalent
to an entire low-cost camera frame, even when both are resized to 640 pixels.
No flight altitude or physical pothole dimensions are prescribed by this audit.

## License handling

CC BY 4.0 requires appropriate attribution, a license link and indication of changes;
additional rights may still matter. Preserve authors, DOI, version and transformation
notes. Dataset, code and model licenses are separate records.

https://creativecommons.org/licenses/by/4.0/
