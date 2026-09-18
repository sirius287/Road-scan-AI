# Pothole Detection Drone — Phase 1.2

Scope: verify provenance, download one official real dataset, audit actual images/XML,
and prepare an UNSPLIT two-class candidate pool. NO training, GPU use, model download,
new dependency installation, final training split, or change to Phase 1.1 files.

The recommended first audit candidate is UAV-PDD2023 v4 (Zenodo 8429208). It is NOT
approved as a training dataset until the local checks and human review are complete.
The package contains code, tests, configuration and documentation only — no dataset.

## Add this patch without overwriting existing files

Save the ZIP in Downloads. Use this PowerShell block from the existing project.
No virtual-environment activation or execution-policy change is needed.

```powershell
Set-Location "$HOME\source\pothole-drone"
$ErrorActionPreference = "Stop"
$Project = (Get-Location).Path
$Stage = Join-Path $env:TEMP ("pothole-phase12-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -LiteralPath "$HOME\Downloads\pothole-drone-phase1-2.zip" -DestinationPath $Stage
$Stage = (Get-Item $Stage).FullName
$Incoming = Get-ChildItem -LiteralPath $Stage -File -Recurse
foreach ($File in $Incoming) {
    $Relative = $File.FullName.Substring($Stage.Length).TrimStart('\')
    $Destination = Join-Path $Project $Relative
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite existing file: $Destination"
    }
}
foreach ($File in $Incoming) {
    $Relative = $File.FullName.Substring($Stage.Length).TrimStart('\')
    $Destination = Join-Path $Project $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Destination
}
```

The preflight checks every incoming file before copying. `SHA256SUMS-PHASE-1.2.json`
is separate from Phase 1.1's checksum manifest. A repeated installation deliberately
refuses to overwrite; do not delete working files to bypass it.

## Validated environment stays unchanged

Use the existing `.venv` on Windows 11. The scripts use the standard library,
OpenCV, NumPy and PyYAML already in the validated environment. Do not upgrade pip,
PyTorch, CUDA, Ultralytics, or any dependencies for this phase.

From the existing project directory in Windows PowerShell:

```powershell
Set-Location "$HOME\source\pothole-drone"
.\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve the existing dependency issue before proceeding.' }
.\.venv\Scripts\python.exe -m pytest -q .\tests\test_phase12_dataset.py
if ($LASTEXITCODE -ne 0) { throw 'Phase 1.2 tooling tests failed.' }
```

Expected test result: 28 passed. Fixture images are generated in pytest temporary
folders for code tests only. They are not real road data or a training dataset.

## 1. Read the source and license, then inspect metadata

```powershell
Start-Process "https://zenodo.org/records/8429208"
Start-Process "https://creativecommons.org/licenses/by/4.0/"
.\.venv\Scripts\python.exe -m dataset_tools.download --metadata-only
if ($LASTEXITCODE -ne 0) { throw 'Source metadata check failed; see the browser fallback below.' }
Get-Content .\data\raw\uav_pdd2023_v4\source_check.json
```

This step downloads metadata only. It verifies record ID, CC-BY-4.0 license ID,
archive name, publisher MD5 and exact byte size. The original API response is saved
as `zenodo_metadata.json`. No API key or account is required by this downloader.

Published archive MD5 for this exact release:
`0745d017ce2dce23bd73944e9b0acec7`.

A changed checksum/version/license is a STOP condition, not permission to select a
mirror or change the expected checksum until it passes.

## 2. Download and verify the archive

Only after reviewing the license:

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.download --accept-license CC-BY-4.0
if ($LASTEXITCODE -ne 0) { throw 'Download/integrity check failed. Do not audit unverified bytes.' }
Get-Content .\data\raw\uav_pdd2023_v4\download_receipt.json
```

The downloader uses chunked reads, bounded retries and validated HTTP Range resume.
It retains a `.part` file after interrupted transfers and checks the completed file
against the publisher MD5 before accepting it. It also records a local SHA256.
No ZIP extraction is needed. Audit reads entries directly from the original archive.

Reserve roughly 12 GiB free as a conservative workspace budget, not a measured
archive expansion size. The downloader calculates a free-space requirement from the
live exact archive size; preparation separately checks capacity for candidate copies.
Do not put `.venv` or large datasets in a cloud-synchronized folder where avoidable.

### Browser fallback when the API/download endpoint is unavailable

Open the exact record above in your browser, read its license, and download its
`UAV-PDD2023.zip` to Downloads. Then run:

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.download --verify-local "$HOME\Downloads\UAV-PDD2023.zip" --accept-license CC-BY-4.0
if ($LASTEXITCODE -ne 0) { throw 'The local ZIP is not the pinned official v4 archive.' }
```

This fallback makes no HTTP calls, verifies the pinned publisher checksum, and records
`live_metadata_verified: false` rather than pretending the API check succeeded.
Do not use earlier releases or third-party re-exports with this command. If a browser
renames the file with `(1)`, supply its real path. This mode copies verified bytes
without overwriting an existing different raw archive.

## 3. Audit every image and annotation

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.audit --coordinate-mode zero-based-edges --preview-limit 80
if ($LASTEXITCODE -ne 0) { throw 'Audit execution failed. Preserve the diagnostic output.' }
Get-Content .\runs\dataset_phase12\audit.json
Import-Csv .\runs\dataset_phase12\class_counts.csv | Format-Table -AutoSize
Start-Process .\runs\dataset_phase12\preview\index.html
```

Exit 0 here means the audit executed, NOT that the dataset is accepted.
Expected status is `AUDIT_COMPLETE_REVIEW_REQUIRED`.

The command makes an explicit PROVISIONAL interpretation of coordinates:
`zero-based-edges` means x spans 0..image_width, y spans 0..image_height, and boxes
use continuous edges `[xmin,ymin,xmax,ymax]`. It is not a claim that all VOC datasets
use this convention. The report records zero/one boundary evidence, retains the exact
source coordinates, and shows overlays. Lack of zero coordinates alone does not
prove a one-based convention. Confirm the convention from the annotations/authoring
pipeline and visual review; do not mark it confirmed merely because parsing passed.

If evidence establishes conventional one-based inclusive VOC coordinates instead,
run a fresh audit (do not overwrite the earlier evidence):

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.audit --coordinate-mode voc-one-based-inclusive --out .\runs\dataset_phase12_voc --preview-limit 80
```

This alternate conversion subtracts 1 from xmin/ymin but not from xmax/ymax, mapping
inclusive pixel indices to half-open image-edge coordinates. A zero origin is
invalid in that mode. Do not switch modes just to hide invalid boxes. All subsequent
commands must then point to the same chosen audit with `--audit .\runs\dataset_phase12_voc`.

### What is checked

- Readable images; original byte and decoded-pixel SHA256; dimensions; unusual JPEG
  endings; EXIF orientation differences; declared XML/image size and filename matches.
- Explicit source labels, target mapping, difficult/truncated flags, finite and valid
  boxes, normalized bounds; no silent clipping or conversion of unknown labels.
- Missing/empty XML, orphan annotations, malformed XML, unsafe archive paths,
  case-insensitive filename collisions and ambiguous same-stem matches.
- Exact decoded-image duplicates and annotation conflicts. Equivalent duplicates
  keep one canonical image; conflicting/invalid duplicate groups are quarantined.
- Known quadrant-crop parent suffixes. Near-duplicate candidate pairs use 64-bit
  dHash, Hamming radius 4, and a BK-tree. This is NOT a physical damage tracker.
- Actual raw label counts, remaining candidate class/image/parent counts, and box
  extent at a hypothetical 640-pixel longest edge. Box extent is not crack width.

Each image is decoded individually. No entire image dataset is held in RAM and no
GPU or training batch is used. dHash can miss rotations, overlapping crops, scene
changes and repeated nearby road surfaces; neither it nor filenames proves route
independence. No train/val/test independence is asserted at this stage.

### Conservative exclusion policy

`Pothole/PH -> 0`. Longitudinal, transverse, alligator and oblique cracks map to `1`.
Explicit full-name/code aliases are recorded in the YAML, with case/whitespace-only
normalization. An unfamiliar actual label is reported and quarantined, never guessed.

Images containing any `Repair/RP` are quarantined as a whole. We do not relabel a
repair as a pothole/crack or silently remove only its box while leaving potentially
ambiguous supervision in the image. This may remove useful positives; the audit
reports that cost before the policy is reconsidered.

Empty XML and missing XML are not automatically verified damage-free negatives.
They are quarantined. Difficult objects, invalid boxes and unmatched sizes/names
also quarantine the whole image. Valid truncated objects retain their boxes and flags.
The resulting pool is positive-only and cannot establish real-world false-alarm rates.
Additional verified negative coverage needs a later, explicit data decision.

## 4. Review the real evidence

Read `audit.json`, `class_counts.csv`, `issues.csv`, `exact_duplicates.csv`,
`near_duplicates.csv` and the preview HTML. The preview selects up to 80 actual images,
covering observed source classes and representative quarantines; it is not all data.
The HTML links unchanged originals and XML plus annotation overlays and a 640-pixel
view for visibility inspection. Reopen originals at native size. Preview images are
not added to the candidate dataset.

Review potholes and every observed crack subtype, repair/non-target exclusions,
missed annotations, shadows, wet surfaces, very thin cracks, and crop boundaries.
Check that valid potholes remain after exclusions. Counts are annotation instances,
not unique physical potholes. Compare publisher-reported totals with observed totals:
a mismatch needs an explanation, not an invented replacement count.

For a broader visual review, run another audit to a new output directory with a
larger `--preview-limit`. Do not overwrite reviewed files or approve all checklist
fields automatically.

Edit the human checklist only after doing the corresponding checks:

```powershell
notepad .\runs\dataset_phase12\review_checklist.json
```

Fill `reviewer` and `notes`, and set each completed check to the JSON boolean `true`
(not the string `"true"`). In notes, record the coordinate-convention evidence,
quarantine decisions, publisher/local count reconciliation and viewpoint/visibility
limitations. Leave unresolved checks `false` and return the report for discussion.
Missing context or a suspicious box is a valid reason to stop, not to force a pass.

## 5. Prepare the unsplit candidate pool

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.prepare
if ($LASTEXITCODE -ne 0) { throw 'Candidate preparation was blocked. Do not train.' }
Get-Content .\data\prepared\uav_pdd2023_v4_candidate\preparation.json
```

Both classes must have at least one valid retained instance, otherwise preparation
stops. This minimum is only a functionality check, not a sufficient sample-size
criterion for training or statistical evaluation.

Original image bytes are copied unchanged under collision-resistant names. YOLO
labels, source labels/boxes, coordinate mode, source paths, hashes, parent groups,
and `split: null` are retained. No hardlinks, symlinks, augmentation, image resizing,
weights or training YAML are created. Source bytes and originals remain in the ZIP.
The output contains `NOT_TRAIN_READY.txt`.

With the alternate audit, also use a fresh pool:

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.prepare --audit .\runs\dataset_phase12_voc --out .\data\prepared\uav_pdd2023_v4_candidate_voc
```

## 6. Run the Phase 1.2 gate and stop

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.gate
$Phase12Exit = $LASTEXITCODE
Write-Host "Phase 1.2 gate exit code: $Phase12Exit"
Get-Content .\runs\dataset_phase12\phase12_gate.json
```

For alternate paths:

```powershell
.\.venv\Scripts\python.exe -m dataset_tools.gate --audit .\runs\dataset_phase12_voc --pool .\data\prepared\uav_pdd2023_v4_candidate_voc
```

Exit meanings:
- 0: `PHASE_1_2_PASS_AUDITED_POOL_ONLY` — byte/schema checks and the supplied manual
  checklist pass. Return the reports before proceeding; this is NOT permission to train.
- 2: `REVIEW_REQUIRED` — unresolved manual checks or semantic/source issues remain.
- 1: `BLOCKED` — an integrity, file, parsing or preparation error prevents acceptance.

The gate independently hashes candidate files, verifies labels against original audit
coordinates/classes, checks round trips and candidate-set completeness, and rejects
unknown/unresolved labels and unreviewed evidence. It does not train or create splits.
Human checklist acceptance is recorded, not independently verified by the program.

Return: `audit.json`, `class_counts.csv`, `phase12_gate.json`, the reviewer notes,
and relevant overlay/original examples when labels look wrong. Do NOT proceed to
training. The next approved work will handle leakage-aware train/val/test preparation
and any unresolved coverage issues, preserving this audit.

## File layout

- `dataset_tools/common.py`: provenance, configuration and safe file/hash helpers.
- `dataset_tools/download.py`: official metadata, resumable transfer and receipt.
- `dataset_tools/voc.py`: strict source parser, aliases and coordinate conversion.
- `dataset_tools/similarity.py`: dHash and bounded-memory candidate lookup.
- `dataset_tools/audit.py`: full local audit, measured manifests and review previews.
- `dataset_tools/prepare.py`: conservative, unsplit candidate pool.
- `dataset_tools/gate.py`: independent verification and manual-review gate.
- `configs/dataset_phase12.yaml`: exact source/version/license and mapping contract.
- `tests/test_phase12_dataset.py`: offline fixture and mocked-network tests.
- `docs/PHASE12-SOURCE-REVIEW.md`: publisher/source audit, with explicit uncertainty.
- `docs/PHASE12-TEST-RESULTS.json`: tests actually performed in the authoring runtime.

Keep large data, runs, `.venv`, secrets and model outputs out of Git. This patch does
not overwrite `.gitignore`; verify your existing ignore rules before committing.
