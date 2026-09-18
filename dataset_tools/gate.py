"""Phase 1.2 completion gate: reviewed, byte-verified candidate pool, NOT permission to train."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from .common import (ROOT, checked_output_file, config, default_run, digest_file,
                     now, read_json, write_json)
from .prepare import load_audit
from .audit import verified_archive

REQUIRED_CHECKS = (
    'license_and_attribution_reviewed',
    'potholes_and_all_observed_crack_subtypes_visually_reviewed',
    'coordinate_mode_and_box_alignment_reviewed',
    'quarantine_and_exact_duplicate_policy_reviewed',
    'source_crop_groups_and_near_duplicate_limitations_reviewed',
    'publisher_vs_observed_counts_reconciled',
    'aerial_viewpoint_and_640px_visibility_reviewed',
    'esp32_camera_transfer_and_missing_negative_data_limits_acknowledged',
)


def verify_pool(pool: Path, audit_report: dict, audit_rows: list[dict]) -> dict:
    preparation = read_json(pool / 'preparation.json')
    if preparation.get('status') != 'UNSPLIT_CANDIDATE_POOL_PREPARED':
        raise ValueError('Preparation did not complete successfully.')
    for key, expected in [('audit_manifest_sha256', audit_report['manifest_sha256']),
                          ('archive_sha256', audit_report['archive_sha256']),
                          ('config_sha256', audit_report['config_sha256'])]:
        if preparation.get(key) != expected:
            raise ValueError(f'Preparation/audit mismatch: {key}')
    if digest_file(pool / 'manifest.jsonl') != preparation.get('pool_manifest_sha256'):
        raise ValueError('Candidate manifest changed after preparation.')
    expected_ids = {r['image_id'] for r in audit_rows if r['eligible_candidate']}
    originals = {r['image_id']: r for r in audit_rows}
    seen, pixels, images, labels = set(), set(), set(), set()
    objects = {'pothole': 0, 'crack': 0}
    with (pool / 'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            image_id = row['image_id']
            if image_id not in expected_ids or image_id in seen:
                raise ValueError('Unexpected or duplicate candidate ID.')
            seen.add(image_id)
            reference = originals[image_id]
            if row['sha256'] != reference['sha256'] or row['parent_group'] != reference['parent_group']:
                raise ValueError('Candidate provenance does not match the source audit.')
            if row['split'] is not None:
                raise ValueError('A split was unexpectedly assigned in Phase 1.2.')
            if row['pixel_sha256'] in pixels:
                raise ValueError('A duplicate decoded image survived candidate preparation.')
            pixels.add(row['pixel_sha256'])
            image = checked_output_file(pool, row['pool_image'])
            label = checked_output_file(pool, row['pool_label'])
            images.add(image.resolve())
            labels.add(label.resolve())
            if digest_file(image) != row['sha256'] or digest_file(label) != row['label_sha256']:
                raise ValueError(f'Candidate image or label was changed: {image_id}')
            lines = label.read_text(encoding='utf-8').splitlines()
            if len(lines) != len(reference['objects']):
                raise ValueError(f'Object count changed for {image_id}.')
            for text, expected in zip(lines, reference['objects']):
                fields = text.split()
                if len(fields) != 5 or fields[0] not in ('0', '1'):
                    raise ValueError('Invalid YOLO class/column count.')
                numbers = [float(x) for x in fields[1:]]
                if not all(math.isfinite(x) for x in numbers):
                    raise ValueError('Non-finite YOLO values.')
                xc, yc, w, h = numbers
                eps = 1e-8
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1
                        and xc - w / 2 >= -eps and yc - h / 2 >= -eps
                        and xc + w / 2 <= 1 + eps and yc + h / 2 <= 1 + eps):
                    raise ValueError('YOLO box lies outside normalized bounds.')
                class_name = ('pothole', 'crack')[int(fields[0])]
                if class_name != expected['target']:
                    raise ValueError('Mapped class changed from audit.')
                if any(abs(a - b) > 1e-8 for a, b in zip(numbers, expected['yolo_xywh'])):
                    raise ValueError('YOLO coordinate round-trip failed.')
                objects[class_name] += 1
    if seen != expected_ids:
        raise ValueError('Candidate image set is incomplete.')
    if images != {p.resolve() for p in (pool / 'images').iterdir() if p.is_file()}:
        raise ValueError('Untracked or missing image files in candidate pool.')
    if labels != {p.resolve() for p in (pool / 'labels').iterdir() if p.is_file()}:
        raise ValueError('Untracked or missing label files in candidate pool.')
    if min(objects.values()) <= 0:
        raise ValueError('Both pothole and crack must be present.')
    if len(seen) != preparation['images'] or objects != preparation['objects_per_class']:
        raise ValueError('Preparation counts did not reproduce from actual files.')
    return {'verified_images': len(seen), 'verified_objects_per_class': objects,
            'pixel_duplicates_in_pool': 0, 'all_labels_match_audit': True}


def review_pending(review: dict, audit_report: dict) -> list[str]:
    pending = []
    if review.get('manifest_sha256') != audit_report['manifest_sha256']:
        pending.append('Review is not tied to this exact audit manifest.')
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        pending.append('reviewer is blank')
    if not isinstance(review.get('notes'), str) or not review['notes'].strip():
        pending.append('notes are blank: document coordinate choice, exclusions, counts, and viewpoint findings')
    checks = review.get('checks', {})
    for name in REQUIRED_CHECKS:
        if checks.get(name) is not True:
            pending.append(name)
    observed = audit_report['observed']
    if observed['unknown_labels']:
        pending.append('Unknown label names need an explicit mapping/semantic decision and a fresh audit.')
    if observed['near_duplicate_list_truncated']:
        pending.append('Near-duplicate report was truncated; dataset grouping strategy needs revision.')
    if observed['invalid_xml'] or observed['orphan_xml']:
        pending.append('Invalid/orphan XML requires resolution before accepting this source version.')
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=default_run())
    parser.add_argument('--pool', type=Path, default=ROOT / 'data' / 'prepared' / 'uav_pdd2023_v4_candidate')
    args = parser.parse_args()
    output = args.audit / 'phase12_gate.json'
    write_json(output, {'status': 'IN_PROGRESS', 'started_at_utc': now(), 'train_ready': False})
    try:
        report, rows = load_audit(args.audit)
        hashes = verified_archive(Path(report['archive_path']), config())
        if hashes['sha256'] != report['archive_sha256']:
            raise ValueError('Source archive differs from the completed audit.')
        verified = verify_pool(args.pool, report, rows)
        pending = review_pending(read_json(args.audit / 'review_checklist.json'), report)
        result = {'status': 'REVIEW_REQUIRED' if pending else 'PHASE_1_2_PASS_AUDITED_POOL_ONLY',
                  'checked_at_utc': now(), 'source_id': report['source_id'], 'verified': verified,
                  'pending': pending, 'splits_created': False, 'train_ready': False,
                  'training_started': False,
                  'next_action': 'Return this report and audit.json for review. Do not train.'}
        write_json(output, result)
        print(json.dumps(result, indent=2))
        return 2 if pending else 0
    except Exception as exc:
        result = {'status': 'BLOCKED', 'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc(), 'train_ready': False, 'training_started': False}
        write_json(output, result)
        print(json.dumps(result, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
