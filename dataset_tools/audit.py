"""Audit actual archive bytes. No extraction, split assignment, ML or training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import random
import stat
import sys
import zipfile

import cv2
import numpy as np

from .common import (CONFIG, archive_digests, config, default_run, digest_file,
                     now, provenance_text, raw_archive, read_json, read_zip_member, safe_member, write_json)
from .similarity import BKTree, dhash64
from .voc import MODES, alias_map, convert_box, normalize_name, parent_group
from .voc import parse_xml, yolo_box

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 100_000_000


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image is empty or exceeds the per-file safety limit.')
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.size == 0:
        raise ValueError('OpenCV could not decode the image.')
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ValueError('Decoded image exceeds the pixel safety limit.')
    return image


def find_members(archive: zipfile.ZipFile):
    images, annotations, other = {}, {}, []
    seen = set()
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = safe_member(info.filename)
        if name.casefold() in seen:
            raise ValueError(f'Case-insensitive duplicate ZIP member: {name}')
        seen.add(name.casefold())
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'Symlink entry is not supported: {name}')
        if info.flag_bits & 1:
            raise ValueError('Encrypted ZIP entries are not supported.')
        suffix = PurePosixPath(name).suffix.casefold()
        if suffix in IMAGE_EXTENSIONS:
            if info.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f'Oversize image entry: {name}')
            images[name] = info
        elif suffix == '.xml':
            if info.file_size > 5 * 1024 * 1024:
                raise ValueError(f'Oversize XML entry: {name}')
            annotations[name] = info
        else:
            other.append({'member': name, 'size_bytes': info.file_size})
    return images, annotations, other


def select_previews(rows: list[dict], limit: int) -> list[dict]:
    """Cover observed source classes and quarantine reasons before filling randomly."""
    if limit < 0:
        raise ValueError('Preview limit cannot be negative.')
    usable = [r for r in rows if r.get('width') and r.get('sha256')]
    usable.sort(key=lambda r: r['image_id'])
    rng = random.Random(42)
    rng.shuffle(usable)
    buckets = defaultdict(list)
    for row in usable:
        for label in set(row.get('source_labels', [])):
            buckets['class:' + label].append(row)
        if row['quarantine_reasons']:
            buckets['quarantine'].append(row)
    chosen = {}
    # Pothole examples get priority in the review set; this is not dataset oversampling.
    order = sorted(buckets, key=lambda x: (not ('pothole' in x.casefold() or x.casefold() == 'class:ph'), x))
    for key in order:
        for row in buckets[key][:min(12, limit)]:
            if len(chosen) < limit:
                chosen[row['image_id']] = row
    for row in usable:
        if len(chosen) >= limit:
            break
        chosen.setdefault(row['image_id'], row)
    return list(chosen.values())


def previews(archive: zipfile.ZipFile, rows: list[dict], output: Path, cfg: dict, limit: int) -> list[str]:
    folder = output / 'preview'
    folder.mkdir()
    originals, overlays, scaled = folder / 'originals', folder / 'overlays', folder / 'view640'
    for p in (originals, overlays, scaled):
        p.mkdir()
    cards, ids = [], []
    for row in select_previews(rows, limit):
        data = read_zip_member(archive, row['image_member'])
        try:
            image = decode_image(data)
        except ValueError:
            continue
        image_id = row['image_id']
        suffix = PurePosixPath(row['image_member']).suffix.casefold()
        original_name = image_id + suffix
        (originals / original_name).write_bytes(data)
        scale = min(1.0, 1280 / max(image.shape[:2]))
        canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for obj in row['objects']:
            box = obj.get('xyxy')
            if not box:
                continue
            x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
            color = (0, 165, 255) if obj.get('target') == 'crack' else (180, 0, 180)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['source_label']} -> {obj.get('target', 'unknown')}"
            cv2.putText(canvas, label, (max(0, x1), max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError('Could not encode preview overlay.')
        (overlays / f'{image_id}.jpg').write_bytes(encoded.tobytes())
        s640 = min(1.0, 640 / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=s640, fy=s640, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError('Could not encode review image.')
        (scaled / f'{image_id}.jpg').write_bytes(encoded.tobytes())
        xml_link = ''
        if row.get('annotation_member'):
            xml_file = originals / f'{image_id}.xml'
            xml_file.write_bytes(read_zip_member(archive, row['annotation_member']))
            xml_link = f' | <a href="originals/{image_id}.xml">Original XML</a>'
        text = json.dumps({k: row[k] for k in ('image_member', 'parent_group', 'source_labels',
                                                'quarantine_reasons')}, indent=2)
        cards.append(f'<article><h2>{html.escape(image_id)}</h2><pre>{html.escape(text)}</pre>'
                     f'<a href="originals/{original_name}">Original image, unchanged</a>{xml_link}'
                     f'<p>Overlay (display preview, not training data):</p>'
                     f'<a href="overlays/{image_id}.jpg"><img loading="lazy" src="overlays/{image_id}.jpg"></a>'
                     f'<p>Unannotated view: longest edge at most 640 pixels; not an ESP32 simulation.</p>'
                     f'<a href="view640/{image_id}.jpg"><img loading="lazy" src="view640/{image_id}.jpg"></a>'
                     '</article>')
        ids.append(image_id)
    head = ('<!doctype html><html><head><meta charset="utf-8"><title>Phase 1.2 dataset review</title>'
            '<style>body{font-family:Arial,sans-serif;max-width:1280px;margin:auto;padding:24px}'
            'article{border-top:2px solid #777;padding:20px 0}img{max-width:100%;height:auto}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>'
            '<h1>Actual dataset annotation review</h1>'
            '<p>Publisher annotations, not AI predictions. Candidate pool only; no model, splits, or metrics.</p>'
            '<p>Check potholes, every crack subtype, repair regions, missing damage, coordinates, '
            'and repeated scenes. Inspect original images at native resolution.</p>')
    tail = '<h2>Attribution</h2><pre>' + html.escape(provenance_text(cfg)) + '</pre></body></html>'
    (folder / 'index.html').write_text(head + '\n'.join(cards) + tail, encoding='utf-8')
    return ids


def audit_archive(archive_path: Path, output: Path, cfg: dict, coordinate_mode: str,
                  preview_limit: int = 80, near_radius: int = 4, archive_sha: str = '') -> dict:
    """Public function for tests. The CLI separately enforces the pinned archive receipt."""
    if output.exists():
        raise FileExistsError(f'Use a new audit directory; never overwrite review evidence: {output}')
    if not 0 <= near_radius <= 8:
        raise ValueError('Near-duplicate radius must be between 0 and 8.')
    output.mkdir(parents=True)
    write_json(output / 'audit.json', {'status': 'IN_PROGRESS', 'started_at_utc': now()})
    aliases = alias_map(cfg)
    rows, issues, exact = [], [], []
    counts_source = Counter()
    counts_target = Counter()
    sizes, invalid_xml = {}, {}
    annotations = {}
    by_stem_xml, by_stem_image = defaultdict(list), defaultdict(list)
    coordinate_evidence = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        image_members, xml_members, others = find_members(archive)
        if not image_members or not xml_members:
            raise ValueError('Archive must contain images and VOC XML annotations.')
        for name in sorted(image_members):
            by_stem_image[PurePosixPath(name).stem.casefold()].append(name)
        for name, info in sorted(xml_members.items()):
            by_stem_xml[PurePosixPath(name).stem.casefold()].append(name)
            try:
                ann = parse_xml(archive.read(info))
                annotations[name] = ann
                for obj in ann['objects']:
                    counts_source[obj['source_label']] += 1
                    mapped = aliases.get(normalize_name(obj['source_label']))
                    if mapped:
                        counts_target[mapped] += 1
                    box = obj['source_xyxy']
                    coordinate_evidence['objects_with_zero_xmin_or_ymin'] += int(box[0] == 0 or box[1] == 0)
                    coordinate_evidence['objects_with_one_xmin_or_ymin'] += int(box[0] == 1 or box[1] == 1)
                    coordinate_evidence['objects_with_fractional_coordinates'] += int(any(v != int(v) for v in box))
            except Exception as exc:
                invalid_xml[name] = str(exc)
                issues.append({'member': name, 'issue': 'invalid_xml', 'detail': str(exc)})
        orphan_xml = [name for name in xml_members if PurePosixPath(name).stem.casefold() not in by_stem_image]
        for name in orphan_xml:
            issues.append({'member': name, 'issue': 'orphan_annotation', 'detail': 'No image shares this stem.'})
        for index, (name, info) in enumerate(sorted(image_members.items()), start=1):
            stem = PurePosixPath(name).stem.casefold()
            image_id = cfg['source_id'] + '_' + hashlib.sha256(name.encode('utf-8')).hexdigest()[:20]
            group, group_basis = parent_group(name, cfg['source_id'])
            row = {'image_id': image_id, 'source_id': cfg['source_id'], 'image_member': name,
                   'annotation_member': None, 'width': None, 'height': None, 'bytes': info.file_size,
                   'sha256': None, 'pixel_sha256': None, 'dhash64': None, 'objects': [], 'source_labels': [],
                   'parent_group': group, 'group_basis': group_basis, 'coordinate_mode': coordinate_mode,
                   'quarantine_reasons': [], 'warnings': [], 'canonical_duplicate_of': None}
            rows.append(row)
            if len(by_stem_image[stem]) > 1:
                row['quarantine_reasons'].append('ambiguous_image_stem')
            matches = by_stem_xml.get(stem, [])
            if len(matches) != 1:
                row['quarantine_reasons'].append('missing_annotation' if not matches else 'ambiguous_annotation_stem')
            else:
                row['annotation_member'] = matches[0]
            try:
                data = archive.read(info)
                row['sha256'] = hashlib.sha256(data).hexdigest()
                image = decode_image(data)
                height, width = image.shape[:2]
                row.update({'width': width, 'height': height,
                            'pixel_sha256': hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
                            'dhash64': f'{dhash64(image):016x}'})
                sizes[(width, height)] = sizes.get((width, height), 0) + 1
                # Different EXIF orientation at training decode would invalidate unchanged XML coordinates.
                default_oriented = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if default_oriented is None or not np.array_equal(default_oriented, image):
                    row['quarantine_reasons'].append('exif_orientation_requires_review')
                del default_oriented
                if PurePosixPath(name).suffix.casefold() in ('.jpg', '.jpeg') and b'\xff\xd9' not in data[-32:]:
                    row['quarantine_reasons'].append('jpeg_end_marker_missing_or_unusual_trailer')
                ann = annotations.get(row['annotation_member'])
                if row['annotation_member'] in invalid_xml:
                    row['quarantine_reasons'].append('invalid_xml')
                if ann is not None:
                    if (ann['xml_width'], ann['xml_height']) != (width, height):
                        row['quarantine_reasons'].append('xml_image_size_mismatch')
                    declared = ann['declared_filename'].replace('\\', '/')
                    if declared and PurePosixPath(declared).stem.casefold() != stem:
                        row['quarantine_reasons'].append('xml_declared_filename_mismatch')
                    if not ann['objects']:
                        row['quarantine_reasons'].append('empty_xml_not_verified_negative')
                    for original in ann['objects']:
                        obj = dict(original)
                        obj['target'] = aliases.get(normalize_name(obj['source_label']))
                        row['source_labels'].append(obj['source_label'])
                        if obj['target'] is None:
                            row['quarantine_reasons'].append('unknown_label')
                        elif obj['target'] == 'non_target_repair':
                            row['quarantine_reasons'].append('contains_repair')
                        if obj['difficult']:
                            row['quarantine_reasons'].append('difficult_object')
                        if obj['truncated']:
                            row['warnings'].append('truncated_object_retained_if_box_valid')
                        try:
                            obj['xyxy'] = convert_box(obj['source_xyxy'], width, height, coordinate_mode)
                            obj['yolo_xywh'] = yolo_box(obj['xyxy'], width, height)
                            obj['short_box_edge_at_640'] = min(obj['xyxy'][2] - obj['xyxy'][0],
                                                             obj['xyxy'][3] - obj['xyxy'][1]) * 640 / max(width, height)
                        except ValueError as exc:
                            obj['xyxy'] = None
                            obj['invalid_reason'] = str(exc)
                            row['quarantine_reasons'].append('invalid_box')
                        row['objects'].append(obj)
                del image, data
            except Exception as exc:
                row['quarantine_reasons'].append('unreadable_image')
                row['warnings'].append(str(exc))
            row['quarantine_reasons'] = sorted(set(row['quarantine_reasons']))
            if index % 250 == 0:
                print(f'Audited {index} / {len(image_members)} images', flush=True)
        # Exact visual duplicates: eliminate only with matching original annotation signatures.
        by_pixels = defaultdict(list)
        for row in rows:
            if row['pixel_sha256']:
                by_pixels[row['pixel_sha256']].append(row)
        for cluster in by_pixels.values():
            if len(cluster) < 2:
                continue
            def signature(row):
                return json.dumps(sorted((normalize_name(o['source_label']), o['source_xyxy'],
                                          o['difficult'], o['truncated']) for o in row['objects']))
            signatures = {signature(row) for row in cluster}
            # Mismatched/missing annotations do not get a fabricated union of their labels.
            conflict = len(signatures) != 1 or any(r['quarantine_reasons'] for r in cluster)
            canonical = sorted(cluster, key=lambda r: r['image_member'])[0]
            for row in cluster:
                if conflict:
                    row['quarantine_reasons'].append('duplicate_annotation_conflict_or_invalid_member')
                elif row is not canonical:
                    row['canonical_duplicate_of'] = canonical['image_id']
                if row is not canonical:
                    exact.append({'image_id_a': canonical['image_id'], 'image_id_b': row['image_id'],
                                  'same_file_bytes': row['sha256'] == canonical['sha256'],
                                  'annotation_conflict': conflict})
        by_id = {r['image_id']: r for r in rows}
        tree = BKTree()
        near_pairs = []
        near_truncated = False
        for row in rows:
            if row['dhash64'] is None:
                continue
            value = int(row['dhash64'], 16)
            for other_id, distance in tree.query(value, near_radius):
                other = by_id[other_id]
                if row['pixel_sha256'] == other['pixel_sha256']:
                    continue
                if len(near_pairs) >= 50000:
                    near_truncated = True
                    break
                near_pairs.append({'image_id_a': other_id, 'image_id_b': row['image_id'],
                                   'hamming_distance': distance,
                                   'same_parent_group': other['parent_group'] == row['parent_group']})
            tree.add(value, row['image_id'])
            if near_truncated:
                break
        for row in rows:
            row['quarantine_reasons'] = sorted(set(row['quarantine_reasons']))
            target_objects = [o for o in row['objects'] if o.get('target') in ('pothole', 'crack') and o.get('xyxy')]
            row['eligible_candidate'] = bool(target_objects) and not row['quarantine_reasons'] and not row['canonical_duplicate_of']
            row['candidate_class_counts'] = dict(Counter(o['target'] for o in target_objects)) if row['eligible_candidate'] else {}
            for reason in row['quarantine_reasons']:
                issues.append({'member': row['image_member'], 'issue': reason, 'detail': row['image_id']})
        preview_ids = previews(archive, rows, output, cfg, preview_limit)
    counts_candidate = Counter()
    images_candidate = Counter()
    groups_candidate = defaultdict(set)
    boxes_at_640 = defaultdict(list)
    for row in rows:
        if row['eligible_candidate']:
            counts_candidate.update(row['candidate_class_counts'])
            images_candidate.update(row['candidate_class_counts'].keys())
            for label in row['candidate_class_counts']:
                groups_candidate[label].add(row['parent_group'])
            for obj in row['objects']:
                if obj.get('target') in ('pothole', 'crack') and obj.get('xyxy'):
                    boxes_at_640[obj['target']].append(obj['short_box_edge_at_640'])
    original_counts = Counter(label for row in rows for label in set(row['source_labels']))
    unknown = {name: count for name, count in counts_source.items() if normalize_name(name) not in aliases}
    with (output / 'images.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_csv(output / 'images.csv', [{**r, 'source_labels': ';'.join(r['source_labels']),
                                      'quarantine_reasons': ';'.join(r['quarantine_reasons'])} for r in rows],
              ['image_id', 'image_member', 'annotation_member', 'width', 'height', 'bytes', 'sha256',
               'pixel_sha256', 'dhash64', 'parent_group', 'group_basis', 'source_labels',
               'eligible_candidate', 'quarantine_reasons', 'canonical_duplicate_of'])
    write_csv(output / 'class_counts.csv', [{'source_label': name, 'mapped_to': aliases.get(normalize_name(name), 'UNKNOWN'),
                                           'instances_in_parseable_xml': count,
                                           'paired_image_count': original_counts[name]}
                                          for name, count in sorted(counts_source.items())],
              ['source_label', 'mapped_to', 'instances_in_parseable_xml', 'paired_image_count'])
    write_csv(output / 'issues.csv', issues, ['member', 'issue', 'detail'])
    write_csv(output / 'exact_duplicates.csv', exact, ['image_id_a', 'image_id_b', 'same_file_bytes', 'annotation_conflict'])
    write_csv(output / 'near_duplicates.csv', near_pairs,
              ['image_id_a', 'image_id_b', 'hamming_distance', 'same_parent_group'])
    write_json(output / 'auxiliary_archive_files.json', others)
    label_geometry = {}
    for label, values in boxes_at_640.items():
        label_geometry[label] = {'minimum_short_box_edge_px': min(values),
                                 'p10_short_box_edge_px': float(np.percentile(values, 10)),
                                 'median_short_box_edge_px': float(np.median(values)),
                                 'boxes_short_edge_below_4px': sum(v < 4 for v in values),
                                 'caution': 'Box extent is not crack width or evidence of visibility.'}
    manifest_sha = digest_file(output / 'images.jsonl')
    report = {
        'status': 'AUDIT_COMPLETE_REVIEW_REQUIRED', 'source_id': cfg['source_id'], 'created_at_utc': now(),
        'archive_path': str(archive_path.resolve()), 'archive_sha256': archive_sha or digest_file(archive_path),
        'config_sha256': digest_file(CONFIG), 'manifest_sha256': manifest_sha,
        'coordinate_mode': coordinate_mode, 'coordinate_evidence': dict(coordinate_evidence),
        'publisher_reported_not_local': cfg.get('publisher_reported_not_locally_measured', {}),
        'observed': {'image_files': len(rows), 'xml_files': len(xml_members), 'parseable_xml': len(annotations),
                     'invalid_xml': len(invalid_xml), 'orphan_xml': len(orphan_xml),
                     'source_object_counts_parseable_xml': dict(sorted(counts_source.items())),
                     'target_object_counts_before_exclusions': dict(counts_target),
                     'unknown_labels': unknown,
                     'image_dimensions': [{'width': w, 'height': h, 'images': n} for (w, h), n in sorted(sizes.items())],
                     'candidate_images': sum(r['eligible_candidate'] for r in rows),
                     'candidate_object_counts': {c: counts_candidate[c] for c in ('pothole', 'crack')},
                     'candidate_images_per_class': {c: images_candidate[c] for c in ('pothole', 'crack')},
                     'candidate_parent_groups_per_class': {c: len(groups_candidate[c]) for c in ('pothole', 'crack')},
                     'quarantined_images': sum(bool(r['quarantine_reasons']) for r in rows),
                     'quarantine_reasons_overlapping_counts': dict(Counter(reason for r in rows for reason in r['quarantine_reasons'])),
                     'equivalent_duplicate_images_removed': sum(bool(r['canonical_duplicate_of']) for r in rows),
                     'exact_duplicate_pairs': len(exact), 'near_duplicate_candidate_pairs': len(near_pairs),
                     'near_duplicate_list_truncated': near_truncated,
                     'parent_groups_all_images': len(set(r['parent_group'] for r in rows)),
                     'group_basis_counts': dict(Counter(r['group_basis'] for r in rows))},
        'candidate_box_geometry_at_640': label_geometry,
        'preview': {'count': len(preview_ids), 'image_ids': preview_ids},
        'policies': cfg['policies'],
        'limitations': ['dHash is a review heuristic, not proof of unique scenes.',
                        'Filename parents do not prove route/flight independence.',
                        'No train/val/test splits or model metrics exist.',
                        'Image/annotation instances do not count unique physical damage.',
                        'No hard negatives are accepted automatically.'],
        'training_started': False, 'train_ready': False}
    write_json(output / 'audit.json', report)
    write_json(output / 'review_checklist.json', {
        'manifest_sha256': manifest_sha, 'reviewer': '', 'notes': '',
        'checks': {'license_and_attribution_reviewed': False,
                   'potholes_and_all_observed_crack_subtypes_visually_reviewed': False,
                   'coordinate_mode_and_box_alignment_reviewed': False,
                   'quarantine_and_exact_duplicate_policy_reviewed': False,
                   'source_crop_groups_and_near_duplicate_limitations_reviewed': False,
                   'publisher_vs_observed_counts_reconciled': False,
                   'aerial_viewpoint_and_640px_visibility_reviewed': False,
                   'esp32_camera_transfer_and_missing_negative_data_limits_acknowledged': False}})
    (output / 'ATTRIBUTION.txt').write_text(provenance_text(cfg), encoding='utf-8')
    return report


def verified_archive(path: Path, cfg: dict) -> dict:
    receipt_path = path.parent / 'download_receipt.json'
    receipt = read_json(receipt_path)
    if receipt.get('status') != 'VERIFIED' or receipt.get('license_explicitly_accepted') is not True:
        raise ValueError('Archive requires a verified download receipt and explicit license acceptance.')
    hashes = archive_digests(path)
    if hashes['md5'] != cfg['archive_md5'] or hashes['sha256'] != receipt.get('sha256'):
        raise ValueError('Archive checksum failed. No audit or preparation will use these bytes.')
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=raw_archive())
    parser.add_argument('--out', type=Path, default=default_run())
    parser.add_argument('--coordinate-mode', choices=MODES, required=True,
                        help='Explicit interpretation; inspect evidence and overlays before approving it.')
    parser.add_argument('--preview-limit', type=int, default=80)
    parser.add_argument('--near-radius', type=int, default=4)
    args = parser.parse_args()
    cfg = config()
    hashes = verified_archive(args.archive, cfg)
    report = audit_archive(args.archive, args.out, cfg, args.coordinate_mode,
                           args.preview_limit, args.near_radius, hashes['sha256'])
    print(json.dumps(report['observed'], indent=2))
    print(f"Audit executed successfully: {args.out / 'audit.json'}")
    print('AUDIT_COMPLETE_REVIEW_REQUIRED. Exit 0 means execution success, not dataset acceptance.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'AUDIT BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
