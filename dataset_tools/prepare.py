"""Create an UNSPLIT two-class candidate pool. Never creates a training dataset YAML."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

from .common import (CONFIG, GIB, ROOT, checked_output_file, config, default_run,
                     digest_file, now, provenance_text, read_json, read_zip_member, write_json)
from .audit import verified_archive
from .voc import yolo_box


def load_audit(run: Path) -> tuple[dict, list[dict]]:
    report = read_json(run / 'audit.json')
    if report.get('status') != 'AUDIT_COMPLETE_REVIEW_REQUIRED':
        raise ValueError('Audit is missing, incomplete, or failed.')
    if digest_file(run / 'images.jsonl') != report['manifest_sha256']:
        raise ValueError('Image manifest changed since audit. Rerun into a fresh audit directory.')
    if digest_file(CONFIG) != report['config_sha256']:
        raise ValueError('Class/source configuration changed since audit. Rerun the audit.')
    rows = [json.loads(line) for line in (run / 'images.jsonl').read_text(encoding='utf-8').splitlines() if line]
    return report, rows


def prepare_pool(archive_path: Path, output: Path, report: dict, rows: list[dict], cfg: dict) -> dict:
    if output.exists():
        raise FileExistsError(f'Use a new candidate directory; no overwrite: {output}')
    candidates = [r for r in rows if r['eligible_candidate']]
    counts = {name: sum(r['candidate_class_counts'].get(name, 0) for r in candidates)
              for name in ('pothole', 'crack')}
    if not candidates or min(counts.values()) <= 0:
        raise ValueError(f'Both target classes must survive the audit. Candidate objects: {counts}')
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(r['bytes'] for r in candidates) + GIB
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f'Insufficient free space: candidate copies plus 1 GiB reserve require {required / GIB:.2f} GiB.')
    output.mkdir()
    write_json(output / 'preparation.json', {'status': 'IN_PROGRESS', 'started_at_utc': now()})
    (output / 'images').mkdir()
    (output / 'labels').mkdir()
    name_to_id = {name: index for index, name in cfg['classes'].items()}
    manifests = []
    with zipfile.ZipFile(archive_path) as archive:
        for row in candidates:
            image_id = row['image_id']
            suffix = PurePosixPath(row['image_member']).suffix.casefold()
            image_relative = f'images/{image_id}{suffix}'
            label_relative = f'labels/{image_id}.txt'
            data = read_zip_member(archive, row['image_member'])
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError(f'Source image changed: {row["image_member"]}')
            image_path = checked_output_file(output, image_relative)
            image_path.write_bytes(data)
            lines = []
            for obj in row['objects']:
                if obj['target'] not in name_to_id or not obj.get('xyxy'):
                    raise ValueError('Eligible image unexpectedly contains a non-target/invalid object.')
                box = yolo_box(obj['xyxy'], row['width'], row['height'])
                lines.append(str(name_to_id[obj['target']]) + ' ' + ' '.join(f'{v:.10f}' for v in box))
            label_path = checked_output_file(output, label_relative)
            label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            manifests.append({**row, 'pool_image': image_relative, 'pool_label': label_relative,
                              'label_sha256': digest_file(label_path), 'split': None})
    manifest = output / 'manifest.jsonl'
    with manifest.open('w', encoding='utf-8') as stream:
        for row in manifests:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(output / 'classes.json', {'0': 'pothole', '1': 'crack'})
    (output / 'ATTRIBUTION.txt').write_text(provenance_text(cfg), encoding='utf-8')
    (output / 'NOT_TRAIN_READY.txt').write_text(
        'Unsplit, positive-only candidate pool. No training was performed.\n'
        'Do not feed this whole pool into model training.\n'
        'Phase 1.2 review must pass, then a separate leakage-aware split procedure must be implemented.\n'
        'Known crop siblings and duplicate relationships must remain together in later splits.\n'
        'GPS, route identity, calibrated dimensions, severity, and unique-pothole IDs are not supplied.\n',
        encoding='utf-8')
    result = {'status': 'UNSPLIT_CANDIDATE_POOL_PREPARED', 'source_id': cfg['source_id'],
              'created_at_utc': now(), 'audit_manifest_sha256': report['manifest_sha256'],
              'archive_sha256': report['archive_sha256'], 'config_sha256': report['config_sha256'],
              'pool_manifest_sha256': digest_file(manifest), 'images': len(manifests),
              'objects_per_class': counts, 'splits_created': False, 'train_ready': False,
              'training_started': False}
    write_json(output / 'preparation.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=default_run())
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'prepared' / 'uav_pdd2023_v4_candidate')
    args = parser.parse_args()
    cfg = config()
    report, rows = load_audit(args.audit)
    archive = Path(report['archive_path'])
    hashes = verified_archive(archive, cfg)
    if hashes['sha256'] != report['archive_sha256']:
        raise ValueError('The archive no longer matches the completed audit.')
    result = prepare_pool(archive, args.out, report, rows, cfg)
    print(json.dumps(result, indent=2))
    print('Unsplit candidate pool only. No model or training split has been created.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'PREPARATION BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
