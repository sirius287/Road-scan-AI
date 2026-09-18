from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'dataset_phase12.yaml'
GIB = 1024 ** 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest_file(path: Path, algorithm: str = 'sha256') -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_digests(path: Path) -> dict[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()  # MD5 matches publisher; SHA256 is also retained.
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    return {'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if value['classes'] != {0: 'pothole', 1: 'crack'}:
        raise ValueError('Phase 1 class contract must remain 0=pothole, 1=crack.')
    return value


def raw_archive() -> Path:
    cfg = config()
    return ROOT / 'data' / 'raw' / cfg['source_id'] / cfg['archive_name']


def default_run() -> Path:
    return ROOT / 'runs' / 'dataset_phase12'


def safe_member(name: str) -> str:
    """Validate archive names without extracting untrusted paths."""
    clean = name.replace('\\', '/')
    p = PurePosixPath(clean)
    if p.is_absolute() or any(part in ('..', '') or ':' in part for part in p.parts):
        raise ValueError(f'Unsafe archive member: {name!r}')
    if not p.parts or '\x00' in clean:
        raise ValueError(f'Invalid archive member: {name!r}')
    return p.as_posix()


def checked_output_file(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Output path escapes root: {relative!r}')
    return target


def provenance_text(cfg: dict[str, Any]) -> str:
    return (f"Source: {cfg['title']}\nCreators: {', '.join(cfg['creators'])}\n"
            f"Version: {cfg['version']}\nDOI: {cfg['doi']}\nRecord: {cfg['record_url']}\n"
            f"License: {cfg['license_id']} - {cfg['license_url']}\n"
            'Changes: two-class annotation mapping, conservative image exclusions, '
            'duplicate removal, normalized YOLO boxes, and renamed candidate images. '
            'Original image bytes are unchanged in the candidate pool. '
            'Review overlays/thumbnails are resized/annotated derivatives.\n'
            'No creator endorsement is implied. No model has been trained.\n')


def read_zip_member(archive: Any, name: str) -> bytes:
    """Read a validated logical member even if its ZIP uses Windows separators."""
    try:
        return archive.read(name)
    except KeyError:
        matches = [info for info in archive.infolist()
                   if not info.is_dir() and safe_member(info.filename) == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous archive member: {name}")
        return archive.read(matches[0])
