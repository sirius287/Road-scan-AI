"""Strict Pascal-VOC parsing and explicit, reversible coordinate conversion."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from typing import Any

MODES = ('zero-based-edges', 'voc-one-based-inclusive')


def normalize_name(value: str) -> str:
    return ' '.join(value.strip().casefold().split())


def alias_map(cfg: dict) -> dict[str, str]:
    result = {}
    for target, aliases in cfg['label_aliases'].items():
        for name in aliases:
            name = normalize_name(str(name))
            if name in result and result[name] != target:
                raise ValueError(f'Conflicting class alias: {name}')
            result[name] = target
    return result


def parse_xml(data: bytes) -> dict[str, Any]:
    if len(data) > 5 * 1024 * 1024:
        raise ValueError('Annotation exceeds 5 MiB safety limit.')
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('DTD/entity declarations are not allowed in annotation XML.')
    root = ET.fromstring(data)
    if root.tag != 'annotation':
        raise ValueError('XML root must be annotation.')
    width, height = int(root.findtext('size/width', '0')), int(root.findtext('size/height', '0'))
    if width <= 0 or height <= 0:
        raise ValueError('XML size must be positive.')
    objects = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        if not name:
            raise ValueError('Object class name is empty.')
        box = []
        for field in ('xmin', 'ymin', 'xmax', 'ymax'):
            value = float(obj.findtext(f'bndbox/{field}', 'nan'))
            if not math.isfinite(value):
                raise ValueError(f'Invalid {field} for {name}.')
            box.append(value)
        difficult = int(obj.findtext('difficult', '0'))
        truncated = int(obj.findtext('truncated', '0'))
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise ValueError('difficult/truncated must be 0 or 1 when supplied.')
        objects.append({'source_label': name, 'source_xyxy': box,
                        'difficult': difficult, 'truncated': truncated})
    return {'xml_width': width, 'xml_height': height,
            'declared_filename': root.findtext('filename', ''), 'objects': objects}


def convert_box(box: list[float], width: int, height: int, mode: str) -> list[float]:
    if mode not in MODES:
        raise ValueError(f'Unsupported coordinate mode: {mode}')
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError('Invalid box or image dimensions.')
    x1, y1, x2, y2 = box
    if mode == 'voc-one-based-inclusive':
        if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
            raise ValueError('Box is invalid under 1-based inclusive VOC coordinates.')
        x1, y1 = x1 - 1, y1 - 1
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Box is empty, inverted, negative, or outside the decoded image.')
    return [x1, y1, x2, y2]


def yolo_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = convert_box(box, width, height, 'zero-based-edges')
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def parent_group(member: str, source_id: str) -> tuple[str, str]:
    stem = PurePosixPath(member).stem
    # These suffixes are present in the publisher's v4 archive preview.
    match = re.fullmatch(r'(.+)_(?:top|bottom)_(?:left|right)', stem, flags=re.IGNORECASE)
    if match:
        return f"{source_id}:{match.group(1).casefold()}", 'filename_crop_parent_inferred'
    return f'{source_id}:{stem.casefold()}', 'single_stem_fallback_not_verified_independent'
