"""Tiny generated fixtures exercise code only; they are NOT a training dataset."""
from __future__ import annotations
from pathlib import Path
import json
import zipfile

import cv2
import numpy as np
import pytest

from dataset_tools.audit import audit_archive, decode_image, find_members
from dataset_tools.common import config, digest_file, safe_member
from dataset_tools.gate import REQUIRED_CHECKS, review_pending, verify_pool
from dataset_tools.prepare import prepare_pool
from dataset_tools.similarity import BKTree, dhash64
from dataset_tools.voc import alias_map, convert_box, normalize_name, parent_group, parse_xml, yolo_box


def xml(label='Pothole', filename='frame.jpg', box=(2, 3, 24, 20), difficult=0):
    return (f'<annotation><filename>{filename}</filename><size><width>32</width><height>24</height></size>'
            f'<object><name>{label}</name><difficult>{difficult}</difficult><bndbox>'
            f'<xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax>'
            '</bndbox></object></annotation>').encode()


def image_bytes(seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return encoded.tobytes()


def make_archive(path, entries):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def read_rows(run):
    return [json.loads(x) for x in (run / 'images.jsonl').read_text(encoding='utf-8').splitlines()]


def test_aliases_are_explicit():
    mapping = alias_map(config())
    assert mapping[normalize_name('  LONGITUDINAL   crack ')] == 'crack'
    assert mapping['ph'] == 'pothole'
    assert mapping['repair'] == 'non_target_repair'
    assert 'unrecognized' not in mapping
    assert '0' not in mapping


def test_voc_coordinate_conversion():
    assert convert_box([1, 1, 32, 24], 32, 24, 'voc-one-based-inclusive') == [0, 0, 32, 24]
    assert convert_box([0, 0, 32, 24], 32, 24, 'zero-based-edges') == [0, 0, 32, 24]
    assert yolo_box([0, 0, 32, 24], 32, 24) == [0.5, 0.5, 1, 1]


@pytest.mark.parametrize('box', [[-1, 0, 5, 5], [0, 0, 33, 20], [4, 2, 4, 7], [5, 2, 4, 7], [0, 0, float('nan'), 7]])
def test_bad_boxes_are_not_silently_clipped(box):
    with pytest.raises(ValueError):
        convert_box(box, 32, 24, 'zero-based-edges')


def test_zero_is_invalid_in_one_based_mode():
    with pytest.raises(ValueError):
        convert_box([0, 1, 8, 8], 32, 24, 'voc-one-based-inclusive')


def test_xml_entity_rejected():
    with pytest.raises(ValueError):
        parse_xml(b'<!DOCTYPE a [<!ENTITY x SYSTEM "file:///etc/passwd">]><annotation/>')


def test_crop_parent_group():
    first = parent_group('JPEGImages/lr_00001_top_left.jpg', 'uav')
    second = parent_group('JPEGImages/lr_00001_bottom_right.jpg', 'uav')
    assert first == second
    assert first[0] == 'uav:lr_00001'
    assert parent_group('abc.jpg', 'uav')[1].startswith('single_stem')


@pytest.mark.parametrize('name', ['../bad.jpg', '/bad.jpg', 'C:\\bad.jpg', 'a/../../bad.xml'])
def test_unsafe_archive_member(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_bktree_matches_brute_force():
    values = [0, 1, 2, 3, 7, 255, 256, 1024, 0]
    tree = BKTree()
    for i, value in enumerate(values):
        assert set(tree.query(value, 2)) == {(str(j), (value ^ old).bit_count())
                                            for j, old in enumerate(values[:i]) if (value ^ old).bit_count() <= 2}
        tree.add(value, str(i))


def test_decode_invalid_image():
    with pytest.raises(ValueError):
        decode_image(b'not an image')


def test_quarantine_and_preparation_roundtrip(tmp_path):
    entries = {}
    labels = {'lr_00001_top_left': 'Pothole', 'lr_00001_bottom_right': 'Transverse crack',
              'repair': 'Repair', 'unknown': 'mystery', 'difficult': 'Pothole'}
    for i, (stem, label) in enumerate(labels.items()):
        entries[f'JPEGImages/{stem}.jpg'] = image_bytes(i)
        entries[f'Annotations/{stem}.xml'] = xml(label, stem + '.jpg', difficult=int(stem == 'difficult'))
    entries['JPEGImages/unlabeled.jpg'] = image_bytes(50)
    archive = make_archive(tmp_path / 'fixture.zip', entries)
    run = tmp_path / 'audit'
    report = audit_archive(archive, run, config(), 'zero-based-edges', preview_limit=3)
    rows = read_rows(run)
    obs = report['observed']
    assert obs['image_files'] == 6
    assert obs['candidate_images'] == 2
    assert obs['candidate_object_counts'] == {'pothole': 1, 'crack': 1}
    assert obs['unknown_labels'] == {'mystery': 1}
    assert (run / 'preview' / 'index.html').exists()
    pool = tmp_path / 'pool'
    prepared = prepare_pool(archive, pool, report, rows, config())
    assert prepared['images'] == 2
    verified = verify_pool(pool, report, rows)
    assert verified['verified_objects_per_class'] == {'pothole': 1, 'crack': 1}
    assert not list(pool.rglob('*.yaml'))
    assert all(row['split'] is None for row in [json.loads(x) for x in (pool / 'manifest.jsonl').read_text().splitlines()])
    label = next((pool / 'labels').glob('*.txt'))
    label.write_text('0 0.5 0.5 1 1\n')
    with pytest.raises(ValueError, match='changed'):
        verify_pool(pool, report, rows)


def test_duplicate_conflicts_quarantine_group(tmp_path):
    data = image_bytes(3)
    entries = {'images/a.jpg': data, 'images/b.jpg': data,
               'annotations/a.xml': xml('Pothole', 'a.jpg'),
               'annotations/b.xml': xml('Transverse crack', 'b.jpg')}
    archive = make_archive(tmp_path / 'fixture.zip', entries)
    report = audit_archive(archive, tmp_path / 'audit', config(), 'zero-based-edges', 0)
    assert report['observed']['candidate_images'] == 0
    assert report['observed']['quarantined_images'] == 2


def test_equivalent_duplicates_keep_one(tmp_path):
    data = image_bytes(3)
    entries = {'images/a.jpg': data, 'images/b.jpg': data,
               'annotations/a.xml': xml('Pothole', 'a.jpg'), 'annotations/b.xml': xml('Pothole', 'b.jpg')}
    archive = make_archive(tmp_path / 'fixture.zip', entries)
    report = audit_archive(archive, tmp_path / 'audit', config(), 'zero-based-edges', 0)
    assert report['observed']['candidate_images'] == 1
    assert report['observed']['equivalent_duplicate_images_removed'] == 1


def test_xml_size_mismatch(tmp_path):
    annotation = xml().replace(b'<width>32</width>', b'<width>33</width>')
    archive = make_archive(tmp_path / 'fixture.zip', {'images/frame.jpg': image_bytes(), 'annotations/frame.xml': annotation})
    report = audit_archive(archive, tmp_path / 'audit', config(), 'zero-based-edges', 0)
    assert report['observed']['candidate_images'] == 0
    assert report['observed']['quarantine_reasons_overlapping_counts']['xml_image_size_mismatch'] == 1


def test_empty_annotation_is_not_negative(tmp_path):
    annotation = b'<annotation><size><width>32</width><height>24</height></size></annotation>'
    archive = make_archive(tmp_path / 'fixture.zip', {'images/frame.jpg': image_bytes(), 'annotations/frame.xml': annotation})
    report = audit_archive(archive, tmp_path / 'audit', config(), 'zero-based-edges', 0)
    assert report['observed']['candidate_images'] == 0
    assert report['observed']['quarantine_reasons_overlapping_counts']['empty_xml_not_verified_negative'] == 1


def test_case_collision_rejected(tmp_path):
    archive = make_archive(tmp_path / 'fixture.zip', {'images/A.jpg': b'x', 'images/a.jpg': b'y'})
    with zipfile.ZipFile(archive) as opened:
        with pytest.raises(ValueError):
            find_members(opened)


def test_review_requires_real_booleans_and_notes():
    report = {'manifest_sha256': 'abc', 'observed': {'unknown_labels': {}, 'near_duplicate_list_truncated': False,
                                                   'invalid_xml': 0, 'orphan_xml': 0}}
    review = {'manifest_sha256': 'abc', 'reviewer': 'Fixture reviewer', 'notes': 'Code-test fixture only.',
              'checks': {name: True for name in REQUIRED_CHECKS}}
    assert review_pending(review, report) == []
    review['checks'][REQUIRED_CHECKS[0]] = 'true'
    assert REQUIRED_CHECKS[0] in review_pending(review, report)
    review['manifest_sha256'] = 'changed'
    assert any('manifest' in message for message in review_pending(review, report))


def test_no_overwrite(tmp_path):
    run = tmp_path / 'existing'
    run.mkdir()
    with pytest.raises(FileExistsError):
        audit_archive(tmp_path / 'not_needed.zip', run, config(), 'zero-based-edges')


def test_windows_separator_zip_supported(tmp_path):
    entries = {'images\\p.jpg': image_bytes(1), 'annotations\\p.xml': xml('Pothole', 'p.jpg'),
               'images\\c.jpg': image_bytes(2), 'annotations\\c.xml': xml('LC', 'c.jpg')}
    archive = make_archive(tmp_path / 'windows.zip', entries)
    run = tmp_path / 'audit'
    report = audit_archive(archive, run, config(), 'zero-based-edges', 2)
    assert report['observed']['candidate_images'] == 2
    pool = tmp_path / 'pool'
    prepare_pool(archive, pool, report, read_rows(run), config())
    assert verify_pool(pool, report, read_rows(run))['verified_images'] == 2


def test_metadata_license_and_checksum_checks(monkeypatch):
    import io
    from dataset_tools import download
    cfg = config()
    raw = {'id': cfg['record_id'], 'metadata': {'license': {'id': 'cc-by-4.0'}},
           'files': [{'key': cfg['archive_name'], 'checksum': 'md5:' + cfg['archive_md5'], 'size': download.PINNED_SIZE}]}
    monkeypatch.setattr(download, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(raw).encode()))
    _, verified = download.fetch_metadata(cfg)
    assert verified['size_bytes'] == download.PINNED_SIZE
    raw['files'][0]['checksum'] = 'md5:wrong'
    with pytest.raises(ValueError, match='checksum'):
        download.fetch_metadata(cfg)
    raw['files'][0]['checksum'] = 'md5:' + cfg['archive_md5']
    raw['metadata']['license']['id'] = 'other'
    with pytest.raises(ValueError, match='License'):
        download.fetch_metadata(cfg)


def test_resume_download_and_final_file_not_promoted_before_hash(tmp_path, monkeypatch):
    import io
    from dataset_tools import download
    path = tmp_path / 'archive.zip'
    path.with_name('archive.zip.part').write_bytes(b'abc')
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 3-5/6'}
    seen = []
    def mock(request, timeout):
        seen.append(request.get_header('Range'))
        return Response(b'def')
    monkeypatch.setattr(download, 'urlopen', mock)
    download.download_file('https://zenodo.org/example', path, 6)
    assert path.with_name('archive.zip.part').read_bytes() == b'abcdef'
    assert seen == ['bytes=3-']
    assert not path.exists()


def test_invalid_resume_is_rejected(tmp_path, monkeypatch):
    import io
    from dataset_tools import download
    path = tmp_path / 'archive.zip'
    path.with_name('archive.zip.part').write_bytes(b'abc')
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 0-5/6'}
    monkeypatch.setattr(download, 'urlopen', lambda *args, **kwargs: Response(b'abcdef'))
    with pytest.raises(ValueError, match='Content-Range'):
        download.download_file('https://zenodo.org/example', path, 6)
    assert path.with_name('archive.zip.part').read_bytes() == b'abc'
