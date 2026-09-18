"""Downloader-only regressions. No internet, GPU, dataset download or pip changes."""
from __future__ import annotations

import http.client
import io
import json
from pathlib import Path
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from dataset_tools import download as d
from dataset_tools.common import config
from dataset_tools.download_watchdog import NetworkFailure, supervise

URL = d.PINS['archive_url']


class Response(io.BytesIO):
    def __init__(self, body: bytes, status=200, headers=None, url=URL):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.url = url


def metadata_fixture():
    cfg = config()
    return {'id': cfg['record_id'], 'metadata': {'license': {'id': 'cc-by-4.0'}},
            'files': [{'key': cfg['archive_name'], 'checksum': 'md5:' + cfg['archive_md5'],
                       'size': d.PINNED_SIZE}]}


def test_new_transfer_writes_part_not_final(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'abcdef', headers={'Content-Length': '6'}))
    dest = tmp_path / 'archive.zip'
    d.download_file(URL, dest, 6)
    assert (tmp_path / 'archive.zip.part').read_bytes() == b'abcdef'
    assert not dest.exists()


def test_uses_read1_and_never_large_buffered_read(tmp_path, monkeypatch):
    class StreamingOnly(Response):
        def read(self, *a, **k):
            raise AssertionError('Large buffered read() must not be used for transfers.')
        def read1(self, amount=-1):
            assert amount == d.CHUNK_BYTES
            return super().read1(min(amount, 2))
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: StreamingOnly(b'abcdef'))
    d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert (tmp_path / 'archive.zip.part').read_bytes() == b'abcdef'


def test_actual_http_stream_writes_before_rest_of_body_arrives(tmp_path, monkeypatch):
    """Real socket HTTPResponse, first 1 KiB followed by a deliberately delayed tail."""
    client, server = socket.socketpair()
    client.settimeout(4)
    server.settimeout(4)
    release = threading.Event()
    body = b'a' * 65536
    failures = []
    def send():
        try:
            server.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 65536\r\n'
                           b'Content-Type: application/zip\r\nConnection: close\r\n\r\n' + body[:1024])
            release.wait(4)
            server.sendall(body[1024:])
        finally:
            server.close()
    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    response = http.client.HTTPResponse(client)
    response.begin()
    response.url = URL
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: response)
    def receive():
        try:
            d.download_file(URL, tmp_path / 'archive.zip', len(body), attempts=1)
        except Exception as exc:
            failures.append(exc)
    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    part = tmp_path / 'archive.zip.part'
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and (not part.exists() or part.stat().st_size == 0):
            time.sleep(0.01)
        assert part.exists() and part.stat().st_size == 1024
        assert receiver.is_alive(), 'Rest of body was not yet released.'
    finally:
        release.set()
        receiver.join(5)
        sender.join(5)
        client.close()
    assert not failures
    assert not receiver.is_alive()
    assert part.read_bytes() == body


@pytest.mark.parametrize('headers,message', [
    ({'Content-Length': '5'}, 'Content-Length'),
    ({'Content-Length': 'broken'}, 'Content-Length'),
    ({'Content-Type': 'text/html'}, 'Content-Type'),
    ({'Content-Type': 'application/problem+json'}, 'Content-Type'),
    ({'Content-Encoding': 'gzip'}, 'Content-Encoding'),
    ({'Content-Range': 'bytes 0-5/6'}, 'Content-Range'),
])
def test_bad_full_response_rejected_before_part_is_opened(tmp_path, monkeypatch, headers, message):
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'abcdef', headers=headers))
    with pytest.raises(ValueError, match=message):
        d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert not (tmp_path / 'archive.zip.part').exists()


@pytest.mark.parametrize('content_range', ['bytes 0-5/6', 'bytes 3-5/7', 'bytes 3-4/6', 'bytes 3-5/*', 'garbage'])
def test_entire_resume_range_is_validated(tmp_path, monkeypatch, content_range):
    part = tmp_path / 'archive.zip.part'
    part.write_bytes(b'abc')
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'def', 206, {'Content-Range': content_range}))
    with pytest.raises(ValueError, match='Content-Range'):
        d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert part.read_bytes() == b'abc'


def test_resume_ignored_never_truncates_existing_bytes(tmp_path, monkeypatch):
    part = tmp_path / 'archive.zip.part'
    part.write_bytes(b'abc')
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'abcdef', 200))
    with pytest.raises(ValueError, match='NOT overwritten'):
        d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert part.read_bytes() == b'abc'


def test_retry_resumes_only_saved_bytes_after_disconnect(tmp_path, monkeypatch):
    calls = []
    def mock(request, timeout):
        calls.append(request.get_header('Range'))
        if len(calls) == 1:
            return Response(b'abc')  # premature EOF
        return Response(b'def', 206, {'Content-Range': 'bytes 3-5/6', 'Content-Length': '3'})
    monkeypatch.setattr(d, 'urlopen', mock)
    monkeypatch.setattr(d.time, 'sleep', lambda _: None)
    d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert calls == [None, 'bytes=3-']
    assert (tmp_path / 'archive.zip.part').read_bytes() == b'abcdef'


def test_complete_part_needs_no_http_request_but_is_not_promoted(tmp_path, monkeypatch):
    part = tmp_path / 'archive.zip.part'
    part.write_bytes(b'abcdef')
    def fail(*a, **k):
        raise AssertionError('No HTTP request is needed at exact size.')
    monkeypatch.setattr(d, 'urlopen', fail)
    d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert not (tmp_path / 'archive.zip').exists()


def test_size_overflow_never_written(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'abcdefg'))
    with pytest.raises(ValueError, match='exceeds'):
        d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert (tmp_path / 'archive.zip.part').stat().st_size == 0


def test_metadata_rejects_different_exact_size(monkeypatch):
    raw = metadata_fixture()
    raw['files'][0]['size'] += 1
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(json.dumps(raw).encode(), url=d.PINS['metadata_url']))
    with pytest.raises(ValueError, match='Published size changed'):
        d.fetch_metadata(config())


@pytest.mark.parametrize('key,value', [
    ('record_id', 123), ('archive_name', 'other.zip'), ('archive_md5', '0'*32),
    ('archive_url', 'https://other.example/file.zip'), ('version', 'v3'),
    ('archive_size_bytes', d.PINNED_SIZE - 1),
])
def test_config_pin_changes_are_rejected(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError, match='Pinned'):
        d.validate_pins(cfg)


@pytest.mark.parametrize('url', [
    'http://zenodo.org/records/8429208/files/UAV-PDD2023.zip',
    'https://mirror.example/records/8429208/files/UAV-PDD2023.zip',
    'https://zenodo.org:444/records/8429208/files/UAV-PDD2023.zip',
    'https://zenodo.org/records/123/files/UAV-PDD2023.zip',
    'https://name:secret@zenodo.org/records/8429208/files/UAV-PDD2023.zip',
])
def test_unsafe_redirect_stopped_before_following(url):
    with pytest.raises(ValueError):
        d._PinnedRedirect().redirect_request(Request(URL), None, 302, '', {}, url)


def test_same_file_canonical_redirect_allowed():
    result = d._PinnedRedirect().redirect_request(Request(URL), None, 302, '', {}, URL.split('?')[0])
    assert result.full_url == URL.split('?')[0]


def test_probe_reads_only_small_prefix_if_range_ignored(monkeypatch):
    data = b'PK\x03\x04' + b'a' * (d.CHUNK_BYTES * 2)
    response = Response(data, 200, {'Content-Length': str(len(data))})
    seen = []
    def mock(request, timeout):
        seen.append(request.get_header('Range'))
        return response
    monkeypatch.setattr(d, 'urlopen', mock)
    result = d.probe_archive(URL, len(data))
    assert seen == ['bytes=0-65535']
    assert result['bytes_read'] == d.CHUNK_BYTES
    assert result['archive_verified'] is False


def test_probe_requires_zip_signature(monkeypatch):
    monkeypatch.setattr(d, 'urlopen', lambda *a, **k: Response(b'not a zip'))
    with pytest.raises(ValueError, match='signature'):
        d.probe_archive(URL, 9)


def test_rate_limit_is_preserved_and_classified():
    exc = HTTPError(URL, 429, 'Too Many Requests', {'Retry-After': '60'}, None)
    result = d._network_failure(exc)
    assert result.retryable
    assert result.retry_after == 60


def test_forbidden_is_not_retried(tmp_path, monkeypatch):
    seen = []
    def mock(*a, **k):
        seen.append(1)
        raise HTTPError(URL, 403, 'Forbidden', {}, None)
    monkeypatch.setattr(d, 'urlopen', mock)
    with pytest.raises(NetworkFailure, match='403'):
        d.download_file(URL, tmp_path / 'archive.zip', 6)
    assert len(seen) == 1


def test_socket_failure_retries_are_bounded(tmp_path, monkeypatch):
    seen = []
    def mock(*a, **k):
        seen.append(1)
        raise URLError('simulated DNS failure')
    monkeypatch.setattr(d, 'urlopen', mock)
    monkeypatch.setattr(d.time, 'sleep', lambda _: None)
    with pytest.raises(NetworkFailure, match='DNS'):
        d.download_file(URL, tmp_path / 'archive.zip', 6, attempts=3)
    assert len(seen) == 3


def test_wrong_sized_local_file_rejected_before_hashing(tmp_path, monkeypatch):
    p = tmp_path / 'tiny.zip'
    p.write_bytes(b'tiny')
    def fail(*a, **k):
        raise AssertionError('Wrong-sized file should not be hashed.')
    monkeypatch.setattr(d, 'archive_digests', fail)
    with pytest.raises(ValueError, match='expected exactly'):
        d.verify_candidate(p, config())


def test_bad_checksum_rejected(tmp_path, monkeypatch):
    p = tmp_path / 'six.zip'
    p.write_bytes(b'abcdef')
    monkeypatch.setattr(d, 'PINNED_SIZE', 6)  # code-only fixture, never changes the project config
    with pytest.raises(ValueError, match='MD5'):
        d.verify_candidate(p, config())


def test_lock_excludes_second_writer_and_cleans_up(tmp_path):
    p = tmp_path / '.download.lock'
    with d.download_lock(p):
        assert json.loads(p.read_text())['pid'] > 0
        with pytest.raises(FileExistsError, match='lock exists'):
            with d.download_lock(p):
                pytest.fail('Second writer must not acquire the lock.')
    assert not p.exists()


def test_lock_is_cleaned_on_keyboard_interrupt(tmp_path):
    p = tmp_path / '.download.lock'
    with pytest.raises(KeyboardInterrupt):
        with d.download_lock(p):
            raise KeyboardInterrupt
    assert not p.exists()


def _stalled_worker(flag: Path, *, _notify=None):
    flag.write_text('network operation began')
    _notify({'event': 'stage', 'stage': 'simulated_blocking_body_read'})
    time.sleep(30)


def _successful_worker(*, _notify=None):
    _notify({'event': 'bytes', 'bytes': 4})
    return {'fixture': True}


def test_spawn_watchdog_terminates_stalled_child(tmp_path):
    started = time.monotonic()
    report = tmp_path / 'network.json'
    with pytest.raises(NetworkFailure, match='No saved-byte progress'):
        supervise(_stalled_worker, (tmp_path / 'started.txt',), stall_timeout=1.5,
                  heartbeat=0.3, report=report)
    assert time.monotonic() - started < 12
    assert json.loads(report.read_text())['status'] == 'FAILED'


def test_spawn_watchdog_returns_success(tmp_path):
    result = supervise(_successful_worker, stall_timeout=10, report=tmp_path / 'network.json')
    assert result == {'fixture': True}
    assert json.loads((tmp_path / 'network.json').read_text())['status'] == 'NETWORK_OPERATION_COMPLETE'
