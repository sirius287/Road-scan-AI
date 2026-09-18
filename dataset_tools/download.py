r"""Pinned UAV-PDD2023 v4 downloader, patch 1.2.1 (standard library only).

PowerShell: .\.venv\Scripts\python.exe -u -m dataset_tools.download --accept-license CC-BY-4.0
Use --probe with the same license flag to read at most 64 KiB without touching .part.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .common import GIB, archive_digests, config, now, raw_archive, write_json, provenance_text
from .download_watchdog import NetworkFailure, supervise

PATCH_VERSION = '1.2.1'
PINNED_SIZE = 2_122_615_013
PINS = {
    'record_id': 8429208,
    'version': 'v4',
    'source_id': 'uav_pdd2023_v4',
    'archive_name': 'UAV-PDD2023.zip',
    'archive_md5': '0745d017ce2dce23bd73944e9b0acec7',
    'record_url': 'https://zenodo.org/records/8429208',
    'metadata_url': 'https://zenodo.org/api/records/8429208',
    'archive_url': 'https://zenodo.org/records/8429208/files/UAV-PDD2023.zip?download=1',
}
HEADERS = {'User-Agent': 'PotholeDroneDatasetAudit/1.2.1', 'Accept-Encoding': 'identity'}
CHUNK_BYTES = 64 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
HTTP_RETRYABLE = {408, 429, 500, 502, 503, 504}


def validate_pins(cfg: dict) -> None:
    for key, expected in PINS.items():
        if cfg.get(key) != expected:
            raise ValueError(f'Pinned {key} differs: expected {expected!r}; no substitution permitted.')
    if cfg.get('license_id', '').casefold() != 'cc-by-4.0':
        raise ValueError('Pinned License must be CC-BY-4.0.')
    if cfg.get('archive_size_bytes', PINNED_SIZE) != PINNED_SIZE:
        raise ValueError('Pinned size must remain 2122615013 bytes.')


def _safe_url(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname != 'zenodo.org'
            or parts.port not in (None, 443) or parts.username or parts.password or parts.fragment):
        raise ValueError('Only HTTPS zenodo.org on port 443 is permitted; no mirror or credentials.')
    return parts.path, parts.query


def _same_resource(before: str, after: str) -> None:
    path, query = _safe_url(before)
    new_path, new_query = _safe_url(after)
    if path != new_path or new_query not in {query, '', 'download=1'}:
        raise ValueError(f'Redirect would change the configured resource: {after!r}. STOP.')


class _PinnedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _same_resource(req.full_url, newurl)
        print(f'[REDIRECT] Same resource: {newurl}', flush=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(request: Request, timeout: float = 20):
    """Keep the legacy monkeypatch point while enforcing redirects before following."""
    _safe_url(request.full_url)
    # Default HTTPS handler retains certificate checks and system proxy handling.
    return build_opener(_PinnedRedirect()).open(request, timeout=timeout)


def _response_info(response, requested_url: str) -> dict:
    final_url = getattr(response, 'url', None) or requested_url
    _same_resource(requested_url, final_url)
    headers = getattr(response, 'headers', {})
    return {'url': final_url, 'http_status': getattr(response, 'status', 200),
            **{k: headers.get(k) for k in ('Content-Type', 'Content-Length', 'Content-Range',
                                           'Content-Encoding', 'Retry-After')}}


def _emit(notify, **message) -> None:
    if notify is not None:
        notify(message)


def _retry_after(value: str | None) -> float:
    if not value:
        return 0
    try:
        delay = float(value)
        if not math.isfinite(delay):
            return 0
        return max(0, delay)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0


def _network_failure(exc: Exception) -> NetworkFailure:
    if isinstance(exc, HTTPError):
        delay = _retry_after(exc.headers.get('Retry-After')) if exc.headers else 0
        exc.close()
        return NetworkFailure(f'HTTP {exc.code}: {exc.reason}; Retry-After={delay:g}s',
                              retryable=exc.code in HTTP_RETRYABLE, retry_after=delay)
    reason = getattr(exc, 'reason', exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return NetworkFailure(f'TLS certificate verification failed: {reason}. Do not disable TLS checks.')
    return NetworkFailure(f'{type(exc).__name__}: {exc}', retryable=True)


def fetch_metadata(cfg: dict, *, socket_timeout: float = 20, _notify=None) -> tuple[dict, dict]:
    validate_pins(cfg)
    print('[METADATA] Requesting the pinned official record...', flush=True)
    _emit(_notify, event='stage', stage='metadata_connecting')
    try:
        with urlopen(Request(cfg['metadata_url'], headers=HEADERS), timeout=socket_timeout) as response:
            info = _response_info(response, cfg['metadata_url'])
            if info['http_status'] != 200:
                raise ValueError(f"Unexpected metadata HTTP status: {info['http_status']}")
            _emit(_notify, event='stage', stage='metadata_body', response=info)
            body = bytearray()
            while chunk := response.read1(min(CHUNK_BYTES, MAX_METADATA_BYTES + 1 - len(body))):
                body.extend(chunk)
                _emit(_notify, event='bytes', bytes=len(body))
                if len(body) > MAX_METADATA_BYTES:
                    raise ValueError('Metadata response exceeds the safety limit.')
    except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        raise _network_failure(exc) from exc
    raw = json.loads(body)
    if int(raw['id']) != cfg['record_id']:
        raise ValueError('Zenodo returned a different record ID. No automatic substitution.')
    metadata = raw.get('metadata', {})
    rights = metadata.get('license', {})
    license_id = rights.get('id', '') if isinstance(rights, dict) else str(rights)
    if not license_id:
        license_id = ','.join(item.get('id', '') for item in metadata.get('rights', [])
                              if isinstance(item, dict))
    if license_id.casefold() != cfg['license_id'].casefold():
        raise ValueError(f'License missing or changed: {license_id!r}. Review official record.')
    raw_files = raw.get('files', [])
    if isinstance(raw_files, dict):
        raw_files = list(raw_files.get('entries', {}).values())
    matches = [f for f in raw_files if f.get('key', f.get('filename')) == cfg['archive_name']]
    if len(matches) != 1:
        raise ValueError('Expected archive is missing or ambiguous in official metadata.')
    item = matches[0]
    checksum = item.get('checksum', '')
    if checksum != 'md5:' + cfg['archive_md5']:
        raise ValueError(f'Published checksum changed: {checksum!r}. STOP.')
    size = int(item.get('size', item.get('filesize', 0)))
    if size != PINNED_SIZE:
        raise ValueError(f'Published size changed: {size}; expected {PINNED_SIZE}. STOP.')
    return raw, {'record_id': cfg['record_id'], 'license_id': license_id,
                 'filename': cfg['archive_name'], 'size_bytes': size,
                 'publisher_md5': cfg['archive_md5'], 'checked_at_utc': now(),
                 'live_metadata_verified': True, 'downloader_version': PATCH_VERSION}


def _validate_response(response, url: str, offset: int, size: int,
                       *, probe_bytes: int | None = None) -> dict:
    info = _response_info(response, url)
    print('[HTTP] ' + json.dumps(info), flush=True)
    status = info['http_status']
    if offset and status != 206:
        raise ValueError(f'HTTP {status}: server did not honor resume at byte {offset}. '
                         'Existing .part was NOT overwritten; choose an explicit fresh download if needed.')
    if status not in (200, 206):
        raise ValueError(f'Unexpected HTTP status: {status}')
    encoding = (info['Content-Encoding'] or '').strip().casefold()
    if encoding not in ('', 'identity'):
        raise ValueError(f'Unexpected Content-Encoding {encoding!r}; byte ranges must refer to original bytes.')
    content_type = (info['Content-Type'] or '').split(';')[0].strip().casefold()
    if (content_type.startswith('text/') or 'html' in content_type
            or 'json' in content_type or 'xml' in content_type):
        raise ValueError(f'Expected ZIP bytes, received Content-Type {content_type!r}; possible error/challenge page.')
    expected_length = size - offset
    if status == 206:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', info['Content-Range'] or '')
        expected_end = min(size, probe_bytes) - 1 if probe_bytes is not None else size - 1
        if not match or tuple(map(int, match.groups())) != (offset, expected_end, size):
            raise ValueError(f"Invalid Content-Range: {info['Content-Range']!r}; "
                             f'expected bytes {offset}-{expected_end}/{size}.')
        expected_length = expected_end - offset + 1
    elif info['Content-Range']:
        raise ValueError('Content-Range is inconsistent with HTTP 200.')
    if info['Content-Length'] is not None:
        try:
            length = int(info['Content-Length'])
        except ValueError as exc:
            raise ValueError('Invalid Content-Length header.') from exc
        if length != expected_length:
            raise ValueError(f'Content-Length {length} does not match expected response bytes {expected_length}.')
    return info


def download_file(url: str, path: Path, size: int, *, socket_timeout: float = 20,
                  attempts: int = 3, _notify=None) -> None:
    """Streaming helper; CLI additionally supervises each attempt in a child.

    Retains the original three-positional-argument interface for existing tests.
    This generic helper accepts tiny test sizes; CLI enforces all source pins.
    """
    _safe_url(url)
    if size <= 0 or attempts < 1:
        raise ValueError('Size and attempts must be positive.')
    part = path.with_name(path.name + '.part')
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise ValueError('Partial file exceeds pinned size. Preserve it separately before a fresh download.')
        if offset == size:
            print('[TRANSFER] Exact byte count already present; checksum verification still required.', flush=True)
            return
        headers = dict(HEADERS)
        if offset:
            headers['Range'] = f'bytes={offset}-'
        print(f'[CONNECT] offset={offset:,} expected_total={size:,} URL={url}', flush=True)
        _emit(_notify, event='stage', stage='archive_connecting', offset=offset, url=url)
        try:
            with urlopen(Request(url, headers=headers), timeout=socket_timeout) as response:
                info = _validate_response(response, url, offset, size)
                _emit(_notify, event='stage', stage='waiting_for_first_body_bytes', response=info)
                count, last_log, log_bytes = offset, time.monotonic(), offset
                # Unbuffered output makes each returned chunk immediately visible as file growth.
                with part.open('ab' if offset else 'wb', buffering=0) as output:
                    while chunk := response.read1(CHUNK_BYTES):
                        if count + len(chunk) > size:
                            raise ValueError('Download exceeds the pinned byte size; refusing extra bytes.')
                        written = output.write(chunk)
                        if written != len(chunk):
                            raise OSError('Short disk write; saved bytes retained but not verified.')
                        count += written
                        if log_bytes == offset:
                            print(f'[FIRST_BYTES] Saved {count - offset:,} new bytes; total={count:,}.', flush=True)
                            _emit(_notify, event='stage', stage='streaming_body')
                            # Ensure only one first-byte log even for a one-byte initial packet.
                            log_bytes = count
                        current = time.monotonic()
                        if _notify is None and current - last_log >= 5:
                            print(f'[PROGRESS] {count:,}/{size:,} bytes; '
                                  f'{(count - log_bytes)/(current-last_log)/1024:.1f} KiB/s', flush=True)
                            last_log, log_bytes = current, count
            if part.stat().st_size != size:
                raise NetworkFailure('Response ended before the full archive arrived.', retryable=True)
            print(f'[TRANSFER] Saved exactly {size:,} bytes; not yet checksum-verified.', flush=True)
            return
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            failure = _network_failure(exc)
        except NetworkFailure as exc:
            failure = exc
        if not failure.retryable or attempt >= attempts or failure.retry_after > 120:
            raise failure
        delay = max(2 ** (attempt - 1), failure.retry_after)
        print(f'[RETRY] {failure}; retrying from saved bytes after {delay:g}s.', flush=True)
        time.sleep(delay)


def probe_archive(url: str, size: int, *, socket_timeout: float = 20, _notify=None) -> dict:
    """Read at most 64 KiB of the *same* archive; no .part file and no full verification."""
    _safe_url(url)
    limit = min(CHUNK_BYTES, size)
    headers = {**HEADERS, 'Range': f'bytes=0-{limit - 1}'}
    print(f'[PROBE] Reading at most {limit:,} bytes from {url}', flush=True)
    _emit(_notify, event='stage', stage='probe_connecting', url=url)
    try:
        with urlopen(Request(url, headers=headers), timeout=socket_timeout) as response:
            info = _validate_response(response, url, 0, size, probe_bytes=limit)
            _emit(_notify, event='stage', stage='probe_body', response=info)
            body = bytearray()
            while len(body) < limit:
                chunk = response.read1(limit - len(body))
                if not chunk:
                    raise NetworkFailure('Probe response ended early.', retryable=True)
                body.extend(chunk)
                _emit(_notify, event='bytes', bytes=len(body))
    except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        raise _network_failure(exc) from exc
    if not body.startswith(b'PK\x03\x04'):
        raise ValueError('Probe did not begin with a ZIP local-file signature; do not bypass checksum checks.')
    return {'status': 'PROBE_PASS_NOT_FULL_VERIFICATION', 'bytes_read': len(body),
            'prefix_hex': body[:8].hex(), 'response': info, 'archive_verified': False}


@contextmanager
def download_lock(path: Path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f'Download lock exists: {path}. Stop the earlier downloader first. '
                              'Remove a stale lock only after confirming no downloader is running.') from exc
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump({'pid': os.getpid(), 'started_at_utc': now()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_candidate(candidate: Path, cfg: dict) -> dict:
    if candidate.stat().st_size != PINNED_SIZE:
        raise ValueError(f'File size is {candidate.stat().st_size}; expected exactly {PINNED_SIZE}. '
                         'No checksum receipt or final ZIP will be created.')
    print('[VERIFY] Exact size OK. Computing publisher MD5 and local SHA256...', flush=True)
    hashes = archive_digests(candidate)
    if hashes['md5'] != cfg['archive_md5']:
        raise ValueError('Archive MD5 does not match pinned v4. Preserve the bad .part separately; '
                         'do not extract it. No final ZIP or VERIFIED receipt was created.')
    return hashes


def _run_network(target, args, options, folder: Path, *, part: Path | None = None,
                 attempts: int = 1):
    for number in range(1, attempts + 1):
        print(f'[NETWORK] Attempt {number}/{attempts}; socket timeout={options.socket_timeout:g}s; '
              f'no-progress deadline={options.stall_timeout:g}s.', flush=True)
        try:
            return supervise(target, args, {'socket_timeout': options.socket_timeout,
                                           **({'attempts': 1} if target is download_file else {})},
                             part=part, stall_timeout=options.stall_timeout,
                             report=folder / 'download_network.json',
                             context={'attempt': number, 'attempt_limit': attempts,
                                      'record_id': PINS['record_id'], 'filename': PINS['archive_name'],
                                      'expected_size_bytes': PINNED_SIZE,
                                      'expected_md5': PINS['archive_md5']})
        except NetworkFailure as exc:
            if not exc.retryable or number >= attempts or exc.retry_after > 120:
                raise
            delay = max(2 ** (number - 1), exc.retry_after)
            print(f'[RETRY] {exc}; next attempt after {delay:g}s. Saved bytes retained.', flush=True)
            time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--metadata-only', action='store_true')
    modes.add_argument('--probe', action='store_true')
    modes.add_argument('--verify-local', type=Path, help='Verify an official browser/curl download; no HTTP calls.')
    parser.add_argument('--accept-license', choices=['CC-BY-4.0'])
    parser.add_argument('--socket-timeout', type=float, default=20)
    parser.add_argument('--stall-timeout', type=float, default=45)
    parser.add_argument('--attempts', type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if (not math.isfinite(args.socket_timeout) or not math.isfinite(args.stall_timeout)
            or args.socket_timeout <= 0 or args.stall_timeout <= 0):
        raise ValueError('Timeout values must be positive finite numbers.')
    cfg = config()
    validate_pins(cfg)
    if not args.metadata_only and args.accept_license != cfg['license_id']:
        raise ValueError('Review the source license, then supply --accept-license CC-BY-4.0.')
    archive = raw_archive().resolve()
    folder = archive.parent
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'download_receipt.json'
    print(f'[DOWNLOADER {PATCH_VERSION}] record=8429208 v4 file=UAV-PDD2023.zip '
          f'exact_bytes={PINNED_SIZE} md5={PINS["archive_md5"]}', flush=True)
    print(f'[OUTPUT] {archive}', flush=True)
    mutates_receipt = not (args.metadata_only or args.probe)
    with download_lock(folder / '.download.lock'):
        if mutates_receipt:
            write_json(receipt, {'status': 'IN_PROGRESS', 'started_at_utc': now(),
                                 'downloader_version': PATCH_VERSION})
        try:
            if args.verify_local:
                if not args.verify_local.is_file():
                    raise FileNotFoundError(args.verify_local)
                source = {'record_id': cfg['record_id'], 'license_id': cfg['license_id'],
                          'filename': cfg['archive_name'], 'publisher_md5': cfg['archive_md5'],
                          'size_bytes': PINNED_SIZE, 'live_metadata_verified': False,
                          'verification_basis': 'Pinned record, exact byte size and publisher MD5; no HTTP calls.',
                          'checked_at_utc': now(), 'downloader_version': PATCH_VERSION}
            else:
                raw, source = _run_network(fetch_metadata, (cfg,), args, folder)
                write_json(folder / 'zenodo_metadata.json', raw)
            write_json(folder / 'source_check.json', source)
            print(json.dumps(source, indent=2), flush=True)
            if args.metadata_only:
                print('Metadata checked. No dataset bytes downloaded.', flush=True)
                return 0
            if args.probe:
                result = _run_network(probe_archive, (cfg['archive_url'], PINNED_SIZE), args, folder)
                write_json(folder / 'download_probe.json', result)
                print(json.dumps(result, indent=2), flush=True)
                return 0
            part = archive.with_name(archive.name + '.part')
            already_here = sum(p.stat().st_size for p in (archive, part) if p.exists())
            required = max(0, 3 * PINNED_SIZE - already_here) + 2 * GIB
            available = shutil.disk_usage(folder).free
            if available < required:
                raise OSError(f'Free space {available / GIB:.2f} GiB; conservative required {required / GIB:.2f} GiB.')
            if args.verify_local:
                candidate = args.verify_local.resolve()
            elif archive.exists():
                candidate = archive
            else:
                _run_network(download_file, (cfg['archive_url'], archive, PINNED_SIZE), args,
                             folder, part=part, attempts=args.attempts)
                candidate = part
            hashes = verify_candidate(candidate, cfg)
            if candidate != archive:
                if archive.exists():
                    if verify_candidate(archive, cfg) != hashes:
                        raise FileExistsError('A different archive already exists; no overwrite performed.')
                elif candidate == part:
                    candidate.replace(archive)
                else:
                    # Import a verified browser/curl file through .part; never expose a partial final ZIP.
                    if part.exists() and part.stat().st_size:
                        raise FileExistsError('Nonempty .part exists. Preserve it separately before importing a local file.')
                    with candidate.open('rb') as src, part.open('wb') as dst:
                        shutil.copyfileobj(src, dst, length=CHUNK_BYTES)
                    if verify_candidate(part, cfg) != hashes:
                        raise ValueError('Local file copy failed integrity verification.')
                    part.replace(archive)
            write_json(receipt, {
                'status': 'VERIFIED', 'source_id': cfg['source_id'], 'record_url': cfg['record_url'],
                'license_id': cfg['license_id'], 'license_explicitly_accepted': True,
                'live_metadata_verified': source['live_metadata_verified'],
                'archive_bytes': archive.stat().st_size, **hashes, 'verified_at_utc': now(),
                'downloader_version': PATCH_VERSION})
            (folder / 'ATTRIBUTION.txt').write_text(provenance_text(cfg), encoding='utf-8')
            print(f'VERIFIED: {archive}\nNo extraction, audit preparation, or training was performed.', flush=True)
            return 0
        except BaseException as exc:
            if mutates_receipt:
                write_json(receipt, {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                                     'error_type': type(exc).__name__, 'error': str(exc),
                                     'downloader_version': PATCH_VERSION, 'updated_at_utc': now()})
            raise


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('DOWNLOAD INTERRUPTED: worker stopped; saved .part bytes retained.', file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f'DOWNLOAD BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
