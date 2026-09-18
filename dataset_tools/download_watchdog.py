"""Spawn-safe, bounded no-progress supervision for an archive network operation.

Only the child opens/writes the .part file. The parent remains responsive during
DNS, proxy, TLS, headers and body reads, and stops/joins its own child on abort.
No third-party dependencies. This is not a training worker.
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import time
from pathlib import Path
from typing import Any, Callable

from .common import now, write_json


class NetworkFailure(OSError):
    def __init__(self, message: str, *, retryable: bool = False, retry_after: float = 0):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


def _worker(connection: Any, target: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    # The parent owns Ctrl+C and always terminates/joins this particular child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = target(*args, **kwargs, _notify=connection.send)
        connection.send({'event': 'done', 'result': result})
    except Exception as exc:
        connection.send({'event': 'error', 'error_type': type(exc).__name__,
                         'message': str(exc), 'retryable': bool(getattr(exc, 'retryable', False)),
                         'retry_after': float(getattr(exc, 'retry_after', 0))})
    finally:
        connection.close()


def _stop(process: Any) -> None:
    if process.is_alive():
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    if process.is_alive():
        raise RuntimeError('Network child could not be stopped. Do not launch another writer.')


def supervise(target: Callable[..., Any], args: tuple = (), kwargs: dict | None = None, *,
              part: Path | None = None, stall_timeout: float = 45,
              heartbeat: float = 5, report: Path | None = None,
              context: dict | None = None) -> Any:
    """Abort if no *saved body bytes* arrive within stall_timeout seconds.

    Header events do NOT reset the deadline. For probes (part=None), byte events
    report application-read bytes instead. No global time limit is imposed on a
    download which is actually making progress. A slow genuine transfer is valid.
    """
    if stall_timeout <= 0 or heartbeat <= 0:
        raise ValueError('Watchdog intervals must be positive.')
    ctx = mp.get_context('spawn')
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(writer, target, args, kwargs or {}),
                          name='pothole-dataset-download')
    saved = part.stat().st_size if part is not None and part.exists() else 0
    start = last_progress = last_log = time.monotonic()
    log_bytes = saved
    state = {**(context or {}), 'status': 'RUNNING', 'stage': 'starting_network_worker',
             'saved_bytes': saved, 'stall_timeout_seconds': stall_timeout}

    def persist() -> None:
        state['updated_at_utc'] = now()
        state['elapsed_seconds'] = round(time.monotonic() - start, 2)
        if report is not None:
            write_json(report, state)

    process.start()
    writer.close()
    try:
        persist()
        pipe_open = True
        while True:
            message = None
            if pipe_open and reader.poll(0.2):
                try:
                    message = reader.recv()
                except EOFError:
                    pipe_open = False
            elif not pipe_open:
                time.sleep(0.05)
            if message:
                kind = message.get('event')
                if kind == 'done':
                    state['status'] = 'NETWORK_OPERATION_COMPLETE'
                    state['stage'] = 'complete_not_checksum_verified'
                    if part is not None and part.exists():
                        state['saved_bytes'] = part.stat().st_size
                    persist()
                    return message.get('result')
                if kind == 'error':
                    raise NetworkFailure(f"{message['error_type']}: {message['message']}",
                                         retryable=message['retryable'],
                                         retry_after=message['retry_after'])
                if kind == 'stage':
                    state.update({k: v for k, v in message.items() if k != 'event'})
                if kind == 'bytes' and part is None:
                    new_saved = int(message['bytes'])
                    if new_saved > saved:
                        saved = new_saved
                        last_progress = time.monotonic()
            if part is not None and part.exists():
                new_saved = part.stat().st_size
                if new_saved > saved:
                    saved = new_saved
                    last_progress = time.monotonic()
                elif new_saved < saved:
                    raise NetworkFailure('Partial file shrank unexpectedly; possible concurrent writer.')
            current = time.monotonic()
            state['saved_bytes'] = saved
            state['seconds_without_progress'] = round(current - last_progress, 2)
            if current - last_log >= heartbeat:
                speed = (saved - log_bytes) / max(current - last_log, 1e-6) / 1024
                print(f"[WATCH] stage={state['stage']} saved={saved:,} bytes "
                      f"rate={speed:.1f} KiB/s idle={current - last_progress:.1f}s", flush=True)
                last_log, log_bytes = current, saved
                persist()
            if current - last_progress >= stall_timeout:
                raise NetworkFailure(f"No saved-byte progress for {stall_timeout:g}s "
                                     f"during {state['stage']}. Partial file retained.", retryable=True)
            if not process.is_alive() and (not pipe_open or not reader.poll()):
                raise NetworkFailure(f'Network worker exited without a completion message '
                                     f'(exit code {process.exitcode}).')
    except BaseException as exc:
        state.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                     error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
    finally:
        try:
            _stop(process)
        finally:
            reader.close()
            process.close()
