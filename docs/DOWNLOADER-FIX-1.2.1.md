# Pothole Detection Drone - dataset downloader fix 1.2.1

## Scope and fixed source

Repair only the Phase 1.2 downloader. No pip installation, environment change,
training, extraction, dataset selection change, or automatic advance of phases.

- Record: 8429208, version v4
- Filename: UAV-PDD2023.zip
- Exact bytes: 2122615013 (from the user's successfully verified live metadata;
  now also an independent local pin)
- Publisher MD5: 0745d017ce2dce23bd73944e9b0acec7
- Archive URL: https://zenodo.org/records/8429208/files/UAV-PDD2023.zip?download=1
- Metadata URL: https://zenodo.org/api/records/8429208
- Required explicit license acceptance: CC-BY-4.0

## Diagnosis

The previous downloader opens the .part output only after HTTP response headers
arrive. It then calls response.read(8 * 1024 * 1024) before writing any bytes,
and does not report progress until 128 MiB has been written. In CPython's HTTP
implementation, a buffered read of that size can wait for the requested amount
(or EOF), keeping bytes in memory while the file stays empty. A slow stream can
keep socket operations alive without filling that first buffer promptly.

Therefore a newly created zero-byte .part is consistent with blocked/slow body
reading after the headers. It is not proof of zero network traffic, a particular
firewall problem, or a GPU/environment problem. An old .part from an earlier run
also cannot establish what the current process is doing. CPU idle is compatible
with waiting for network I/O.

The prior code also trusted any positive metadata size instead of independently
pinning 2122615013, validated only the start of Content-Range, and did not check
redirect targets before following them. Its existing tests did not cover a real
slow body or a hard no-progress deadline. The ordinary timeout=60 parameter was
not a whole-transfer wall-clock deadline.

## Changes

- Use HTTPResponse.read1(64 KiB) and unbuffered output; save returned bytes without
  waiting to fill a large Python read buffer.
- Print request and selected response headers, first saved bytes and five-second
  parent-process progress heartbeats, even while the network child is blocked.
- Default blocking-operation timeout: 20 seconds. An independent spawn-safe parent
  stops and joins its child after 45 seconds without saved-byte progress. This
  covers blocked DNS, proxy, TLS, headers or body handling. Header messages do not
  reset the deadline. For metadata/probes, actual body bytes read reset it.
- Up to three archive attempts by default; saved bytes are retained. Permanent
  errors are not repeatedly retried. Retry-After is respected; a requested delay
  over 120 seconds is surfaced as failure rather than silently sleeping longer.
- Slow transfers with continuing saved-byte progress are NOT force-aborted for
  taking a long total time. Timeout values are configurable.
- Resume only on the exact requested 206 range, matching end and total byte size.
  A 200 response to a nonzero resume is rejected without truncating the partial.
- Reject inconsistent Content-Length, encoded bodies and obvious error-page MIME
  types. Enforce HTTPS, the same host and resource, and normal certificate checks.
- Verify exact size and pinned MD5 before renaming .part to .zip. Record SHA256.
- Guard against a second patched downloader; clean up the lock on ordinary errors
  and Ctrl+C. Never auto-delete stale locks or nonempty partial files.
- Include a small --probe which reads at most 65536 bytes of the same archive and
  does not modify the download .part or VERIFIED receipt. It checks response
  headers and a ZIP prefix but is not a full archive-integrity check.
- Preserve source_check.json and download_receipt.json compatibility with audit.py.
- Record network diagnostics, failures and interrupted receipts rather than leave
  an apparently active/successful receipt after a handled failure.

## Files

The installer replaces dataset_tools/download.py and updates two lines in the
existing metadata unit test so its fixture uses the newly independent size pin.
It adds dataset_tools/download_watchdog.py, tests/test_download_patch.py and these
diagnostic documents. The old 28 tests remain and 42 additional cases are added.
The installer checks old-file hashes, backs up replaced files, and refuses to
silently overwrite custom edits. An already-identical installation is a no-op.
Configuration and existing .part files are not changed by installation.

## Windows PowerShell commands

First stop the original hanging command with Ctrl+C. Do not run two downloaders.
Do not terminate every Python process: unrelated development tools may be running.
To inspect only commands mentioning this downloader:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'dataset_tools\.download' } | Select-Object ProcessId, ParentProcessId, CommandLine
```

Download pothole-drone-download-fix-1.2.1.zip to Downloads. Use your actual project
folder when different from the earlier setup path. Execute these in one PowerShell:

```powershell
Set-Location "$HOME\source\pothole-drone"
$ErrorActionPreference = "Stop"
$Project = (Get-Location).Path
$Patch = Join-Path $env:TEMP ("pothole-download-fix-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -LiteralPath "$HOME\Downloads\pothole-drone-download-fix-1.2.1.zip" -DestinationPath $Patch
.\.venv\Scripts\python.exe "$Patch\apply_download_fix.py" --project "$Project"
if ($LASTEXITCODE -ne 0) { throw "Patch installation failed. Do not overwrite customized files blindly." }
.\.venv\Scripts\python.exe -m pytest -q .\tests\test_phase12_dataset.py .\tests\test_download_patch.py
if ($LASTEXITCODE -ne 0) { throw "Downloader regression tests failed." }
```

Expected test result: 70 passed (elapsed time depends on the laptop).
No pip install is needed. The patch does not affect the environment-validation result.

### Probe first

```powershell
.\.venv\Scripts\python.exe -u -m dataset_tools.download --probe --accept-license CC-BY-4.0
$ProbeExit = $LASTEXITCODE
Get-Content .\data\raw\uav_pdd2023_v4\download_network.json
if ($ProbeExit -ne 0) { throw "Archive probe failed; inspect network diagnostics before a full transfer." }
Get-Content .\data\raw\uav_pdd2023_v4\download_probe.json
```

Expect PROBE_PASS_NOT_FULL_VERIFICATION and bytes_read=65536. The response may be
206 with Content-Range bytes 0-65535/2122615013; a 200 response is also acceptable
when the server ignores Range, because this Python probe closes after reading
at most 65536 body bytes. A passing probe does not prove a complete transfer will
succeed. A previous probe JSON can be old: only use it after current exit code 0;
the network report records the current operation and UTC update time.

### Download the pinned archive

Keep the original zero-byte .part; it is safe for this patch to reuse it.
Nonzero .part bytes are preserved and a validated resume is attempted.

```powershell
.\.venv\Scripts\python.exe -u -m dataset_tools.download --accept-license CC-BY-4.0
$DownloadExit = $LASTEXITCODE
if ($DownloadExit -ne 0) {
    Get-Content .\data\raw\uav_pdd2023_v4\download_network.json
    throw "Download failed safely; the archive has not been approved."
}
Get-Content .\data\raw\uav_pdd2023_v4\download_receipt.json
(Get-Item .\data\raw\uav_pdd2023_v4\UAV-PDD2023.zip).Length
```

The receipt must have status VERIFIED, archive_bytes 2122615013 and the pinned MD5.
An independent MD5 recheck is optional (it reads the entire archive again):

```powershell
(Get-FileHash .\data\raw\uav_pdd2023_v4\UAV-PDD2023.zip -Algorithm MD5).Hash.ToLowerInvariant()
```

Expected hash: 0745d017ce2dce23bd73944e9b0acec7.

Optional explicit settings (defaults shown):

```powershell
.\.venv\Scripts\python.exe -u -m dataset_tools.download --accept-license CC-BY-4.0 --socket-timeout 20 --stall-timeout 45 --attempts 3
```

The separate child process is a network worker, not model inference or training.
The 45-second deadline is about byte progress, not total download duration. If a
particular connection genuinely takes longer before delivering any byte, change
the timeout explicitly only after inspecting the stage report.

## Diagnostic interpretation

- archive_connecting: request has not returned complete headers. DNS, proxy, TLS,
  connection or server-header latency is possible; this report cannot alone tell
  which one. Capture the exception without assuming the cause.
- waiting_for_first_body_bytes: acceptable headers arrived but no body bytes were
  saved. Server/proxy body delay is possible; the prior 8 MiB visibility problem
  has been removed.
- streaming_body with increasing saved_bytes: real transfer progress; not a hang.
- HTTP 403: file request was refused. Do not keep hammering or disable TLS.
- HTTP 429: honor Retry-After; do not launch many concurrent copies.
- Invalid Content-Length / Content-Range / ZIP prefix: source response is not what
  the pinned download requires. Do not bypass the check.
- MD5 mismatch: not the approved archive, even if the byte count is correct.

Do not share proxy passwords, cookies or an unrestricted dump of environment
variables. The built-in report logs selected ordinary response headers only.

## Optional independent native-curl transfer, same exact endpoint

Use this only after stopping the Python download. It changes the HTTP client,
not the dataset. It is not a mirror and does not change the pinned record.
This command uses curl.exe explicitly, disables curlrc configuration, keeps TLS
verification, does not follow redirects, and has connection and low-speed limits.
It saves to a separate temporary archive and must pass the Python verifier before
being accepted. If curl.exe is unavailable, use the official-record browser option
below; no Python dependencies need to change.

```powershell
$Curl = (Get-Command curl.exe -ErrorAction Stop).Source
& $Curl --version
$Url = "https://zenodo.org/records/8429208/files/UAV-PDD2023.zip?download=1"
$Native = Join-Path $Project "data\raw\uav_pdd2023_v4\UAV-PDD2023.native.part"
$CurlArgs = @(
    "--disable", "--proto", "=https", "--http1.1",
    "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "30",
    "--fail", "--show-error", "--progress-bar", "--continue-at", "-",
    "--header", "Accept-Encoding: identity", "--max-filesize", "2122615013",
    "--output", $Native, $Url
)
& $Curl @CurlArgs
if ($LASTEXITCODE -ne 0) { throw "Native curl transfer failed; do not verify as successful." }
```

Before importing, an original nonempty UAV-PDD2023.zip.part must be preserved under
another name, because the importer will not overwrite it. Only after BOTH download
processes have stopped, the following moves (does not delete) that prior partial:

```powershell
$OldPart = Join-Path $Project "data\raw\uav_pdd2023_v4\UAV-PDD2023.zip.part"
if ((Test-Path -LiteralPath $OldPart) -and (Get-Item -LiteralPath $OldPart).Length -gt 0) {
    Move-Item -LiteralPath $OldPart -Destination ($OldPart + ".saved-" + [guid]::NewGuid().ToString("N"))
}
.\.venv\Scripts\python.exe -u -m dataset_tools.download --verify-local "$Native" --accept-license CC-BY-4.0
if ($LASTEXITCODE -ne 0) { throw "Local archive failed the pinned verifier." }
```

A native-client success alongside Python failure suggests a client/proxy/TLS path
difference, not proof of any specific root cause. A downloaded filename or a curl
exit code alone does not establish archive identity; exact size and MD5 do.
The native temporary copy is retained; account for its extra disk space.

## Official browser option

Open the same record and use its UAV-PDD2023.zip download:

```powershell
Start-Process "https://zenodo.org/records/8429208"
```

After it finishes, preserve any nonempty original .part as above, then:

```powershell
.\.venv\Scripts\python.exe -u -m dataset_tools.download --verify-local "$HOME\Downloads\UAV-PDD2023.zip" --accept-license CC-BY-4.0
if ($LASTEXITCODE -ne 0) { throw "Browser download did not pass exact-size/MD5 verification." }
```

Offline import records live_metadata_verified=false honestly; it still requires
exact pinned size and MD5. Never substitute another Zenodo record or re-export.

## Stale lock and stop conditions

Ctrl+C is handled by the patched parent, which stops its worker and removes its
lock. An abrupt process kill, power failure or crash may leave .download.lock.
Remove only that lock after confirming no original OR patched download processes
remain; inspect its recorded PID first. Do not delete nonempty .part bytes merely
to clear the lock. Do not launch a native-client writer concurrently with Python.

This patch's success gate is a real VERIFIED download_receipt.json with correct
size and MD5. No dataset audit or model training is run automatically. Return the
current network/probe result on failure, or the verified receipt on success.

## Evidence and limits

The authoring runtime passed 70 tests, including a real local socket HTTPResponse
that delivers 1024 bytes and withholds the tail: the patched .part grows before
the tail is released. Separate tests use Windows-compatible spawn supervision to
terminate a stalled child and return success from a responsive child. Fixtures
are code diagnostics, not a training dataset. These tests were run on Linux, not
on the user's Windows laptop; Windows network/driver state is not accessible here.
A live bounded probe in the authoring runtime failed DNS resolution. No full
Zenodo archive was downloaded or claimed to be verified in that runtime.

Primary references checked September 18, 2026:
- Official pinned record and published MD5: https://zenodo.org/records/8429208
- CPython 3.13.15 HTTPResponse read/read1 implementation:
  https://raw.githubusercontent.com/python/cpython/v3.13.15/Lib/http/client.py
- urllib blocking-operation timeout and redirect behavior:
  https://docs.python.org/3.13/library/urllib.request.html
- Buffered I/O read/read1 behavior: https://docs.python.org/3.13/library/io.html
- Native curl options: https://curl.se/docs/manpage.html
