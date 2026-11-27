"""Object-store abstraction for landing raw files: Google Cloud Storage plus a local-directory stand-in.

Two implementations share one small interface so that landing logic is tested without a cloud:

* :class:`GcsObjectStore` talks to Cloud Storage with Application Default Credentials. Credentials
  are never read from configuration files and never logged.
* :class:`DirectoryObjectStore` keeps objects as files under ``<root>/<bucket>/``. It enforces the
  same ``if_generation_match`` preconditions and reports the same checksums, and is used for tests
  and for ``--local-bucket-dir`` rehearsals. It is not a cloud check.

Transfer integrity (the CRC32C the client computed vs the CRC32C the store reports after the
upload) is deliberately separate from the manifest's content MD5.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

CHUNK = 1024 * 1024


class ObjectStoreError(Exception):
    """Generic object-store failure."""


class CloudConfigError(ObjectStoreError):
    """Bucket or credentials are missing/invalid/unauthorised (nothing was changed)."""


class PreconditionFailed(ObjectStoreError):
    """The object changed between the check and the write (generation precondition failed)."""


class TransferIntegrityError(ObjectStoreError):
    """The store reports different bytes than the client sent."""


@dataclass(frozen=True)
class RemoteObject:
    name: str
    size: int
    md5_base64: str | None
    crc32c_base64: str | None
    generation: int | None


@dataclass(frozen=True)
class LocalChecksums:
    size: int
    md5_base64: str
    crc32c_base64: str | None


class ObjectStore(Protocol):
    def describe(self) -> str: ...

    def stat(self, name: str) -> RemoteObject | None: ...

    def upload_file(
        self, local_path: Path, name: str, *, content_type: str, if_generation_match: int, chunk_size: int
    ) -> RemoteObject: ...


def local_checksums(path: Path) -> LocalChecksums:
    """Stream ``path`` once for size, MD5 (base64, as GCS reports it) and CRC32C (None if unavailable)."""
    md5 = hashlib.md5()
    crc = None
    try:
        import google_crc32c

        crc = google_crc32c.Checksum()
    except ImportError:  # CRC32C is optional; MD5 is then the only comparison
        crc = None
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            md5.update(chunk)
            size += len(chunk)
            if crc is not None:
                crc.update(chunk)
    return LocalChecksums(
        size=size,
        md5_base64=base64.b64encode(md5.digest()).decode("ascii"),
        crc32c_base64=base64.b64encode(crc.digest()).decode("ascii") if crc is not None else None,
    )


def same_content(local: LocalChecksums, remote: RemoteObject) -> bool:
    """True when size and every checksum both sides report agree (CRC32C preferred, MD5 otherwise)."""
    if local.size != remote.size:
        return False
    compared = False
    if local.crc32c_base64 and remote.crc32c_base64:
        compared = True
        if local.crc32c_base64 != remote.crc32c_base64:
            return False
    if remote.md5_base64:
        compared = True
        if local.md5_base64 != remote.md5_base64:
            return False
    return compared


# --------------------------------------------------------------------------- GCS


class GcsObjectStore:
    """Cloud Storage via ``google-cloud-storage`` and Application Default Credentials."""

    def __init__(self, bucket: str, *, project: str | None = None, client: Any | None = None):
        self.bucket_name = bucket
        try:
            from google.api_core import exceptions as gexc
            from google.auth import exceptions as auth_exc
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise CloudConfigError(
                "google-cloud-storage is not installed; run: pip install -r ingestion/requirements-gcs.txt"
            ) from exc
        self._gexc = gexc
        if client is None:
            try:
                client = storage.Client(project=project)
            except auth_exc.DefaultCredentialsError as exc:
                raise CloudConfigError(
                    "no Google Application Default Credentials found. Run "
                    "`gcloud auth application-default login`, or set GOOGLE_APPLICATION_CREDENTIALS to a "
                    "service-account key file stored outside the repository."
                ) from exc
        self.client = client
        self.bucket = client.bucket(bucket)

    def describe(self) -> str:
        return f"gs://{self.bucket_name}"

    def _translate(self, exc: Exception) -> ObjectStoreError:
        gexc = self._gexc
        if isinstance(exc, gexc.PreconditionFailed):
            return PreconditionFailed(str(exc))
        if isinstance(exc, (gexc.Forbidden, gexc.Unauthorized)):
            return CloudConfigError(f"access denied for bucket {self.bucket_name}: {exc}")
        if isinstance(exc, gexc.NotFound):
            return CloudConfigError(
                f"bucket {self.bucket_name} was not found (it is created by the infra owner): {exc}"
            )
        return ObjectStoreError(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _remote(blob: Any) -> RemoteObject:
        return RemoteObject(
            name=blob.name,
            size=int(blob.size),
            md5_base64=blob.md5_hash,
            crc32c_base64=blob.crc32c,
            generation=int(blob.generation) if blob.generation is not None else None,
        )

    def stat(self, name: str) -> RemoteObject | None:
        try:
            blob = self.bucket.get_blob(name)
        except self._gexc.GoogleAPICallError as exc:
            # `get_blob` returns None for a missing object, so NotFound here means a missing bucket.
            raise self._translate(exc) from exc
        return None if blob is None else self._remote(blob)

    def upload_file(
        self, local_path: Path, name: str, *, content_type: str, if_generation_match: int, chunk_size: int
    ) -> RemoteObject:
        blob = self.bucket.blob(name, chunk_size=chunk_size)
        try:
            # No Content-Encoding is set, so GCS stores and serves the exact bytes (no transcoding).
            blob.upload_from_filename(
                str(local_path),
                content_type=content_type,
                if_generation_match=if_generation_match,
                checksum="crc32c",
            )
        except self._gexc.GoogleAPICallError as exc:
            raise self._translate(exc) from exc
        except Exception as exc:  # google-resumable-media DataCorruption and friends
            if type(exc).__name__ == "DataCorruption":
                raise TransferIntegrityError(f"upload of {name} failed client-side CRC32C verification: {exc}") from exc
            raise
        return self._remote(blob)


# --------------------------------------------------------------------------- local directory


class DirectoryObjectStore:
    """File-system object store with GCS-like preconditions (tests and local rehearsal only)."""

    def __init__(self, root: Path, bucket: str):
        self.root = Path(root)
        self.bucket_name = bucket
        self.base = self.root / bucket
        self.base.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return f"local-directory://{self.base}"

    def _path(self, name: str) -> Path:
        path = (self.base / name).resolve()
        if self.base.resolve() not in path.parents:
            raise ObjectStoreError(f"object name escapes the bucket directory: {name!r}")
        return path

    def stat(self, name: str) -> RemoteObject | None:
        path = self._path(name)
        if not path.is_file():
            return None
        checks = local_checksums(path)
        return RemoteObject(name, checks.size, checks.md5_base64, checks.crc32c_base64, path.stat().st_mtime_ns)

    def upload_file(
        self, local_path: Path, name: str, *, content_type: str, if_generation_match: int, chunk_size: int
    ) -> RemoteObject:
        target = self._path(name)
        existing = self.stat(name)
        current_generation = 0 if existing is None else existing.generation
        if current_generation != if_generation_match:
            raise PreconditionFailed(f"{name}: generation {current_generation} != required {if_generation_match}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".upload.")
        os.close(fd)
        try:
            shutil.copyfile(local_path, tmp)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        stored = self.stat(name)
        assert stored is not None
        return stored
