from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import os


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def queue_lock(path, inherited_fd=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.fdopen(int(inherited_fd), 'a+') if inherited_fd is not None else path.open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle
    finally:
        # Closing rather than LOCK_UN preserves the inherited lock while an
        # active training/evaluation subprocess still owns the descriptor.
        handle.close()
