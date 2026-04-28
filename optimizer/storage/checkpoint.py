from optimizer.errors import FingerprintMismatchError


def check_resume(storage, fingerprints, force=False):
    old = storage.load_meta() if hasattr(storage, "load_meta") else None
    if old and old != fingerprints and not force:
        raise FingerprintMismatchError(
            "resume fingerprint mismatch; use force_resume_on_fingerprint_mismatch=True to override"
        )
    storage.init_run(fingerprints)
    return old
