from optimizer.errors import FingerprintMismatchError


def check_resume(storage, fingerprints, force=False, required_non_null=()):
    old = storage.load_meta() if hasattr(storage, "load_meta") else None
    missing = [name for name in required_non_null if not fingerprints.get(name)]
    if old and missing and not force:
        raise FingerprintMismatchError(
            "resume requires complete resume identity; missing: "
            + ", ".join(sorted(missing))
        )
    if old and old != fingerprints and not force:
        raise FingerprintMismatchError(
            "resume fingerprint mismatch; use force_resume_on_fingerprint_mismatch=True to override"
        )
    storage.init_run(fingerprints)
    return old
