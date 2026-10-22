## What and why

## Checklist

- [ ] `pytest` passes (and `pytest -m slow` if the numerics are touched)
- [ ] Numerics changed? The reference runs (`test/test_reference.py`) still reproduce, or were re-recorded with `tools/reference_run.py` in a separate commit that explains why
- [ ] Numerics or model changed? The corresponding chapter in `docs/` is updated
- [ ] New parameter? It has a docstring below its field (shows up in `droplet-relax show-config` and the docs)
- [ ] Run-file layout changed? `io/schema.py` updated and `FORMAT_VERSION` bumped
