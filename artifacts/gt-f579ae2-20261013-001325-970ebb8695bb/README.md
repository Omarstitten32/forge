# Ground Truth Snapshot

- **SHA**:        f579ae2
- **Dirty hash**: 970ebb8695bb
- **Timestamp**:  20261013-001325
- **Artifact**:   /root/forge/artifacts/gt-f579ae2-20261013-001325-970ebb8695bb

## Structure

- `meta/`     — git, env, toolchain, build, binary facts
- `cli/`      — every CLI invocation (raw stdout/stderr/exit/determinism/time)
- `codegen/`  — actual output of `-o` for each backend
- `perf/`     — hyperfine 200-run measurements
- `memory/`   — valgrind full log
- `manifest.sha256` — integrity of all files

## Reproduce

    bash /root/forge/scripts/stage-1-ground-truth.sh

## Note

This snapshot contains **raw facts only**. No PASS/FAIL labels.
Interpretation happens in review, not here.
