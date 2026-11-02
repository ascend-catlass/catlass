---
nav_order: 20
---

# Building CATLASS Python

Unless otherwise noted, all commands are run in `/path/to/catlass/python/tla_dsl`, where `/path/to/catlass` is the root directory of the CATLASS repository you cloned.

## Development Mode Build (Development State)

```bash
cd /path/to/catlass/python/tla_dsl
./build.sh
# Force a clean and rebuild
./build.sh --clean
```

`build.sh` performs a Development mode build by default. The main steps are:

1. Check `ASCEND_HOME_PATH` and the AscendNPU-IR build artifacts.
2. Build the compiler and Python extension in `csrc/mlir/build`.
3. Install `ascend-catlass-dsl` in editable mode without reinstalling dependencies.

- `--clean` deletes `build/`, `csrc/mlir/build/`, `dist/`, egg-info, pytest caches, and the binary library (`_tla_type_bridge_native*.so`) in the CATLASS Python subproject, then performs a Development mode build.

After a successful build, check the key artifacts:

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
python -c "from catlass import tla"
```

If all of the commands above produce no output and exit with code `0`, the checks pass and the artifacts are ready.

## Release Mode

```bash
./build.sh --release
ls dist/*.whl
```

Release mode generates a wheel in `dist/`.
