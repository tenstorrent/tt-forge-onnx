# Build Docker Images

## Overview

`.github/build-docker-images.sh` is the primary script responsible for building and publishing all Docker images used by the tt-forge-onnx CI pipeline. It is invoked automatically by the [`build-image.yml`](../../.github/workflows/build-image.yml) GitHub Actions workflow, and can also be run locally by developers.

The script builds four images in a fixed dependency order, skipping any image that already exists in the registry at the computed tag.

---

## Image Hierarchy

The four images form a layered dependency chain:

```
public.ecr.aws/ubuntu/ubuntu:24.04   ← upstream Ubuntu base
        │
        ▼
  tt-forge-onnx-base           ← OS + build deps + Python 3.12 + Clang 17
  (Dockerfile.base)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  tt-forge-onnx-base-ird            tt-forge-onnx-ci
  (Dockerfile.ird, FROM=base)       (Dockerfile.ci)
  SSH, sudo, vim, zsh, GDB 14.2     Toolchain build artifacts only
        │
        ▼
  tt-forge-onnx-ird
  (Dockerfile.ird, FROM=ci)
  CI toolchain + developer tools
```

| Image name | Dockerfile | Purpose |
|---|---|---|
| `tt-forge-onnx-base-ubuntu-24-04` | `Dockerfile.base` | Ubuntu 24.04 with all build-time system packages (Clang 17, Ninja, ccache, Python 3.12, GTest, etc.) |
| `tt-forge-onnx-base-ird-ubuntu-24-04` | `Dockerfile.ird` (FROM=`base`) | Base image extended with interactive developer tools (SSH, sudo, vim, zsh, GDB 14.2) |
| `tt-forge-onnx-ci-ubuntu-24-04` | `Dockerfile.ci` | CI image: multi-stage build that compiles the tt-mlir/ttforge toolchain and copies only the output artifacts on top of the base image — keeps the image lean |
| `tt-forge-onnx-ird-ubuntu-24-04` | `Dockerfile.ird` (FROM=`ci`) | Full interactive developer image — CI toolchain artifacts plus all developer tools |

All images are hosted on the GitHub Container Registry (GHCR) under `ghcr.io/tenstorrent/tt-forge-onnx/`.

---

## Docker Tag Computation (`get-docker-tag.sh`)

Every image is tagged with a deterministic content-based hash computed by `.github/get-docker-tag.sh`. This ensures images are only rebuilt when their inputs actually change.

The hash is a SHA-256 over three inputs combined:

| Input | What it covers |
|---|---|
| **tt-mlir docker tag** | The tag computed inside the `third_party/tt-mlir` submodule (transitively covers its own Dockerfiles and env files) |
| **`env/` requirements hash** | SHA-256 of all git-tracked `*.txt` files under `env/` (Python requirements, pinned versions) |
| **Dockerfile hash** | SHA-256 of the concatenated content of `Dockerfile.base`, `Dockerfile.ci`, and `Dockerfile.ird` |

The final tag has the form `dt-<sha256>`, e.g. `dt-a3f7c2...`.

This means:
- Changing any Python requirement in `env/` triggers a rebuild.
- Changing any Dockerfile triggers a rebuild.
- Updating the `tt-mlir` submodule triggers a rebuild.
- Nothing else does — the tag is content-addressed, not time-based.

---

## The `build_and_push` function

```bash
build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3
    ...
}
```

Each image is built and pushed via this single reusable function. Its behavior:

1. **Check existence** — runs `docker manifest inspect <image>:<tag>`. If the tag already exists in the registry, the image is **skipped** entirely (no rebuild, no push). This is the cache-hit fast path.
2. **Build** — runs `docker build` with:
   - `--progress=plain` — full streaming log output (important for CI debugging)
   - `--build-arg FROM_TAG=$DOCKER_TAG` — injects the computed tag so child images pull the correct parent version
   - `--build-arg FROM_IMAGE=$from_image` — only passed when a `from_image` argument is provided; used by `Dockerfile.ird` to select whether to extend `base` or `ci`
   - `-f $dockerfile` — explicit Dockerfile path
   - Build context is always the **repo root** (`.`)
3. **Push** — pushes the newly built image to GHCR.

---

## Build order

The four `build_and_push` calls are ordered to respect the dependency chain:

```bash
# 1. Build the OS/compiler base image (no parent in this repo)
build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base

# 2. Build the interactive developer image on top of base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base

# 3. Build the CI image (multi-stage; compiles toolchain; no FROM_IMAGE arg)
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci

# 4. Build the full IRD image on top of the CI toolchain
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci
```

Steps 2 and 3 are independent of each other (both depend only on step 1), but steps 2 and 4 both use `Dockerfile.ird` with different `FROM_IMAGE` values (`base` vs `ci`).

---

## `Dockerfile.ci` — multi-stage build explained

The CI Dockerfile uses a two-stage build to avoid shipping the full build environment in the final image:

**Stage 1 (`ci-build`)** — extends the base image, copies the entire repo, and runs the toolchain build:
```
source env/activate
cmake -B env/build env
cmake --build env/build
```
This populates `/opt/ttmlir-toolchain` and `/opt/ttforge-toolchain` with compiled artifacts.

**Stage 2 (`ci`)** — starts fresh from the base image and only copies the two toolchain directories from stage 1:
```
COPY --from=ci-build /opt/ttmlir-toolchain /opt/ttmlir-toolchain
COPY --from=ci-build /opt/ttforge-toolchain /opt/ttforge-toolchain
```

The result is a lean image containing the base OS environment plus pre-built toolchain artifacts, without the source tree or intermediate build files.

---

## Script output

The last two lines of the script print the final CI image name and tag:

```
All images built and pushed successfully
CI_IMAGE_NAME:
ghcr.io/tenstorrent/tt-forge-onnx/tt-forge-onnx-ci-ubuntu-24-04:<tag>
```

The `build-image.yml` workflow captures this last line as the `docker-image` output and passes it downstream to `build.yml` and the test workflows, which use it as the container image for all CI jobs.

---

## Running locally

Prerequisites: Docker installed and authenticated to GHCR (`docker login ghcr.io`).

```bash
# From the repository root
./.github/build-docker-images.sh
```

To inspect the tag that would be computed without building:

```bash
./.github/get-docker-tag.sh
```

---

## Related files

| File | Role |
|---|---|
| `.github/get-docker-tag.sh` | Computes the deterministic content-based image tag |
| `.github/Dockerfile.base` | Ubuntu 24.04 base with system packages, Python 3.12, Clang 17 |
| `.github/Dockerfile.ci` | Multi-stage CI image (toolchain build + artifact copy) |
| `.github/Dockerfile.ird` | Interactive developer image (extends `base` or `ci`) |
| `.github/workflows/build-image.yml` | GitHub Actions workflow that calls this script |
