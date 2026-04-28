# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path
import re

from setuptools.command.editable_wheel import editable_wheel
from wheel.wheelfile import WheelFile
from typing import Set

# WORKAROUND: make editable installation work
#
# The setuptools generates `MetaPathFinder` and hooks them to the python import machinery (via `sys.meta_path`),
# to be able to resolve imports of packages in editable installation. These finders are used as a fallback
# when python isn't able to find the package from the `sys.path`.
#
# However, their logic isn't able to resolve `import forge` properly. The problem is that the `forge` package
# is contained in the `forge` directory, which is a subdirectory of the root of the repository. If we execute
# `import forge` from the root of the repository, the `importlib` will find the top-level directory `forge` and
# won't fallback to the `MetaPathFinder` logic.
#
# To workaround this, we create our `.pth` file and add it to the editable wheel. Python will automatically
# load this file and populate the `sys.path` with the paths specified in the `.pth` file.
#
# NOTE: Needs `wheel` to be installed.
class EditableWheel(editable_wheel):
    def run(self):
        # Build the editable wheel first.
        super().run()

        # Create a .pth file with paths to the repo root, ttnn and tools directories.
        # This file gets loaded automatically by the python interpreter and its content gets populated into `sys.path`;
        # i.e. as if these paths were added to the PYTHONPATH.
        pth_filename = "forge-custom.pth"
        pth_content = f"{Path(__file__).parent / 'forge' }\n"

        print(f"EditableWheel.run: adding {pth_filename} to the wheel")

        # Find .whl file in the dist_dir (e.g. `forge-0.1*.whl`)
        wheel = next((f for f in os.listdir(self.dist_dir) if f.endswith(".whl") and "editable" in f), None)

        assert wheel, f"Expected to see editable wheel in dist dir: {self.dist_dir}, but didn't find one"

        # Add the .pth file to the wheel archive.
        WheelFile(os.path.join(self.dist_dir, wheel), mode="a").writestr(pth_filename, pth_content)


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        if self.editable_mode:
            return
        for ext in self.extensions:
            if "forge" in ext.name:
                self.build_forge(ext)
            else:
                raise Exception("Unknown extension")

    def build_forge(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install forge at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd / "build"
        install_dir = extension_path.parent / "forge"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DTTMLIR_RUNTIME_DEBUG=OFF",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])
        self.spawn(["cmake", "--install", str(build_dir)])

        _prune_bloat_from_wheel(install_dir)
        _add_so_dependencies(install_dir)


def _prune_bloat_from_wheel(install_dir: str) -> None:
    """Remove unnecessary files from the wheel that are not needed for runtime."""
    _remove_broken_symlinks(install_dir)
    _remove_static_archives(install_dir)

    _remove_bloat_dir(install_dir / "lib" / "cmake")
    _remove_bloat_dir(install_dir / "lib" / "pkgconfig")
    _remove_bloat_dir(install_dir / "include")
    _remove_bloat_dir(install_dir / "tt-metal" / ".cpmcache")
    _fix_file(install_dir / "lib" / "libtt-umd.so.0", install_dir)
    _remove_bloat_file(install_dir / "lib" / "libtt-umd.so")
    _remove_bloat_file(install_dir / "lib" / "libtt-umd.so.0.*")
    _strip_shared_objects(install_dir)


def _fix_file(file_path: Path, install_dir: Path) -> None:
    if file_path.is_symlink():
        target = file_path.resolve()
        file_path.unlink()
        shutil.copy2(target, file_path)
    adjust_rpath(file_path, "$ORIGIN:$ORIGIN/lib", install_dir)


def _remove_broken_symlinks(root: Path) -> None:
    """Remove broken symlinks that would cause wheel packaging to fail."""
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            rel = path.relative_to(root)
            print(f"Removing broken symlink: {rel}")
            path.unlink()


def _strip_shared_objects(root: Path) -> None:
    strip_path = shutil.which("strip")
    if strip_path is None:
        print("strip tool not found; skipping debug symbol stripping")
        return

    for so_file in root.rglob("*.so"):
        if so_file.is_symlink() or not so_file.is_file():
            continue
        try:
            subprocess.run([strip_path, "--strip-unneeded", str(so_file)], check=True)
            rel = so_file.relative_to(root)
            print(f"Stripped debug symbols: {rel}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to strip {so_file}: {exc}")


def _remove_static_archives(root: Path) -> None:
    for archive in root.rglob("*.a"):
        if archive.is_symlink() or not archive.is_file():
            continue
        rel = archive.relative_to(root)
        if rel.parts and rel.parts[0] in ("lib", "lib64"):
            print(f"Removing static archive: {rel}")
            archive.unlink()


def _remove_bloat_dir(dir_path: Path) -> None:
    if dir_path.exists() and dir_path.is_dir():
        print(f"Removing bloat directory: {dir_path}")
        shutil.rmtree(dir_path)


def _remove_bloat_file(file_path: Path) -> None:
    # Handle wildcard patterns
    if "*" in file_path.as_posix():
        parent = file_path.parent
        pattern = file_path.name
        for matched_file in parent.glob(pattern):
            if matched_file.is_file():
                print(f"Removing bloat file: {matched_file}")
                matched_file.unlink()
    else:
        if file_path.exists() and file_path.is_file():
            print(f"Removing bloat file: {file_path}")
            file_path.unlink()


def adjust_rpath(so_file: str, new_rpath: str, install_dir: Path) -> None:
    """Adjust rpath of a .so file using patchelf."""
    patchelf_path = shutil.which("patchelf")
    if patchelf_path is None:
        print("patchelf not found; skipping rpath adjustment")
        return

    try:
        subprocess.run([patchelf_path, "--set-rpath", new_rpath, so_file], check=True)
        rel = Path(so_file).relative_to(install_dir)
        print(f"Adjusted rpath: {rel}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to adjust rpath for {so_file}: {exc}")


def _add_so_dependencies(install_dir: Path) -> None:
    """Copy non-standard .so dependencies into install_dir/lib and adjust rpath."""

    def get_so_dependencies(so_file: str) -> Set[str]:
        """Get list of .so dependencies for a given .so file."""
        try:
            output = subprocess.check_output(["ldd", so_file], stderr=subprocess.DEVNULL, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return set()

        dependencies = set()
        for line in output.splitlines():
            line = line.strip()
            if not line or "=>" not in line:
                continue
            parts = line.split("=>")
            if len(parts) < 2:
                continue
            path = parts[1].strip().split()[0]
            if path and path.startswith("/"):
                dependencies.add(path)
        return dependencies

    def is_standard_library(so_path: str) -> bool:
        """Check if .so is a standard system library."""
        standard_libs = ["libpython", "libc.", "libm.", "libdl.", "libpthread.", "libstdc++", "libudev."]

        so_name = so_path.split("/")[-1]
        if any(so_name.startswith(lib) for lib in standard_libs):
            print(f"{so_name} is a standard library, treating as standard.")
            return True

        try:
            output = subprocess.check_output(["dpkg", "-S", so_path], stderr=subprocess.DEVNULL, text=True)
            if any(line.startswith("libc6:") for line in output.splitlines()):
                print(f"{so_path} is a standard library.")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{so_path} is NOT a standard library.")
        return False

    def collect_libs(lib_dir: Path) -> set[str]:
        all_libs = set()
        for so_file in install_dir.rglob("*.so*"):
            if so_file.is_symlink() or not so_file.is_file():
                continue

            all_libs.update(get_so_dependencies(str(so_file)))

        print(f"Total dependencies found: {len(all_libs)}")
        print(all_libs)
        return all_libs

    def copy_libs(lib_dir: Path, all_libs: Set[str]) -> None:
        copied_libs = set()
        for dep in all_libs:
            dep_path = Path(dep).resolve()
            if dep_path.parts[:-1] != install_dir.parts and not is_standard_library(dep):
                dep_path = Path(dep)
                dest_path = lib_dir / dep_path.name
                if not dest_path.exists():
                    print(f"Copying dependency {dep} to {dest_path}...")
                    shutil.copy2(dep, dest_path)
                    # adjust_rpath(dest_path, "$ORIGIN/../lib:$ORIGIN")
                    copied_libs.add(dest_path)
                    adjust_rpath(str(dest_path), "$ORIGIN:$ORIGIN/lib", install_dir)
            else:
                print(f"Skipping standard/our library dependency: {dep}")

        print(f"Copied dependencies {len(copied_libs)}:")
        print(copied_libs)

    def get_full_paths(lib_names: Set[str]) -> Set[str]:
        """Convert library names to their full paths using ldconfig or system search."""
        full_paths = set()
        output = subprocess.check_output(["ldconfig", "-p"], text=True)
        for lib_name in lib_names:
            for line in output.splitlines():
                if lib_name in line:
                    parts = line.split("=>")
                    if len(parts) > 1:
                        path = parts[1].strip()
                        if path:
                            full_paths.add(path)
                            break
        return full_paths

    alllibs = collect_libs(install_dir)
    alllibs.update(collect_libs(install_dir / "lib"))
    copy_libs(install_dir / "lib", alllibs)


with open("README.md", "r") as f:
    long_description = f.read()

# Compute requirements
with open("env/core_requirements.txt", "r") as f:
    core_requirements = f.read().splitlines()

with open("env/linux_requirements.txt", "r") as f:
    linux_requirements = [r for r in f.read().splitlines() if not r.startswith("-r")]


def collect_model_requirements(requirements_root: str) -> list[str]:
    """
    Collect and deduplicate model-specific Python package requirements from all `requirements.txt` files
    under the given root directory.

    Handles version conflicts as follows:
    - If the same package appears with different versions, an error is raised.
    - If one occurrence has a version and another does not, the no-version spec (latest) is preferred.
    - Duplicate entries with the same version are ignored.

    Args:
        requirements_root (str): Path to the directory to search for requirements.txt files.

    Returns:
        List[str]: A sorted list of unique requirement strings (e.g., ["numpy>=1.21", "torch"]).
    """

    # Regex to capture package name and optional version specifier
    # e.g., "torch>=2.1.0" → ("torch", ">=2.1.0")
    version_pattern = re.compile(r"^([a-zA-Z0-9_\-]+)([<>=!~]+.+)?$")

    # Tracks source of each package → {package_name: (version_str, file_path)}
    requirement_map = {}

    # Final deduplicated output → {package_name: version_str}
    # This ensures no duplicates and consistent overwrite behavior
    final_requirements = {}

    # Find all requirements.txt files under the given root directory
    for req_file in Path(requirements_root).rglob("requirements.txt"):

        # Open and read each requirements.txt file
        with open(req_file, "r") as f:
            for line in f:
                line = line.strip()  # Remove whitespace at start/end

                # Skip empty lines or comments
                if not line or line.startswith("#"):
                    continue

                # Extract package name and version using regex
                match = version_pattern.match(line)
                if not match:
                    raise ValueError(f"Unrecognized requirement format: '{line}' in {req_file}")

                # Extract matched groups
                pkg_name, version = match.groups()

                # If version is None (e.g., just "torch"), treat it as empty string
                version = version or ""

                if pkg_name in requirement_map:
                    # We've already seen this package in another file
                    prev_version, prev_file = requirement_map[pkg_name]

                    if prev_version != version:
                        # Conflict: one has version, other has a different version or none

                        if version == "":
                            # Current one has no version → prefer this (more general)
                            requirement_map[pkg_name] = ("", req_file)
                            final_requirements[pkg_name] = ""

                        elif prev_version == "":
                            # Previous one was no version → keep it, ignore current versioned one
                            continue

                        else:
                            # Actual version mismatch → raise an error
                            raise AssertionError(
                                f"Conflicting versions for '{pkg_name}':\n"
                                f"- {prev_version} in {prev_file}\n"
                                f"- {version} in {req_file}"
                            )

                    # else: same version → ignore duplicate
                else:
                    # First time seeing this package → record it
                    requirement_map[pkg_name] = (version, req_file)
                    final_requirements[pkg_name] = version

    # Convert the final dictionary to a list of strings
    # e.g., {"torch": "==2.1.0", "numpy": ""} → ["torch==2.1.0", "numpy"]
    return [pkg + ver if ver else pkg for pkg, ver in sorted(final_requirements.items())]


model_requirements_root = "forge/test/models"
model_requirements = collect_model_requirements(model_requirements_root)

requirements = core_requirements + linux_requirements + model_requirements


# Parse build type from command line arguments
def parse_build_type():
    """
    Parse --build-type argument from command line.
    This is passed from build.yml: python3 setup.py bdist_wheel --build-type release
    Defaults to "release" if not provided.
    """
    build_type = None
    args_to_remove = []

    for i, arg in enumerate(sys.argv):
        if arg == "--build-type":
            if i + 1 < len(sys.argv):
                build_type = sys.argv[i + 1]
                args_to_remove = [i, i + 1]
                break
            else:
                raise ValueError("--build-type argument requires a value")

    # Remove the arguments so setuptools doesn't see them
    for i in reversed(args_to_remove):
        sys.argv.pop(i)

    # Default to Release if not provided
    if build_type is None:
        build_type = "release"

    return build_type.lower()


def get_git_commit_hash(repo_path: str = ".") -> str:
    """Get full git commit hash from a repository path."""
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ValueError(f"Failed to get git commit hash from {repo_path}: {e}")


def get_tt_mlir_commit_hash() -> str:
    """
    Get tt-mlir SHA from the git submodule.
    In tt-forge-onnx, tt-mlir is a git submodule, so we get the commit hash directly.
    """
    mlir_path = "third_party/tt-mlir"
    if not os.path.exists(mlir_path):
        raise ValueError(f"tt-mlir submodule not found at {mlir_path}")

    return get_git_commit_hash(mlir_path)


def get_tt_metal_commit_hash() -> str:
    """
    Fetch tt-metal SHA from tt-mlir repo's CMakeLists.txt.
    Matches tt-xla approach: https://github.com/tenstorrent/tt-xla/blob/main/python_package/setup.py#L86
    """

    # Extract tt-metal SHA from third_party/tt-mlir/third_party/CMakeLists.txt
    cmake_file = pathlib.Path(__file__).resolve().parent / "third_party" / "tt-mlir" / "third_party" / "CMakeLists.txt"
    with cmake_file.open() as f:
        cmake_content = f.read()
    metal_match = re.search(r'set\(TT_METAL_VERSION "([^"]+)"\)', cmake_content)
    if not metal_match:
        raise ValueError("Failed to extract TT_METAL_VERSION from tt-mlir CMakeLists.txt")
    return metal_match.group(1)


def get_build_date() -> str:
    """Get build date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def format_build_summary(build_type: str) -> str:
    """Format build metadata summary string."""
    commit_hash = get_git_commit_hash()
    tt_mlir_commit = get_tt_mlir_commit_hash()
    tt_metal_commit = get_tt_metal_commit_hash()
    built_date = get_build_date()

    return (
        f"commit={commit_hash}, "
        f"tt-mlir-commit={tt_mlir_commit}, "
        f"tt-metal-commit={tt_metal_commit}, "
        f"built-date={built_date}, "
        f"build-type={build_type}"
    )


# Parse build type from command line
build_type = parse_build_type()

# Get build metadata summary
build_summary = format_build_summary(build_type)

# Compute a dynamic version from git
short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
date = (
    subprocess.check_output(["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"])
    .decode("ascii")
    .strip()
)
version = "0.1." + date + "+dev." + short_hash

forge_c = TTExtension("forge")

# Find packages as before
packages = [p for p in find_packages("forge") if not p.startswith("test")]


setup(
    name="tt_forge_onnx",
    version=version,
    description=build_summary,
    install_requires=requirements,
    packages=packages,
    package_dir={"forge": "forge/forge"},
    ext_modules=[forge_c],
    cmdclass={"build_ext": CMakeBuild, "editable_wheel": EditableWheel},
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
