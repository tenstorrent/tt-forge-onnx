Overview
-----
This directory contains all environment dependencies for tt-forge. By default all dependencies will be installed
to the /opt/ttforge-toolchain directory. To override this behavior, set the `TTFORGE_TOOLCHAIN_DIR` environment variable.
Note that you need to build toolchain only once, unless you want to update it.

Dependencies:
-----
* [Python.3.12](https://www.python.org/downloads/release/python-3120/) - Version of python which is compatible with the project.
* [cmake3.20](https://cmake.org/download/) - Version of cmake which is compatible with the project.

Building the toolchain
-----
To build toolchain you need to run the following commands from the root directory of the project:
* `source env/activate` - This command will set all environment variables needed for the project.
* `cmake -B env/build env` - This command will generate the build files for the project.
* `cmake --build env/build` - This command will build the project.
