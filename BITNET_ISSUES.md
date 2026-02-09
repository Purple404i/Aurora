# bitnet.cpp Issues and Workarounds

During the integration of `bitnet.cpp` into the Aurora project, several issues were encountered with the official repository (https://github.com/microsoft/BitNet). This document details these issues, their causes, and suggested workarounds.

## 1. Clang Segmentation Faults

### Issue
When compiling `bitnet.cpp` using Clang 18.1.3 (the recommended version), the compiler frequently encounters segmentation faults during the compilation of `ggml.c`.

### Cause
This appears to be an environment-specific issue with Clang 18 on certain Linux distributions or containerized environments when handling complex macro expansions in `ggml.c` and `xmmintrin.h`.

### Workaround
Use GCC/G++ instead of Clang. Although Clang is recommended for optimal performance, GCC builds successfully and remains highly efficient.
```bash
cd bitnet.cpp
mkdir -p build && cd build
CC=gcc CXX=g++ cmake ..
make -j
```

## 2. Header Redefinitions (SPM Headers)

### Issue
The build fails with errors like `error: redefinition of enumerator 'GGML_TYPE_Q8_0'` and similar redefinition errors in `ggml.h`.

### Cause
The `bitnet.cpp` repository contains a submodule `3rdparty/llama.cpp` which sometimes has conflicting headers in `spm-headers/` and `ggml/include/`. The build system may pick up both, leading to redefinitions of the same types.

### Workaround
Ensure that only one set of headers is used. Renaming or removing the `spm-headers` directory within the submodule can resolve this if the conflict persists.
```bash
mv bitnet.cpp/3rdparty/llama.cpp/spm-headers bitnet.cpp/3rdparty/llama.cpp/spm-headers.bak
```

## 3. Missing Kernel Headers

### Issue
CMake fails with: `Cannot find source file: ../../../../include/bitnet-lut-kernels.h`.

### Cause
The `setup_env.py` script is supposed to copy these headers from `preset_kernels/` to `include/`, but it often fails to do so before the compilation phase is triggered, or it searches in the wrong directory.

### Workaround
Manually copy the required kernels for your model and architecture before building:
```bash
# For x86_64 and bitnet_b1_58-large
cp bitnet.cpp/preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h bitnet.cpp/include/bitnet-lut-kernels.h
cp bitnet.cpp/preset_kernels/bitnet_b1_58-large/kernel_config_tl2.ini bitnet.cpp/include/kernel_config.ini
```

## 4. `huggingface-cli` vs `hf`

### Issue
The `setup_env.py` and `bitnet_utils.py` previously assumed `huggingface-cli` was always available, but some newer environments use the `hf` command.

### Workaround
Updated `bitnet_utils.py` in the Aurora repo to automatically detect and use either `huggingface-cli` or `hf`.

## 5. Absolute Path Issues in `run_inference.py`

### Issue
The `bitnet.cpp` inference scripts often fail when called from outside the `bitnet.cpp` directory because they use relative paths for their internal components (like the `llama-cli` binary).

### Workaround
Updated `bitnet_utils.py` in Aurora to always use absolute paths for the `bitnet.cpp` directory and model files, ensuring reliability regardless of the current working directory.
