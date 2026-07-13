#!/usr/bin/env bash
# Activate the CUDA-enabled WSL venv and expose the nvidia pip CUDA libraries to
# the dynamic loader. TensorFlow 2.21's pip GPU wheel ships CUDA/cuDNN as
# nvidia-*-cu12 packages under site-packages/nvidia/*/lib, but does not always
# add them to LD_LIBRARY_PATH automatically - hence "Cannot dlopen some GPU
# libraries" and an empty GPU list. This puts them on the path.
source ~/venvs/mht/bin/activate
NVBASE="$(python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="$(find "$NVBASE" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
PTXAS_DIR="$(dirname "$(find "$NVBASE" -name ptxas -type f 2>/dev/null | head -1)")"
[ -n "$PTXAS_DIR" ] && export PATH="$PTXAS_DIR:$PATH"
