# Installation
```shell
cd benchmark/libero
uv sync
source .venv/bin/activate
cd ../../third_party/libero
uv pip install -e .
cd ../..
```

### TroubleShooting
If the installation failed due to error like `Compatibility with CMake < 3.5 has been removed from CMake. Update the VERSION argument <min> value.`, please set the configurations by
```shell
export CMAKE_POLICY_VERSION_MINIMUM=X.X # your cmake version
