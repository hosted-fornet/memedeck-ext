# kinode-ml

The Kinode ml-ext and ml process.

## Usage

ORDER MATTERS

ml process MUST be started before ml-ext

```bash
# Terminal 1: run Kinode
# Use `develop` branch of Kinode
kit f -r ~/path/to/kinode

# Terminal 2: build & start ml process
kit bs ml

# Terminal 3: build & start python-ext
./ml-ext/src/main.py --port 8080 --models '{"mnist": "models/TFKeras.h5"}'

# Run a model (from Kinode terminal)
mnist:ml:sys
mnist_preloaded:ml:sys
```

If needed, Python dependencies can be fetched using

```bash
./ml-ext/setup.py
```

## Notes

Get vgg16 model from https://keras.io/api/applications/

## TODOs and obvious improvements

1. Add PyTorch
2. Add TensorFlow
3. Add TinyGrad
4. Handle errors
5. Better story for loading models (during run-time?)
