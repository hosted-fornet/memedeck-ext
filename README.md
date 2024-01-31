# kinode-ml

The Kinode ml-ext and ml process.

## Usage

ORDER MATTERS

ml process MUST be started before ml-ext

```
# Terminal 1: run Kinode
# Use `develop` branch of Kinode
kit f -r ~/path/to/kinode

# Terminal 2: build & start ml process
kit bs ml

# Terminal 3: build & start python-ext
cd python-ext
cargo run --release -- --port 8080

# Run a model
# TODO
```
