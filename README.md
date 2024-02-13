# memedeck-ext

The Kinode memedect-ext and predict process.

## Usage

ORDER MATTERS

predict process MUST be started before memedeck-ext

```bash
# Terminal 1: run Kinode
# Use `develop` branch of Kinode
kit f -r ~/path/to/kinode

# Terminal 2: build & start ml process
kit bs predict

# Terminal 3: build & start python-ext
./memedeck-ext/src/main.py --port 8082 --kmeans ~/scripts/models/kmeans_model_500_4.bin --pca ~/scripts/models/pca_model_500_4.bin --cluster-to-template-ids ~/scripts/models/cluster_to_template_ids.bin --vgg16 ~/scripts/models/vgg16_2.keras

# Run memedeck
```

If needed, Python dependencies can be fetched using

```bash
./memedeck-ext/setup.py
```

## Notes

Get vgg16 model from https://keras.io/api/applications/

or models in general from https://github.com/holium/memedeck/pull/13
