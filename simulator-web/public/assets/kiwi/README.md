# Kiwi robot CAD asset

`kiwi-robot.glb` is the visual model exported from the Onshape `Assembly 1` in
the [`solved-robot` document](https://cad.onshape.com/documents/d50ef7c2d9aa5075f93c03cf/w/dc8ff0658582b32bf40e2101/e/b046d07eaa0552db828cb31e).

Onshape export settings:

- format: GLB
- orientation: Y axis up
- resolution: Coarse
- compression: enabled
- hidden instances: excluded

The raw export is optimized into one render primitive while retaining the
assembly colors:

```sh
npx --yes @gltf-transform/cli@4.4.2 optimize \
  kiwi-robot-onshape.glb kiwi-robot.glb \
  --compress draco \
  --texture-compress false \
  --instance false \
  --simplify-ratio 0.35 \
  --simplify-error 0.0005
```

The renderer rotates the model so the CAD camera axis is simulator-forward and
grounds it from the loaded mesh bounds. Motion and collision geometry remain in
the deterministic simulation core.
