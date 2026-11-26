# Deployment Contract — draft

## Observation input (provisional, must match NERO client)
- `state`: float32 `[8]` = NERO joint7 + gripper.
- `images.front`: RGB uint8 image.
- `images.wrist`: RGB uint8 image.
- `prompt`: non-empty instruction string.

## Model input
- State: `[1, 60]`, slots 0:3 position, 3:9 rotation6d, 9 gripper; normalized with training q01/q99.
- Images and prompt use same processor/prompt layout as NERO training.

## Model output
- raw prediction: `[30, 60]`.
- de-normalized with training mean/std and action mask.
- slot 0:3 local position delta.
- slot 3:6 local axis-angle delta.
- slot 6 gripper absolute target (current config).
- slot 7 elbow delta.
- slot 8 Step Flag.

## Client output (target)
- `actions`: float32 `[30, 8]` absolute NERO joint targets + gripper target.
- Step Flag should remain a separate side-channel until the existing client protocol is confirmed.

The observation and output contract above is a design target inferred from the current NERO joint service and the new EE training layout. It has not yet been tested against the NERO client.
