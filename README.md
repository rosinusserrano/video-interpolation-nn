# Video Frame Interpolation with Neural Nets

## TODOs
- [ ] Encoding the first and last patch embeddings separately may not be the best idea. Instead concatenate x and z and then just do a single forward pass through the encoder.
- [ ] Instead of predicting the frame in the middle between x and z, predict the next frame after x and condition that on the embedding of z
- [ ] Add convolutional layers at the end for resolution
- [ ] Make the upconv not with stride 2, but make  with stride 1 and make more  often between skip connections
- [ ] Use different datasets
- [ ] Predict the change from one frame to the next (full skip connections), then also pass as a parameter the amount of change. That would be something like (y_pred - x).abs().sum()
- [ ] generate video with moviepy
- [ ] add more attention layers
- [ ] very the kernel size

## DONE
- [x] Instead of predicting next frame or frame extractly in between, let the representation vector of predicted frame be linear interpolation between the two and the position of interpolation be an additional training input that defines how to compute the interpolated images representation -> **PatchAttentionUNETLinearInterpolation**