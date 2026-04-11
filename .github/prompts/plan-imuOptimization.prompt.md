## Plan: IMU Rep Counter Optimization

To break past the 50% accuracy plateau, we will smooth the noisy IMU inputs, give the network more capacity to learn complex temporal patterns, and tackle any class imbalances.

**Steps**
1. Update data loader to apply a rolling average (window=5) to IMU features before normalization.
2. Upgrade the GRU model by expanding hidden nodes, adding a second layer, and replacing the final linear layer with an MLP classification head.
3. Calculate class frequencies across the dataset and apply inverse weights to the Cross Entropy Loss objective in training.

**Relevant files**
- Project/WorkUp_DATA.py — Add rolling mean in the CSV parsing loop.
- Project/WorkUp_GRU.py — Expand `RNNPredictor` hidden size, layers, and update the fully-connected layer to `nn.Sequential`.
- Project/WorkUp_Train.py — Calculate and pass class weights to `nn.CrossEntropyLoss`.

**Verification**
1. Run the training script to ensure the rolling mean parsing does not introduce NaNs at sequence start.
2. Monitor the training loop to verify that the Test Accuracy steadily climbs consistently above 50% while Training Loss declines seamlessly.

**Decisions**
- A simple moving average is chosen over complex Butterworth filters to guarantee easy C++ translation on the ESP32 later.
- Added Dropout layers to the GRU and MLP to counteract potential overfitting from the increased model capacity.