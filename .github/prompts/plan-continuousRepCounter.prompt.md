## Plan: Real-Time Continuous Rep Counter

**TL;DR** 
Build a PyTorch GRU model to classify IMU time-series windows into 0, 1, 2, or 3 reps. The ESP32-C3 streams raw IMU data over serial to a PC holding a real-time FIFO buffer. The PC runs the GRU continuously on this sliding buffer to detect rep events and maintains a running total from 0 to infinity.

**Steps**
1. **Data Pipeline Strategy (Current phase)**
   - Finalize `IMURepDataset` to load all 336 CSVs into RAM.
   - Implement `collate_fn` to handle variable-length sequences using padding and `pack_padded_sequence`.
2. **Model Architecture**
   - Create `RNNPredictor` with a GRU layer (input size 10) and a Linear classification layer (output size 4 for classes 0, 1, 2, 3).
3. **Training & Validation Loop**
   - Write the train/test loops using `CrossEntropyLoss` (handling the 4 discrete classes) and `Adam` optimizer. Track accuracy to ensure generalization across users.
4. **Real-Time Inference Simulation**
   - Write a simulation script to slide a FIFO window across a continuous stream of test data.
   - **Crucial step**: Map the discrete window outputs (0, 1, 2, 3) to a state-machine or cooldown logic that safely increments a total rep counter without double-counting overlapping windows from the same rep.
5. **Hardware Integration & Deliverables**
   - Connect the trained model to the live wireless data stream from the ESP32-C3.
   - Finalize timesheet and record project presentation emphasizing the pipeline architecture.

**Relevant files**
- Project/WorkUp_DATA.py — `IMURepDataset` and `collate_fn` data pipeline.
- Project/WorkUp_GRU.py — GRU model definition, training loop, and evaluation.
- Project/WorkUp_Train.py — Main training script to run the training and validation loops.
- Project/ — Future location for the real-time inference continuous loop.

**Verification**
1. Verify batches from `DataLoader` yield correctly padded tensors.
2. Verify live continuous inference counts sequential reps ascending smoothly (0 -> infinity) from the physical ESP32-C3 without dropping reps or phantom counting noise.

**Decisions**
- The dataset (336 files) will be loaded entirely into RAM to prevent disk I/O bottlenecks.
- Imbalance in user data and the `0` label class will be preserved to maximize variance and background noise recognition.
- Model inference will happen on the PC via Python, receiving data over serial. The ESP32-C3 will only handle raw data streaming, not inference.

**Further Considerations**
1. **Overlap Prevention**: In a continuous FIFO buffer, a single physical rep might trigger a `1` in multiple overlapping windows. We will need a cooldown logic or state-tracking in Step 4 to prevent counting the same rep twice. Do you anticipate needing a script specifically for this logic? Probably not an individual file, but it will be a crucial part of the real-time inference loop.