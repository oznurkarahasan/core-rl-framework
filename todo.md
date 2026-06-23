## Phase 1: Foundation (Memory & Data Management)

[x] Task 1.1: Design the GenericReplayBuffer class. Ensure it supports multimodal observations (e.g., nested Dict structures combining image matrices and numerical telemetrics), rather than just standard flat arrays.

[x] Task 1.2: Implement push (experience storage) and sample (batch retrieval) methods, optimizing them for seamless integration with PyTorch tensors.

## Phase 2: The Brain Core (Decision Engines)

[x] Task 2.1: Create the BaseAgent Abstract Base Class (ABC). Define the standard interface template, including select_action, update, and save/load methods.

[x] Task 2.2: Implement the SACAgent (Soft Actor-Critic) for continuous action spaces. Build the Actor and Critic neural networks and integrate automatic entropy tuning (alpha).

[x] Task 2.3: Develop the basic PPOAgent skeleton, providing support for discrete action spaces (e.g., trading bots or grid-based environments).

## Pre-Phase 3: Cleanup & Validation
 
[x] Task 2.4 (Cleanup): Add gradient clipping to PPOAgent.update() to prevent training instability from large gradient updates.
 
[x] Task 2.5 (Cleanup): Make PPO entropy coefficient a constructor hyperparameter (`entropy_coef=0.01`) instead of a hardcoded magic number.
 
[x] Task 2.6 (Cleanup): Rename `self.MseLoss` → `self.mse_loss` in PPOAgent to follow Python naming conventions.
 
[x] Task 2.7 (Cleanup): Add a coverage threshold to CI (e.g., fail if coverage drops below 80%) so regressions are caught automatically.
 
[x] Task 2.8 (Cleanup): Update reports/12June_foundation-and-brain-core.md — remove stale "Known Remaining Issues" entries that have already been resolved (empty __init__.py, missing pyproject.toml, no unit tests).
 
[x] Task 2.9 (Validation): Write an end-to-end Gymnasium training loop. Use LunarLanderContinuous-v3 with SACAgent and LunarLander-v3 with PPOAgent. This validates that the brain works before building on top of it, and establishes a baseline entropy range needed for Phase 3 threshold tuning.

## Phase 3: Active Learning Layer (Human-in-the-Loop)

[x] Task 3.1: Develop the ActiveLearningCore module. Implement the mathematical functions required to calculate policy entropy (uncertainty distribution) during inference.

[x] Task 3.2: Design an asynchronous "Review Queue". Ensure that when the uncertainty threshold is exceeded, the uncertain state is queued for human review without freezing the main simulation loop.

## Phase 4: Continual Learning

[x] Task 4.1: Build the EWC (Elastic Weight Consolidation) class. Implement the logic to compute and store the Fisher Information Matrix to identify and protect critical network weights.

[x] Task 4.2: Integrate the EWC penalty function into the SAC/PPO agent's loss calculation to actively prevent catastrophic forgetting when learning new edge cases.

## Phase 5: Feedback Engine (Human Interface & API)

[x] Task 5.1: Set up the FastAPI backend. Create API endpoints (GET /health, GET /queue/status, GET /queue/items, POST /queue/resolve/{id}) to expose the active learning queue to the web-based teacher UI.

[x] Task 5.2: Implement the reward-update bridge. Ensure that once a human label is submitted, the system dynamically updates the corresponding experience reward within the Replay Buffer.

## Phase 6: Minimal Web UI

[x] Task 6.1: Build a minimal HTML/JS interface that calls the Phase 5 API. Displays pending uncertain states from the queue, allows the human teacher to approve or reject with a reward value.

[x] Task 6.2: Connect the UI to the /queue/items and /queue/resolve/{id} endpoints. Ensure the full human-in-the-loop cycle works end-to-end: uncertain state appears on screen → human labels it → reward updates in buffer.

## Phase 7: Visual Learning Platform (CAPTCHA-Style Active Learning)

### Phase 7A: Data & Session Management

[x] Task 7.1: Set up SQLite database schema for sessions, images, and labels.
    - sessions: id, name, created_at, checkpoint_path
    - categories: id, session_id, name, created_at (session-scoped, normalized)
    - images: id, session_id, filename, path, uploaded_at
    - labels: id, image_id, category_id, confirmed, created_at

[x] Task 7.2: Implement image upload endpoint (POST /platform/upload).
    - Accept batch image uploads, auto-resize to 416x416
    - Save to uploads/ directory, register in SQLite

[x] Task 7.3: Implement category management endpoints.
    - POST /platform/categories — add new category dynamically
    - GET /platform/categories — list all categories for current session

### Phase 7B: CAPTCHA-Style Labeling UI

[x] Task 7.4: Build annotate.html — CAPTCHA-style labeling interface.
    - Show 1 image at a time: "Is this a stop sign?" → [Yes] [No]
    - Or grid mode: "Select all stop signs" → multiple images shown (not done)
    - Progress bar: X images labeled / total

[x] Task 7.5: Connect labeling UI to FastAPI.
    - POST /platform/label — save human answer to SQLite
    - GET /platform/next — fetch next unlabeled image for review

### Phase 7C: Model Training

[x] Task 7.6: Implement lightweight classification model (MobileNetV2).
    - Input: 416x416 image
    - Output: category probabilities
    - Supports resume from checkpoint (continual learning via EWC) (not done)

[x] Task 7.7: Implement training endpoint (POST /platform/train).
    - Load labeled data from SQLite
    - Resume from last checkpoint if exists
    - Train in background, stream progress via GET /platform/train/status

[x] Task 7.8: Implement ONNX export endpoint (POST /platform/export).
    - Export current checkpoint to checkpoints/model_v{n}.onnx
    - Return download link

### Phase 7D: Test & Validation

[x] Task 7.9: Build test UI — model prediction screen.
    - Show unlabeled image → model predicts → confidence score shown
    - Human confirms or rejects → accuracy score updates live

[x] Task 7.10: Implement accuracy tracking endpoint (GET /platform/stats).
    - Per-category accuracy
    - Confusion matrix
    - Total images labeled / confirmed

### Phase 7E: Qwen2.5-VL Comparison

[ ] Task 7.11: Integrate Ollama + Qwen2.5-VL-3B as auto-labeling assistant.
    - POST /platform/qwen/predict — send image, get Qwen's label suggestion
    - Show suggestion in UI: "Qwen thinks: stop sign (94%)" → sen onayla/reddet

[ ] Task 7.12: Build comparison screen.
    - Same image → Your model prediction vs Qwen prediction
    - Side by side confidence scores
    - "Where does my model disagree with Qwen?" görünür

## Phase 7F: Segmentation Labeling & Training

### Phase 7F-A: Segmentation Data Collection

[ ] Task 7F.1: Extend SQLite schema for segmentation masks.
    - Add `masks` table: id, image_id, category_id, mask_path, created_at
    - Mask stored as PNG (binary pixel map, same size as image)

[ ] Task 7F.2: Add canvas drawing UI to annotate.html.
    - Toggle between classification mode and segmentation mode
    - Canvas overlay on image: brush tool to draw mask
    - Eraser tool, brush size control
    - "Save mask" → POST /platform/segmentation/mask
    - Saved mask shown as colored overlay

[ ] Task 7F.3: Add segmentation mask endpoints to platform_router.
    - POST /platform/segmentation/mask — save drawn mask as PNG
    - GET  /platform/segmentation/next — next image without mask
    - GET  /platform/segmentation/mask/{image_id} — serve mask PNG

### Phase 7F-B: Segmentation Model Training

[ ] Task 7F.4: Implement MobileNetV2 + U-Net decoder.
    - Encoder: MobileNetV2 backbone (pretrained, shared with classifier)
    - Decoder: U-Net style upsampling blocks
    - Output: binary mask, same resolution as input (224x224)
    - Supports multiple segmentation categories (şerit, engel, yol)

[ ] Task 7F.5: Implement SegmentationTrainer.
    - Loads image + mask pairs from DB
    - Loss: BCE + Dice loss combination
    - Resume from checkpoint (continual learning)
    - on_progress callback: epoch, loss, iou (intersection over union)

[ ] Task 7F.6: Add segmentation training endpoints.
    - POST /platform/segmentation/train   — start training in background
    - GET  /platform/segmentation/status  — poll progress (epoch, loss, IoU)
    - POST /platform/segmentation/export  — export to ONNX

### Phase 7F-C: Test & Validation

[ ] Task 7F.7: Add segmentation test UI.
    - Show image → model draws predicted mask overlay
    - Human rates: good / bad / partial
    - IoU score shown live

[ ] Task 7F.8: ONNX inference for segmentation.
    - SegmentationPredictor class
    - Input: image path
    - Output: mask array + overlay image
    - Compatible with ROS2 node (same interface as TwinLiteNet)