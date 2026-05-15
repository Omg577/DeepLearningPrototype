# Territory Painter ML-Agents Training

This project uses the Unity ML-Agents package for a competitive territory-painting environment.

## Unity Setup

Unity should import `com.unity.ml-agents` from `Packages/manifest.json`.

Press Play to run the environment manually. The runtime bootstrap creates the arena, agents, behavior parameters, decision requesters, camera, lighting, and HUD.

Controls in Play Mode:

- `Space`: pause
- `R`: reset the current episode
- `N`: end the current episode
- `F1`: cycle the blue team control mode
- `F2`: cycle the orange team control mode

Team control modes:

- `Learner`: controlled by `mlagents-learn` when a trainer is connected; falls back to heuristic movement otherwise.
- `TrainedModel`: uses `Assets/Resources/Models/TerritoryPainter.onnx` for frozen-policy inference.
- `RandomWalker`: uses a simple scripted random movement baseline.
- `Keyboard`: uses WASD input. Every agent on that team receives the same input.

This lets you run matchups like trained model vs trained model, learner vs trained model, learner vs random, or manual keyboard experiments.

## Training Setup

ML-Agents currently expects Python 3.10.x. This project has a local Python 3.10.11 install at `.tools\Python310` and a local trainer virtual environment at `.venv_mlagents`.

Recreate the trainer environment if needed:

```powershell
.\.tools\Python310\python.exe -m venv .venv_mlagents
.\.venv_mlagents\Scripts\python.exe -m pip install -r requirements-mlagents.txt
```

Start training:

```powershell
.\.venv_mlagents\Scripts\mlagents-learn.exe config\territory_painter_ppo.yaml --run-id territory-painter-v2
```

When the terminal says to start the Unity environment, press Play in the Unity editor.

Use a new run ID when the observation size changes. The current v2 environment uses richer grid observations and is not compatible with the earlier `territory-painter-001` checkpoint.

The behavior name is `TerritoryPainter`. Both teams use the same behavior name with different ML-Agents team IDs, so the trainer can run self-play.

## Environment Details

Observation size: `87`

Continuous actions: `2`

- action `0`: horizontal movement
- action `1`: vertical movement

Rewards:

- small step penalty for wasting time
- reward for painting neutral tiles
- larger reward for repainting enemy tiles
- small bonus for painting while spread out from teammates
- small penalty for crowding too close to teammates
- reward for tagging enemies on friendly territory
- end-of-episode reward based on final territory difference

Each agent observes a `5x5` local grid centered on itself. For each grid cell it receives tile ownership, teammate presence, and enemy presence.
