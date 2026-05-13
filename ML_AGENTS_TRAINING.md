# Territory Painter ML-Agents Training

This project uses the Unity ML-Agents package for a competitive territory-painting environment.

## Unity Setup

Unity should import `com.unity.ml-agents` from `Packages/manifest.json`.

Press Play to run the environment manually. The runtime bootstrap creates the arena, agents, behavior parameters, decision requesters, camera, lighting, and HUD.

Controls in Play Mode:

- `Space`: pause
- `R`: reset the current episode
- `N`: end the current episode

## Training Setup

ML-Agents currently expects Python 3.10.x. This project has a local Python 3.10.11 install at `.tools\Python310` and a local trainer virtual environment at `.venv_mlagents`.

Recreate the trainer environment if needed:

```powershell
.\.tools\Python310\python.exe -m venv .venv_mlagents
.\.venv_mlagents\Scripts\python.exe -m pip install -r requirements-mlagents.txt
```

Start training:

```powershell
.\.venv_mlagents\Scripts\mlagents-learn.exe config\territory_painter_ppo.yaml --run-id territory-painter-001
```

When the terminal says to start the Unity environment, press Play in the Unity editor.

The behavior name is `TerritoryPainter`. Both teams use the same behavior name with different ML-Agents team IDs, so the trainer can run self-play.

## Environment Details

Observation size: `21`

Continuous actions: `2`

- action `0`: horizontal movement
- action `1`: vertical movement

Rewards:

- small step penalty for wasting time
- reward for painting neutral tiles
- larger reward for repainting enemy tiles
- reward for tagging enemies on friendly territory
- end-of-episode reward based on final territory difference
