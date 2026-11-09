# Setup

## Training machine (Isaac Lab)

Tested target: Isaac Sim 5.1.0 + Isaac Lab 2.3.2 (the versions booster_train is tested with), NVIDIA RTX GPU with 16 GB VRAM.

1. Install Isaac Sim and Isaac Lab following the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). The conda install is easiest.
2. Fetch the external repos: `python setup/clone_third_party.py`
3. In the Isaac Lab Python environment:
   ```bash
   pip install -e third_party/booster_assets   # follow its README if it needs extra steps
   pip install -e third_party/booster_train/source/booster_train
   python third_party/booster_train/scripts/list_envs.py   # should list Booster-K1-MJ_Dance_002-v0
   ```

## Retargeting (GMR)

GMR gets its own Python environment inside `third_party/GMR/.venv`. Retargeting runs on CPU, so any laptop works.

```bash
cd third_party/GMR
uv venv --python 3.11
uv pip install -e . "proxsuite==0.7.2"
```

Don't use `uv sync` on Windows: GMR's lockfile pins `proxsuite` 0.7.3, which has no Windows wheels, so the build fails. GMR's default IK solver is `daqp`, so proxsuite isn't actually used.

SMPL-X body models (needed for AIST++ and AMASS): register at https://smpl-x.is.tue.mpg.de/, download the SMPL-X model, and put `SMPLX_{NEUTRAL,MALE,FEMALE}.{npz,pkl}` in `third_party/GMR/assets/body_models/smplx/`. The license forbids redistribution, so never commit them.

Then retarget with our script (not GMR's batch script, which needs 30 GB free RAM and a CUDA GPU):

```bash
third_party/GMR/.venv/Scripts/python retarget/retarget_to_k1.py motions/raw/aistpp/ --limit 5
```

## Deployment check (MuJoCo)

```bash
cd third_party/booster_deploy
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/deploy.py --task k1_mj2 --mujoco     # Booster's shipped K1 dance
```
