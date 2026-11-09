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

GMR uses its own Python 3.10 environment. Follow `third_party/GMR/README.md`. LAFAN1 BVH files go in `motions/raw/`.

## Deployment check (MuJoCo)

```bash
cd third_party/booster_deploy
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/deploy.py --task k1_mj2 --mujoco     # Booster's shipped K1 dance
```
