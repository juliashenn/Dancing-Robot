# third_party/

Local clones of external repos, fetched by `python setup/clone_third_party.py` at the commits pinned in that script. Everything here except this README is gitignored. Don't edit these clones: put our changes in `train/`, `deploy/`, etc.

| Repo | Used for |
|---|---|
| booster_train | Isaac Lab K1 motion-tracking tasks (BeyondMimic), `csv_to_npz.py`, train/play scripts |
| booster_deploy | Running exported policies in MuJoCo and on the real K1 |
| booster_assets | K1 robot models and Booster's example motions |
| GMR | Retargeting human mocap to the K1 (`booster_k1`) |

Isaac Lab and Isaac Sim are installed separately (see `docs/setup.md`).
