# train/windows/

Helpers for training on the team's Windows desktop (copied to `C:\isaac\` there).

| Script | What it does |
|---|---|
| `start_training.ps1` | Starts a booster_train run as a scheduled task (survives SSH disconnects) with a **watchdog**: if training dies before `-MaxIterations`, it resumes from the newest checkpoint (up to 5 restarts). Checkpoints every 250 iterations by default. |
| `stop_training.ps1` | Stops a run on purpose (leaves a stop flag so the watchdog doesn't restart it). Saved checkpoints are kept. |
| `no_throttle.ps1` | Opts a process out of Windows 11 power throttling and sets Normal priority (called by the launcher). |

```powershell
powershell -File C:\isaac\start_training.ps1 -Task Booster-K1-MJ_Dance_002-v0 -RunName mj2_full
powershell -File C:\isaac\start_training.ps1 -Task <task> -RunName <name> -MaxIterations 8000
powershell -File C:\isaac\stop_training.ps1 -RunName <name>
```

Logs in `C:\isaac\logs\`: `train_<run>.log` (training output) and `train_<run>.watchdog.log` (starts, crashes, resumes).
A resumed attempt writes a new `<timestamp>_<run>` folder; iteration numbers continue, and the final checkpoint is `model_<MaxIterations-1>.pt`.

## Why these exist (problems we hit)

- **Slow training:** scheduled tasks start at BelowNormal priority and Windows then power-throttles them (efficiency mode): 5.2 s vs 2.7 s per iteration.
- **Run killed overnight:** the first MJ run died at iteration 3421 with `forrtl: error (200): program aborting due to window-CLOSE event` when its minimized console window was closed. Training now runs with hidden windows, and the watchdog resumes it after any crash.
- **Still unprotected:** a reboot (e.g. Windows Update) or logging off ends the run. Restart it with the same command; it does not resume an older run automatically, so pass `-ExtraArgs "--resume --load_run <folder> --checkpoint model_<n>.pt"` if needed.
