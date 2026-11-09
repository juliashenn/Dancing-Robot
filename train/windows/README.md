# train/windows/

Helpers for training on the team's Windows desktop (copied to `C:\isaac\` there).

- `start_training.ps1`: starts a booster_train run as a scheduled task (survives SSH disconnects), logs to `C:\isaac\logs\train_<run>.log`.
- `no_throttle.ps1`: opts a process out of Windows 11 power throttling and sets Normal priority.

**Why:** scheduled tasks start at BelowNormal priority, and Windows then throttles the process (efficiency mode). That made K1 training 1.9x slower (5.2 s vs 2.7 s per iteration at 4096 envs).

```powershell
powershell -File C:\isaac\start_training.ps1 -Task Booster-K1-MJ_Dance_002-v0 -RunName mj2_full
# stop:  Stop-ScheduledTask -TaskName train_mj2_full
```
