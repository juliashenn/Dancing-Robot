# Start a booster_train run on the desktop as a Windows scheduled task, with a watchdog that
# resumes from the newest checkpoint if training dies before reaching -MaxIterations.
#
#   powershell -File C:\isaac\start_training.ps1 -Task Booster-K1-MJ_Dance_002-v0 -RunName mj2_full
#   powershell -File C:\isaac\start_training.ps1 -Task <task> -RunName <name> -MaxIterations 8000
#   powershell -File C:\isaac\stop_training.ps1 -RunName <name>      (stop on purpose; no auto-restart)
#
# Logs: C:\isaac\logs\train_<RunName>.log (training output), train_<RunName>.watchdog.log (restarts)
# Checkpoints: C:\isaac\Dancing-Robot\logs\rsl_rl\<experiment>\<timestamp>_<RunName>\model_<iter>.pt
#   (a resumed attempt writes a new <timestamp>_<RunName> folder; iteration numbers continue)
#
# Lessons built in:
#  * Scheduled tasks start at BelowNormal priority and Windows 11 power-throttles them (~1.9x slower):
#    the task runs at normal priority and no_throttle.ps1 is applied to each training process.
#  * Closing a visible console window aborts training ("forrtl: error (200) ... window-CLOSE event"):
#    everything runs with hidden windows.
param(
    [Parameter(Mandatory)] [string]$Task,
    [Parameter(Mandatory)] [string]$RunName,
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$SaveInterval = 250,
    [int]$MaxRestarts = 5,
    [string]$ExtraArgs = "",
    [switch]$Worker   # internal: set when the scheduled task runs this script
)
$Log = "C:\isaac\logs\train_$RunName.log"                 # training output
$WatchLog = "C:\isaac\logs\train_$RunName.watchdog.log"   # watchdog decisions (separate: the training log is held open)
$StopFlag = "C:\isaac\logs\train_$RunName.stop"
$TaskName = "train_$RunName"
$LogRoot = "C:\isaac\Dancing-Robot\logs\rsl_rl"

function Write-Log($msg) { "== $(Get-Date -Format s) $msg" | Out-File -Encoding ascii -Append $WatchLog }

if (-not $Worker) {
    Remove-Item $StopFlag -ErrorAction SilentlyContinue
    $argList = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\isaac\start_training.ps1 " +
               "-Task $Task -RunName $RunName -NumEnvs $NumEnvs -MaxIterations $MaxIterations " +
               "-SaveInterval $SaveInterval -MaxRestarts $MaxRestarts -Worker"
    if ($ExtraArgs) { $argList += " -ExtraArgs `"$ExtraArgs`"" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argList
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3) -Priority 4 `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    "Started scheduled task $TaskName (target $MaxIterations iterations). Log: $Log"
    return
}

# Newest checkpoint across this run's folders created since the worker started.
function Get-LatestCheckpoint($since) {
    $runs = Get-ChildItem $LogRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory -Filter "*_$RunName" } |
        Where-Object { $_.CreationTime -ge $since }
    $best = $null
    foreach ($r in $runs) {
        foreach ($m in Get-ChildItem $r.FullName -Filter "model_*.pt") {
            $it = [int]($m.BaseName -replace "model_", "")
            if (-not $best -or $it -gt $best.Iter) { $best = [pscustomobject]@{ Iter = $it; Run = $r.Name; File = $m.Name } }
        }
    }
    return $best
}

. C:\isaac\isaac-env.ps1
Set-Location C:\isaac\Dancing-Robot
$workerStart = (Get-Date).AddSeconds(-5)
"== $(Get-Date -Format s) start: task=$Task run=$RunName envs=$NumEnvs target=$MaxIterations save_every=$SaveInterval extra=$ExtraArgs" |
    Out-File -Encoding ascii $WatchLog
"" | Out-File -Encoding ascii $Log

for ($attempt = 0; $attempt -le $MaxRestarts; $attempt++) {
    $latest = Get-LatestCheckpoint $workerStart
    if ($latest -and $latest.Iter -ge $MaxIterations - 1) { Write-Log "target reached (model_$($latest.Iter))"; break }
    if ($latest) {
        # rsl_rl resumed from iteration N with --max_iterations R saves its last checkpoint as model_(N+R-1),
        # so R = MaxIterations - N ends on model_(MaxIterations-1), the same final checkpoint as a fresh run.
        $remaining = $MaxIterations - $latest.Iter
        $resumeArgs = "--resume --load_run $($latest.Run) --checkpoint $($latest.File) --max_iterations $remaining"
        Write-Log "attempt $attempt`: resuming from $($latest.Run)\$($latest.File), $remaining iterations left"
    } else {
        $resumeArgs = "--max_iterations $MaxIterations"
        Write-Log "attempt $attempt`: starting fresh"
    }
    $cmd = "python third_party\booster_train\scripts\rsl_rl\train.py --task $Task --headless --num_envs $NumEnvs " +
           "--run_name $RunName $resumeArgs $ExtraArgs agent.save_interval=$SaveInterval >> `"$Log`" 2>&1"
    $proc = Start-Process cmd.exe -ArgumentList "/c $cmd" -WindowStyle Hidden -PassThru

    # Wait for python to appear under cmd.exe, then remove throttling from it.
    $py = $null
    for ($i = 0; $i -lt 60 -and -not $py; $i++) {
        $py = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id) AND Name='python.exe'"
        if (-not $py) { Start-Sleep -Seconds 2 }
    }
    if ($py) { & C:\isaac\no_throttle.ps1 -ProcessIds $py.ProcessId | Out-File -Encoding ascii -Append $WatchLog }
    else { Write-Log "WARNING: python process not found; throttling not disabled" }

    $proc.WaitForExit()
    Write-Log "attempt $attempt exited with code $($proc.ExitCode)"
    if (Test-Path $StopFlag) { Write-Log "stop flag found; not restarting"; break }
    $latest = Get-LatestCheckpoint $workerStart
    if ($latest -and $latest.Iter -ge $MaxIterations - 1) { Write-Log "target reached (model_$($latest.Iter))"; break }
    if ($attempt -lt $MaxRestarts) { Write-Log "training ended early; restarting in 60 s"; Start-Sleep -Seconds 60 }
    else { Write-Log "giving up after $MaxRestarts restarts" }
}
Write-Log "worker done"
