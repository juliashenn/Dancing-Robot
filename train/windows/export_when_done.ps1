# Wait for training runs to finish, then export each one's final checkpoint (play.py writes exported\*.pt/.onnx).
# Runs as a hidden scheduled task so it survives SSH disconnects; works for runs that are already queued.
#
#   powershell -File C:\isaac\export_when_done.ps1 -Runs lock1n,pop1n,hiphop1n,sjazz1n,krump1n,pop2n
#
# For each run it waits until C:\isaac\logs\train_<run>.watchdog.log says "worker done", reads the task from
# that log, copies the newest model_<N>.pt of the newest *_<run> folder to C:\isaac\export\<run>_<N>\ and runs
#   python train\run_rsl_rl.py play --task <task>-Play --num_envs 1 --headless --checkpoint <copy>
# Log: C:\isaac\logs\export_when_done.log
param(
    [Parameter(Mandatory)] [string[]]$Runs,
    [switch]$Worker
)
$Log = "C:\isaac\logs\export_when_done.log"
function Write-Log($msg) { "== $(Get-Date -Format s) $msg" | Out-File -Encoding ascii -Append $Log }

if (-not $Worker) {
    $argList = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\isaac\export_when_done.ps1 " +
               "-Runs $($Runs -join ',') -Worker"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argList
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3) -Priority 4 `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName "export_when_done" -Action $action -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName "export_when_done"
    "Started scheduled task export_when_done for: $($Runs -join ', '). Log: $Log"
    return
}

if ($Runs.Count -eq 1) { $Runs = $Runs[0] -split "," }
. C:\isaac\isaac-env.ps1
Set-Location C:\isaac\Dancing-Robot
Write-Log "waiting for: $($Runs -join ', ')"
foreach ($run in $Runs) {
    $watch = "C:\isaac\logs\train_$run.watchdog.log"
    while (-not ((Test-Path $watch) -and (Select-String -Path $watch -Pattern "worker done" -Quiet))) { Start-Sleep -Seconds 120 }
    $task = ((Select-String -Path $watch -Pattern "task=(\S+)" | Select-Object -First 1).Matches[0].Groups[1].Value)
    $folder = Get-ChildItem C:\isaac\Dancing-Robot\logs\rsl_rl\*\ -Directory -Filter "*_$run" | Sort-Object CreationTime | Select-Object -Last 1
    $best = Get-ChildItem $folder.FullName -Filter model_*.pt |
        Sort-Object { [int]($_.BaseName -replace "model_", "") } | Select-Object -Last 1
    $iter = [int]($best.BaseName -replace "model_", "")
    $dir = "C:\isaac\export\${run}_$iter"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item $best.FullName $dir -Force
    Write-Log "$run finished: exporting $($folder.Name)\$($best.Name) with $task-Play"
    cmd /c "python train\run_rsl_rl.py play --task $task-Play --num_envs 1 --headless --checkpoint $dir\$($best.Name) > C:\isaac\logs\export_${run}_$iter.log 2>&1"
    $files = (Get-ChildItem "$dir\exported" -ErrorAction SilentlyContinue).Name -join ", "
    Write-Log "$run export exit $LASTEXITCODE -> $dir\exported: $files"
}
Write-Log "all exports done"
