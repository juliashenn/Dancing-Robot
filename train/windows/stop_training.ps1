# Stop a run started with start_training.ps1, on purpose (the watchdog will not restart it).
#   powershell -File C:\isaac\stop_training.ps1 -RunName <name>
# Checkpoints already saved are kept.
param([Parameter(Mandatory)] [string]$RunName)
$StopFlag = "C:\isaac\logs\train_$RunName.stop"
"stopped $(Get-Date -Format s)" | Out-File -Encoding ascii $StopFlag
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "--run_name $RunName( |$)" }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force; "killed python $($p.ProcessId)" }
Start-Sleep -Seconds 3
Stop-ScheduledTask -TaskName "train_$RunName" -ErrorAction SilentlyContinue
"stopped train_$RunName"
