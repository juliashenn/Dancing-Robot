# Start a booster_train run as a Windows scheduled task, so it survives SSH disconnects.
#
#   powershell -File C:\isaac\start_training.ps1 -Task Booster-K1-MJ_Dance_002-v0 -RunName mj2_full
#   powershell -File C:\isaac\start_training.ps1 -Task <task> -RunName <name> -ExtraArgs "--max_iterations 8000"
#
# Log: C:\isaac\logs\train_<RunName>.log   Stop: Stop-ScheduledTask -TaskName train_<RunName>
#
# Scheduled tasks start at BelowNormal priority, and Windows 11 then power-throttles the process
# (efficiency mode), which made training ~1.9x slower. The task here runs at normal priority and
# the training process is opted out of power throttling once it starts.
param(
    [Parameter(Mandatory)] [string]$Task,
    [Parameter(Mandatory)] [string]$RunName,
    [int]$NumEnvs = 4096,
    [string]$ExtraArgs = "",
    [switch]$Worker   # internal: set when the scheduled task runs this script
)
$Log = "C:\isaac\logs\train_$RunName.log"
$TaskName = "train_$RunName"

if (-not $Worker) {
    $argList = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File C:\isaac\start_training.ps1 " +
               "-Task $Task -RunName $RunName -NumEnvs $NumEnvs -Worker"
    if ($ExtraArgs) { $argList += " -ExtraArgs `"$ExtraArgs`"" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argList
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 48) -Priority 4 `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    "Started scheduled task $TaskName. Log: $Log"
    return
}

. C:\isaac\isaac-env.ps1
Set-Location C:\isaac\Dancing-Robot
"== start $(Get-Date -Format s): task=$Task run=$RunName envs=$NumEnvs extra=$ExtraArgs" | Out-File -Encoding ascii $Log
$cmd = "python third_party\booster_train\scripts\rsl_rl\train.py --task $Task --headless --num_envs $NumEnvs --run_name $RunName $ExtraArgs >> `"$Log`" 2>&1"
$proc = Start-Process cmd.exe -ArgumentList "/c $cmd" -WindowStyle Minimized -PassThru

# Wait for python to appear under cmd.exe, then remove throttling from it.
for ($i = 0; $i -lt 60; $i++) {
    $py = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id) AND Name='python.exe'"
    if ($py) { break }
    Start-Sleep -Seconds 2
}
if ($py) {
    & C:\isaac\no_throttle.ps1 -ProcessIds $py.ProcessId | Out-File -Encoding ascii -Append $Log
} else {
    "WARNING: python process not found; throttling not disabled" | Out-File -Encoding ascii -Append $Log
}
$proc.WaitForExit()
"== exit code $($proc.ExitCode) at $(Get-Date -Format s)" | Out-File -Encoding ascii -Append $Log
