# Run a command in the logged-in desktop session (via a one-off scheduled task) and wait for it.
# Needed for anything that renders (play.py --video, csv_to_npz.py): SSH sessions have no access to the
# interactive desktop, and Isaac Sim's renderer crashes there (access violation in __enable_hydra_engine).
#
#   powershell -File C:\isaac\run_interactive.ps1 -Name play_mj2 -Command "python third_party\...\play.py ..."
#
# The command runs in C:\isaac\Dancing-Robot with the Isaac env loaded. Output: C:\isaac\logs\<Name>.log
param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [string]$Command,
    [int]$TimeoutMinutes = 60,
    [switch]$Worker
)
$Log = "C:\isaac\logs\$Name.log"
$Done = "C:\isaac\logs\$Name.done"

if ($Worker) {
    . C:\isaac\isaac-env.ps1
    Set-Location C:\isaac\Dancing-Robot
    $proc = Start-Process cmd.exe -ArgumentList "/c $Command > `"$Log`" 2>&1" -WindowStyle Hidden -PassThru
    $proc.WaitForExit()
    "$($proc.ExitCode)" | Out-File -Encoding ascii $Done
    return
}

Remove-Item $Done, $Log -ErrorAction SilentlyContinue
$encoded = $Command.Replace('"', '\"')
$argList = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\isaac\run_interactive.ps1 -Name $Name -Worker -Command `"$encoded`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argList
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes $TimeoutMinutes) -Priority 4
Register-ScheduledTask -TaskName "run_$Name" -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "run_$Name"

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while (-not (Test-Path $Done) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
Unregister-ScheduledTask -TaskName "run_$Name" -Confirm:$false
if (Test-Path $Done) { "exit code: " + (Get-Content $Done) } else { "timed out after $TimeoutMinutes min" }
"log: $Log"
