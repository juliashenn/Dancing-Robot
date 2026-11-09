# Opt the given processes out of Windows power throttling (EcoQoS / efficiency mode) and set Normal priority.
param([int[]]$ProcessIds)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PowerThrottle {
    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_POWER_THROTTLING_STATE { public uint Version; public uint ControlMask; public uint StateMask; }
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetProcessInformation(IntPtr h, int infoClass, ref PROCESS_POWER_THROTTLING_STATE info, int size);
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr h);
    public static bool Disable(int pid) {
        IntPtr h = OpenProcess(0x0200 /* PROCESS_SET_INFORMATION */, false, pid);
        if (h == IntPtr.Zero) return false;
        // ProcessPowerThrottling = 4; EXECUTION_SPEED (0x1) | IGNORE_TIMER_RESOLUTION (0x4) controlled, state 0 = never throttle
        var s = new PROCESS_POWER_THROTTLING_STATE { Version = 1, ControlMask = 0x5, StateMask = 0 };
        bool ok = SetProcessInformation(h, 4, ref s, Marshal.SizeOf(typeof(PROCESS_POWER_THROTTLING_STATE)));
        CloseHandle(h);
        return ok;
    }
}
"@
foreach ($id in $ProcessIds) {
    $p = Get-Process -Id $id
    $p.PriorityClass = "Normal"
    "{0} ({1}): throttling disabled = {2}, priority = {3}" -f $id, $p.ProcessName, [PowerThrottle]::Disable($id), $p.PriorityClass
}
