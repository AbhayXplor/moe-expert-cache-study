# ============================================================================
# set_sysmem_fallback.ps1 — programmatically enforce the NVIDIA driver policy
#   "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback"
# This is the driver-level kill switch for silent VRAM -> Shared-System-RAM
# spill (the 4-5 tok/s disease). No manual Control Panel clicking required.
#
# Mechanism (verified against working community automation):
#   Driver setting ID 283962569 (0x10ECECC9), Dword:
#     1 = Prefer No Sysmem Fallback (spill DISABLED -> loud CUDA OOM)
#     0 = Driver Default            (spill ALLOWED  -> silent slow path)
#   Applied through NVIDIA's DRS database via nvidiaProfileInspector, which
#   accepts a .nip XML profile as a command-line argument (NVAPI DRS wrapper).
#
# Usage (Administrator PowerShell):
#   .\set_sysmem_fallback.ps1                              # global (Base Profile)
#   .\set_sysmem_fallback.ps1 -TargetExe llama-server.exe  # scope to one app
#   .\set_sysmem_fallback.ps1 -Policy DriverDefault        # revert
#   .\set_sysmem_fallback.ps1 -InspectorExe C:\tools\nvidiaProfileInspector.exe
#
# Requires: Windows + NVIDIA driver >= 546.01 (NVIDIA KB 5490). No reboot;
#           restart the target application afterwards.
# ============================================================================
param(
  [ValidateSet("PreferNoFallback", "DriverDefault")]
  [string]$Policy = "PreferNoFallback",
  [string]$TargetExe   = "",     # e.g. "llama-server.exe"; empty = global Base Profile
  [string]$InspectorExe = "",    # explicit path; empty = auto-download latest release
  [string]$WorkDir      = "$env:TEMP\nv-drs-automation"
)
$ErrorActionPreference = "Stop"

# ---- elevation (DRS is machine-wide) ---------------------------------------
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
      ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "[*] Relaunching elevated ..."
  Start-Process powershell.exe -Verb RunAs -ArgumentList `
    ("-NoProfile -ExecutionPolicy Bypass -File `"{0}`" -Policy {1} -TargetExe `"{2}`"" -f $PSCommandPath, $Policy, $TargetExe)
  exit
}

# ---- driver version gate ----------------------------------------------------
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "[*] NVIDIA driver: $drv"
try {
  if ([version]$drv -lt [version]"546.01") {
    Write-Error "Driver < 546.01 does not expose the Sysmem Fallback Policy. Update the driver first."
  }
} catch { Write-Warning "Non-standard driver string '$drv' — verify manually that it is >= 546.01." }

# ---- obtain nvidiaProfileInspector ------------------------------------------
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
if ($InspectorExe -eq "") {
  $InspectorExe = Join-Path $WorkDir "nvidiaProfileInspector.exe"
  if (-not (Test-Path $InspectorExe)) {
    $zip = Join-Path $WorkDir "nvidiaProfileInspector.zip"
    Write-Host "[*] Downloading nvidiaProfileInspector (Orbmu2k, open source) ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest "https://github.com/Orbmu2k/nvidiaProfileInspector/releases/latest/download/nvidiaProfileInspector.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $WorkDir -Force
    $found = Get-ChildItem $WorkDir -Recurse -Filter "nvidiaProfileInspector.exe" | Select-Object -First 1
    if (-not $found) { Write-Error "inspector exe not found in archive" }
    Move-Item $found.FullName $InspectorExe -Force
  }
}
Unblock-File $InspectorExe -ErrorAction SilentlyContinue
# DRS blobs must be writable for import:
Get-ChildItem "$env:ProgramData\NVIDIA Corporation\Drs" -Filter "nvdrsdb*.bin" -ErrorAction SilentlyContinue |
  ForEach-Object { $_.IsReadOnly = $false }

# ---- build the .nip profile XML ----------------------------------------------
$value = if ($Policy -eq "PreferNoFallback") { 1 } else { 0 }
$profileName = if ($TargetExe -ne "") { "LLM VRAM Isolation ($TargetExe)" } else { "Base Profile" }
$executables = if ($TargetExe -ne "") { "<Executeables>$TargetExe</Executeables>" } else { "<Executeables />" }

$nip = @"
<?xml version="1.0" encoding="utf-16"?>
<ArrayOfProfile>
  <Profile>
    <ProfileName>$profileName</ProfileName>
    $executables
    <Settings>
      <ProfileSetting>
        <SettingNameInfo>CUDA Sysmem Fallback Policy</SettingNameInfo>
        <SettingID>283962569</SettingID>
        <SettingValue>$value</SettingValue>
        <ValueType>Dword</ValueType>
      </ProfileSetting>
    </Settings>
  </Profile>
</ArrayOfProfile>
"@
$nipPath = Join-Path $WorkDir "sysmem_fallback.nip"
Set-Content -Path $nipPath -Value $nip -Encoding Unicode -Force   # must be UTF-16
Write-Host "[*] Applying policy '$Policy' (SettingID 283962569 = $value) to profile '$profileName' ..."

# ---- import via inspector CLI (proven pattern) --------------------------------
Start-Process -Wait -FilePath $InspectorExe -ArgumentList "`"$nipPath`""
Write-Host "[*] Import finished. Restart any running CUDA applications (llama-server)."

# ---- verification ---------------------------------------------------------------
Write-Host @"

== VERIFY (do both) ==
1. Positive: run  .\isolate_and_verify.ps1   -> 'Shared Usage' delta must be 0 during generation.
2. Negative: start an intentionally oversized load, e.g.
     llama-server.exe -m <model.gguf> -ngl 99 -fa on -c 131072 --cache-type-k q8_0 --cache-type-v q8_0
   EXPECTED : 'cudaMalloc failed: out of memory'  (policy ACTIVE)
   WRONG    : it starts and crawls at 4-5 tok/s   (policy NOT applied)

Notes:
- Per-app profile (-TargetExe) is recommended on gaming machines; global Base
  Profile affects every CUDA process.
- WSL2 inherits this host-side policy (its CUDA rides host WDDM).
- Setting persists across driver upgrades; re-run the negative test after
  major driver branch changes.
"@
