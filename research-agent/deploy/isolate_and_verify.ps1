# ============================================================================
# isolate_and_verify.ps1 — Mandate 3 proof on Windows (WDDM)
#   1) driver version gate (>= 546.01 required for Sysmem Fallback control)
#   2) optional: pin llama-server.exe to the NVIDIA GPU (UserGpuPreferences)
#   3) monitors 'GPU Process Memory' Shared Usage / Dedicated Usage counters
#      while firing a real generation => proves ZERO shared-RAM spill
# Usage (admin not required for monitoring):
#   .\isolate_and_verify.ps1                                     # monitor + load test
#   .\isolate_and_verify.ps1 -PinGpuPreference "C:\tools\llama-server.exe"
#   .\isolate_and_verify.ps1 -ServerUrl "http://127.0.0.1:8080"
# ============================================================================
param(
  [string]$ServerUrl = "http://127.0.0.1:8080",
  [string]$ProcessName = "llama-server",
  [string]$PinGpuPreference = "",   # full path to exe to pin to High Performance GPU
  [int]$SampleSeconds = 25
)
$ErrorActionPreference = "Stop"

Write-Host "== Mandate 3 isolation audit (Windows/WDDM) =="

# ---- 1. driver gate ---------------------------------------------------------
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "[driver] $drv"
try {
  if ([version]$drv -lt [version]"546.01") {
    Write-Warning "Driver < 546.01: 'CUDA - Sysmem Fallback Policy' NOT available. Update driver first."
  }
} catch { Write-Host "[driver] non-standard version string; verify >= 546.01 manually." }

Write-Host @"
[manual step — REQUIRED, cannot be scripted safely]
  NVIDIA Control Panel -> Manage 3D Settings -> Global Settings (or a Program
  profile for llama-server.exe) -> 'CUDA - Sysmem Fallback Policy' ->
  'Prefer No Sysmem Fallback'.  (Automation alternative: NVIDIA Profile
  Inspector -> search 'Sysmem Fallback' -> 'Prefer No Sysmem Fallback'.)
  Without this, Windows silently spills VRAM into shared system RAM over
  PCIe (the 4-5 tok/s disease) instead of failing loudly.
"@

# ---- 2. optional GPU preference pin ----------------------------------------
if ($PinGpuPreference -ne "") {
  $k = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
  New-Item -Path $k -Force | Out-Null
  New-ItemProperty -Path $k -Name $PinGpuPreference -Value "GpuPreference=2;" -PropertyType String -Force | Out-Null
  Write-Host "[pin] High-Performance GPU preference set for $PinGpuPreference"
}

# ---- 3. counter helpers (English counter names; localized Windows: use IDs) --
function Get-GpuMemBytes([string]$kind) {
  $c = (Get-Counter "\GPU Process Memory(*)\$kind Usage" -ErrorAction SilentlyContinue)
  if (-not $c) { return 0 }
  ($c.CounterSamples | Where-Object { $_.InstanceName -match "pid_$($procId)_|_luid_" } |
     Measure-Object -Property CookedValue -Sum).Sum
}
$proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc) { Write-Warning "$ProcessName not running — starting nothing; totals will include all GPU processes." ; $procId = 0 }
else { $procId = $proc.Id; Write-Host "[proc] $ProcessName pid=$procId" }

$sharedBase  = Get-GpuMemBytes "Shared"
$dedicBase   = Get-GpuMemBytes "Dedicated"
Write-Host ("[base] shared={0:N0} B dedicated={1:N0} B" -f $sharedBase, $dedicBase)

# ---- 4. fire load + sample ---------------------------------------------------
$job = Start-Job -ScriptBlock {
  param($u)
  $body = @{ prompt = "Explain in detail how POSIX bash handles file descriptor inheritance across pipelines, with exact syntax examples."; n_predict = 384; temperature = 1.0; top_p = 0.95; top_k = 64; cache_prompt = $false } | ConvertTo-Json
  Invoke-RestMethod -Uri "$u/completion" -Method Post -ContentType "application/json" -Body $body | Out-Null
} -ArgumentList $ServerUrl

$sharedPeak = $sharedBase; $dedicPeak = $dedicBase
for ($i = 0; $i -lt $SampleSeconds; $i++) {
  Start-Sleep -Seconds 1
  $s = Get-GpuMemBytes "Shared"; $d = Get-GpuMemBytes "Dedicated"
  if ($s -gt $sharedPeak) { $sharedPeak = $s }
  if ($d -gt $dedicPeak)  { $dedicPeak  = $d }
}
Wait-Job $job -Timeout 120 | Out-Null; Remove-Job $job -Force

$sharedDelta = $sharedPeak - $sharedBase
Write-Host ("[peak] shared={0:N0} B (+{1:N0}) dedicated={2:N0} B" -f $sharedPeak, $sharedDelta, $dedicPeak)

# ---- 5. verdict ---------------------------------------------------------------
$tolerance = 1MB
if ($sharedDelta -le $tolerance -and $dedicPeak -le 7.9GB) {
  Write-Host "RESULT ISOLATION PASS shared_delta=$([math]::Round($sharedDelta/1MB,2))MiB dedicated_peak=$([math]::Round($dedicPeak/1GB,2))GiB"
} else {
  Write-Host "RESULT ISOLATION FAIL — shared memory moved (+$([math]::Round($sharedDelta/1MB,2)) MiB) or dedicated over budget."
  Write-Host "  => Set 'Prefer No Sysmem Fallback' (step above), close GPU apps, or lower FIT_MODE."
}
Write-Host @"

== Negative test (proves enforcement is live) ==
  Start the server with an impossible config, e.g.:
    llama-server.exe -m <model> -ngl 99 -fa on -c 131072 --cache-type-k q8_0 --cache-type-v q8_0
  EXPECTED: 'cudaMalloc failed: out of memory' at load (loud failure).
  WRONG   : it starts and crawls at 4-5 tok/s => Sysmem Fallback is still ENABLED.
"@
