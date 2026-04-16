# NSight Systems Vulkan profiling for meganeura.
#
# Must be run from an ELEVATED (Administrator) PowerShell terminal.
# Vulkan API tracing and GPU metrics require admin privileges.
#
# Usage (from Admin PowerShell):
#   .\tools\nsys_profile.ps1 matmul_throughput
#   .\tools\nsys_profile.ps1 bench_smolvla_train -ExtraArgs "--profile"
#   .\tools\nsys_profile.ps1 profile_smollm2_decode
#
# Output: nsys_<name>.nsys-rep
# Open with NSight Systems UI (nsys-ui) or export stats with:
#   nsys stats nsys_<name>.nsys-rep

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Example,

    [string]$ExtraArgs = "",

    [switch]$SkipBuild
)

# Note: we don't set $ErrorActionPreference = "Stop" globally because
# cargo writes build progress to stderr, which PowerShell treats as errors.

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script must be run as Administrator for Vulkan GPU profiling.`nRight-click PowerShell -> Run as Administrator, then re-run."
    exit 1
}

$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\target-windows-x64\nsys.exe"
if (-not (Test-Path $nsys)) {
    Write-Error "nsys not found at $nsys`nInstall NSight Systems from https://developer.nvidia.com/nsight-systems"
    exit 1
}

# Register Vulkan layer if not already
$layerPath = "HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers"
$layerJson = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\target-windows-x64\vulkan-layers\VkLayer_nsight-sys_windows.json"
if (-not (Test-Path $layerPath)) {
    New-Item -Path $layerPath -Force | Out-Null
}
$existing = Get-ItemProperty -Path $layerPath -ErrorAction SilentlyContinue
if ($null -eq $existing -or $null -eq $existing.$layerJson) {
    New-ItemProperty -Path $layerPath -Name $layerJson -Value 0 -PropertyType DWORD -Force | Out-Null
    Write-Host "Registered nsys Vulkan layer in $layerPath"
}

# Build
if (-not $SkipBuild) {
    Write-Host "Building $Example..."
    & cargo build --release --example $Example
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }
}

$exe = "target\release\examples\$Example.exe"
if (-not (Test-Path $exe)) {
    Write-Error "$exe not found"
    exit 1
}

$output = "nsys_$Example"
Write-Host ""
Write-Host "Profiling with NSight Systems..."
Write-Host "  Vulkan API + individual GPU workloads"
Write-Host "  Output: $output.nsys-rep"
Write-Host ""

$args = @(
    "profile",
    "--trace=vulkan,wddm",
    "--vulkan-gpu-workload=individual",
    "--force-overwrite=true",
    "--output=$output",
    $exe
)
if ($ExtraArgs) {
    $args += $ExtraArgs.Split(" ")
}

& $nsys @args

Write-Host ""
Write-Host "=== Vulkan API summary ==="
& $nsys stats --report vulkan_api_sum "$output.nsys-rep" 2>$null

Write-Host ""
Write-Host "=== Vulkan GPU workload summary ==="
& $nsys stats --report vulkan_gpu_workload "$output.nsys-rep" 2>$null

Write-Host ""
Write-Host "Full report: $output.nsys-rep"
Write-Host "Open with:   nsys-ui $output.nsys-rep"
