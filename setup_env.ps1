$env_name = "nemo"

# Prefer mamba if available
if (Get-Command mamba -ErrorAction SilentlyContinue) {
    $install_method = "mamba"
} else {
    $install_method = "conda"
}

# Detect GPU via nvidia-smi
$env_postfix = "_cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = nvidia-smi 2>&1
    if ($smi -match "CUDA Version") {
        $env_postfix = "_gpu"
    }
}

$is_windows_host = $env:OS -eq "Windows_NT"

if ($is_windows_host) {
    ${env_file} = "yml/$env_name$env_postfix`_win.yml"
} else {
    ${env_file} = "yml/$env_name$env_postfix.yml"
}

if (-not (Test-Path $env_file)) {
    throw "Environment file not found: $env_file"
}

Write-Host "Using $install_method to install packages in $env_file"

conda create -n $env_name -y
& $install_method env update -n $env_name --file $env_file
