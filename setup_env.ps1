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

Write-Host "Using $install_method to install packages in $env_name$env_postfix.yml"

conda create -n $env_name
& $install_method env update -n $env_name --file "yml/$env_name$env_postfix.yml"
