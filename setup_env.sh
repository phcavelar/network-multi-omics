#!/bin/bash
env_name='nemo'

platform_postfix=''
if [[ "$OSTYPE" == "darwin"* ]]; then
   platform_postfix='_mac'
fi

# Prefer mamba if possible
if command -v mamba; then
    install_method='mamba'
else
    install_method='conda'
fi

if command -v nvidia-smi; then
    if nvidia-smi | grep 'CUDA Version:'; then
        env_postfix='_gpu'
    else
        env_postfix='_cpu'
    fi
else
    env_postfix='_cpu'
fi

echo Using ${install_method} to install packages in ${env_name}${env_postfix}${platform_postfix}.yml

conda create -n ${env_name}
command ${install_method} env update -n ${env_name} --file yml/${env_name}${env_postfix}${platform_postfix}.yml
