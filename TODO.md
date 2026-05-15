# TODO — Next Steps

## 1. Train and release pretrained model weights

- Train one full run of the Kotliarov2020 RNA+CITE experiment in [experiments/kotliarov2020/02_train_scone.py](experiments/kotliarov2020/02_train_scone.py).
- Save and publish the model directory as [pretrained/kotliarov2020-scone-run0/](pretrained/kotliarov2020-scone-run0/).
- Verify the released folder includes: `metadata.json`, `loading_dict.json`, `model`, `mod_criteria_0`, `mod_criteria_1`, `contrastive_modules_0`, `contrastive_modules_1`, `opt`, `sched`, `history.json`.
- If model files exceed ~50 MB, host via GitHub Releases or Zenodo and add the link to [README.md](README.md).

## 2. Decide fate of archived CPU env file

- Compare [yml/archive/netemo_cpu.yml](yml/archive/netemo_cpu.yml) and [yml/nemo_cpu.yml](yml/nemo_cpu.yml).
- Remove [yml/archive/netemo_cpu.yml](yml/archive/netemo_cpu.yml) only if it is a true duplicate.

## 3. Optional API cleanup for model loading

- Update `SubsetContrastive.load(path, v)` in [nemo/models/subsetcontrastive.py](nemo/models/subsetcontrastive.py) so missing `v` defaults cleanly to version 1.
