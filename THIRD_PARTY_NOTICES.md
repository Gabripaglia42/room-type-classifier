# Third-party notices

## cubicasa_model.py

The network definition in `cubicasa_model.py` is copied from the CubiCasa5K repository
(<https://github.com/CubiCasa/CubiCasa5k>, file `floortrans/models/hg_furukawa_original.py`), with the
pretraining-weights loader removed. It is **not** covered by this repository's Apache 2.0 licence. It remains
under the original licence:

> CubiCasa5K is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License
> (CC BY-NC 4.0). Copyright (c) 2019.

Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., Kannala, J. *CubiCasa5K: A Dataset and an Improved
Multi-Task Model for Floorplan Image Analysis.* Scandinavian Conference on Image Analysis (SCIA), 2019.

## Data and weights (not included)

| Resource | Licence | How it is used |
|---|---|---|
| CubiCasa5K dataset (<https://zenodo.org/records/2613548>) | CC BY-NC 4.0 | Downloaded by the user; training data |
| CubiCasa5K pretrained weights | CC BY-NC 4.0 | Downloaded by the user; baseline and room outlines |
| torchvision ResNet-18 ImageNet weights | BSD-3-Clause (torchvision) | Downloaded automatically by torchvision |

Every model trained with this code on CubiCasa5K inherits the non-commercial restriction.
