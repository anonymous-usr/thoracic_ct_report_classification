# Data

This directory contains the external test set consisting of 1,000 English radiology reports from the [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) manually labeled according to our reference standard.

## File Format

`external_test_set.csv` contains the following columns:

| Column | Description |
|--------|-------------|
| `VolumeName` | CT-RATE volume identifier (e.g., `train_6_b_1.nii.gz`) |
| `Label` | Reference standard label |
| `Findings_EN` | Findings section of the radiology report |
| `Impressions_EN` | Impressions section of the radiology report |

## Label Definitions

Labels were manually assigned based on the presence and location of current lung abnormalities:

| Label | Name | Definition |
|-------|------|------------|
| 0 | Normal | Neither findings nor impressions describe a current lung-related abnormality |
| 1 | Findings-level abnormal | A current lung abnormality is described in findings but not in impressions |
| 2 | Impressions-level abnormal | A current lung abnormality is described in the impressions |

**Lung structures include:** parenchyma, pleura, bronchi, and pulmonary vessels.

## Citation

If you use this data, please cite the CT-RATE dataset:

```bibtex
@article{hamamci2024generalist,
  title     = {Generalist foundation models from a multimodal dataset for 
               3D computed tomography},
  author    = {Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and 
               Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and 
               Dogan, Irem and Durugol, Omer Faruk and Hou, Benjamin and 
               Shit, Suprosanna and others},
  journal   = {Nature Biomedical Engineering},
  pages     = {1--19},
  year      = {2026},
  publisher = {Nature Publishing Group UK London}
}
```

## License

This data is released under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), consistent with the CT-RATE dataset license. 
