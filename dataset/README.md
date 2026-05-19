# Dataset Links

Large hyperspectral datasets should not be committed to this repository.

Put downloaded datasets outside Git tracking, for example:

```text
/data2/lzj/lab/Mamba_test/dataset
```

This directory keeps only metadata and download/source links. Use `datasets.yaml` to record where each dataset can be obtained, expected files, and any split protocol notes.

Local data files such as `.mat`, `.tif`, `.hdr`, `.img`, `.npy`, and archives are ignored by `.gitignore`.

