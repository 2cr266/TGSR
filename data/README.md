# ImageNet-Test-100

This directory records the fixed 100-image subset used for TGSR's ImageNet-Test evaluation.

- `imagenet_test_100.txt`: exact selected filenames
- `imagenet_test_100.json`: filenames plus the reproducibility metadata

The 3,000 HR candidate filenames are sorted before 100 indices are sampled without replacement with `numpy.random.RandomState(0)`. No manual or content-based filtering is applied.

```bash
python scripts/select_imagenet_test_100.py /path/to/imagenet512.zip
```

The source can be either the released ZIP archive or its extracted directory.
