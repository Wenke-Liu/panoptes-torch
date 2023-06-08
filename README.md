# Panoptes model in PyTorch 2.0

Under construction. CallBacks need work.

Model abbreviations:
- X1: Panoptes2, InceptionResNetV2 X 3, without feature pooling
- X3: Panoptes4, InceptionResNetV2 X 3, with feature pooling
- F{1, 3}: X{1, 3} with covariates (eg., Age, BMI)

Usage:
`python main.py --config=config\.yaml`
Modify the input configure file to instantiate the desired Panoptes variant.

See [examples](src/config.yaml) for more details.