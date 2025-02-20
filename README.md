# BODEX baseline: Frogger

### BODex: [Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

### Frogger: [Website](https://alberthli.github.io/frogger/) [Paper](https://arxiv.org/abs/2302.13687)

## Environment
Follow the instructions in the README_official.md file to install the required packages.

## Generate the dataset

1. Link the dataset to the assets folder
ln -s ${YOUR_PATH}/MeshProcess/assets/object assets/object

1. Generate the dataset
```
cd scripts
python timing.py # generate grasps for single object
python multi_plan.py        # generate grasps for multiple objects
## Attention: the timing.py may hang, when runing multi_plan.py, please restart the multi_plan.py every 2 hours
```