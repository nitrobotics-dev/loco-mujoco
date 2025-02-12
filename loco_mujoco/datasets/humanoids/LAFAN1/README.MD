# LAFAN1 Dataset Installation Guide

The LAFAN1 dataset will be downloaded and cached automatically from the Hugging Face hub. 
For the UnitreeH1, UnitreeG1 and UnitreeH1v2 environments, we use the converted datasets by 
Unitree (https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/)

While caching is handleled by the hugging face automatically on your systems, the actual dataset
using in LocoMuJoCo require further processing. To speed up data loading and processing, 
you can define a path to which the converted dataset will be saved. If this path is not set, 
the dataset will be converted online.

``` bash
loco-mujoco-set-conv-lafan1-path --path "/path/to/converted/LAFAN1_Retargeting_Dataset"
```


## That's It! 🎉
You're now ready to use the LAFAN1 Retargeting Dataset with LocoMuJoCo. Enjoy!