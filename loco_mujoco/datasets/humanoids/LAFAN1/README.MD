# LAFAN1 Dataset Installation Guide

The LAFAN1 dataset will be downloaded and cached automatically from [LocoMuJoCo's Hugging Face hub](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main). 


While caching is handleled by the hugging face automatically on your systems, the actual dataset
used in LocoMuJoCo requires further processing. To speed up data loading and processing, 
you can define a path to which the converted dataset will be saved. If this path is not set, 
the dataset will be converted online.

``` bash
loco-mujoco-set-conv-lafan1-path --path "/path/to/converted/LAFAN1_Retargeting_Dataset"
```


## That's It! 🎉
You're now ready to use the LAFAN1 Retargeting Dataset with LocoMuJoCo. Enjoy!