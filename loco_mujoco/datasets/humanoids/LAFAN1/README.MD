# LAFAN1 Dataset Installation Guide

This guide helps you install and set up the **LAFAN1 Retargeting Dataset**, designed for Unitree robots (H1, G1, and H1_2).

---

## Step 1: Clone the Dataset Repository

Navigate to the directory where you'd like to store the dataset, and clone the repository from the Hugging Face Hub:

```bash
git clone https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset
```

## Step 2: Configure the Dataset Path
To ensure LocoMuJoCo can locate the dataset, configure the dataset path using the following command:

``` bash
loco-mujoco-set-lafan1-path --path "/path/to/LAFAN1_Retargeting_Dataset"
```

Replace /path/to/LAFAN1_Retargeting_Dataset with the actual path to the cloned dataset on your system.

## That's It! 🎉
You're now ready to use the LAFAN1 Retargeting Dataset with LocoMuJoCo. Enjoy!