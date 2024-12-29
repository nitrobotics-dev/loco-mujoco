# Installing the AMASS dependencies

To use the AMASS dataset, you need to download and set up the required data and models. This includes specifying the storage paths for the AMASS dataset and SMPL-H models and ensuring the dependencies are installed. Follow the instructions below to complete the setup.

---

## Additional dependencies

Currently, only editable installations are supported. Install the additional dependencies with:

```bash
pip install -e '.[smpl]'
```

---

## Download the AMASS Dataset and SMPL-H Model

### 1. Setting the Storage Directory
Before downloading the datasets and models, specify the directory where you’d like to store them:

```bash
loco-mujoco-set-amass-path --path "/path/to/amass/data"
loco-mujoco-set-smpl-model-path --path "/path/to/smpl/models"
```

### 2. Downloading the AMASS Dataset
1. Visit the [AMASS website](https://amass.is.tue.mpg.de/index.html).
2. Register and download the datasets you wish to use.
3. Select the `SMPL-H G` version of each dataset.
4. Extract and place them in the directory specified above, e.g., `/path/to/amass/data`.

### 3. Downloading the SMPL-H Models
1. Go to the [MANO website](https://mano.is.tue.mpg.de/download.php).
2. Register and download the following:
   - **Extended SMPL+H model** (includes the SMPL-H model w/o hands).
   - **Models & Code** (includes the hand models).
3. Extract and place them in the directory specified above, e.g., `/path/to/smpl/models`.

---

## Generating the SMPL-H Model

To combine the body and hand models into a single `SMPLH_NEUTRAL.PKL` file, follow one of the methods below:

---

### Automatic Method (Recommended; Requires Anaconda)

To resolve dependency issues with deprecated packages, we provide an automated [scipt](./install_smplh.sh). This script uses a Conda environment to manage dependencies and ensure proper installation.

1. Ensure [Anaconda](https://www.anaconda.com/) is installed.
2. Copy and paste the [scipt](./install_smplh.sh) somewhere.
3. Run the script:
   ```bash
   chmod u+x install_smplh.sh   # this is using the original name of the script
   ./install_smplh.sh
   ```

This script will:
- Create a Python 3.10 environment.
- Install all required dependencies.
- Generate the `SMPLH_NEUTRAL.PKL` file and place it in `/path/to/smpl/models`.
- Finally, deletes the created Python environment and dependencies. 

---

### Manual Method (Without Conda)

If you don’t have Conda installed or prefer a manual setup, follow these steps. **This method only supports Python 3.10.**

1. **Create a Virtual Environment**:
   ```bash
   python3.10 -m venv smplh_model_conversion_env
   source smplh_model_conversion_env/bin/activate  # On Windows: smplh_model_conversion_env\Scripts\activate
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install 'numpy<1.23.0' chumpy tqdm pyyaml
   ```

3. **Run the Script**:
   ```bash
   cd path/to/loco-mujoco
   python loco_mujoco/smpl/generate_script.py --smpl-conf-file /path/to/smpl/conf.yaml
   ```

4. **Deactivate the Environment**:
   ```bash
   deactivate
   ```
---


