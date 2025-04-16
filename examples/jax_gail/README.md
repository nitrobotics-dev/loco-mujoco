## GAIL Imitation Learning Example

This example demonstrates training a GAIL (Generative Adversarial Imitation Learning) agent on the Unitree H1 robot to mimic a dataset of human running. It serves as a minimal showcase of the GAIL algorithm and is **not** intended to produce state-of-the-art results.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the GAIL agent on the Unitree H1 robot for 75 million steps (approximately 7-10 minutes on an RTX 3080 Ti).
- Save the trained agent (as `GAILJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis (check the command line logs for details).

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```

> ⚠️ **Note:** Evaluating with MuJoCo may not yield results as robust as with Mjx due to simulator differences. For reliable policy transfer between the two, consider applying domain randomization techniques.
nks to the dataset, or more details about the environment or architecture!