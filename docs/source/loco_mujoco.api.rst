API Documentation
====================

Core
-----------

The core consists of fundamental modules that serve as the backbone for all tasks and environments.
It supports both reinforcement learning and imitation learning, providing essential components such
as the main simulation in MuJoCo and Mjx, observation and goal interfaces, initial and terminal state
handlers, control and reward functions, as well as domain and terrain interfaces.

.. toctree::
    :hidden:

    ./loco_mujoco.core.rst


Environments
-----------

LocoMuJoCo comes with 16 different environments, including 12 humanoids and 4 quadrupeds. The environments
are designed to be simple and user-friendly, offering a variety of tasks for both imitation and reinforcement
learning. The environments are built on top of the core modules, providing a seamless integration with the
rest of the library.


.. toctree::
    :hidden:

    ./loco_mujoco.environments.humanoids.rst
    ./loco_mujoco.environments.quadrupeds.rst
    ./loco_mujoco.environments.base.rst



Datasets
-----------

LocoMuJoCo provides different datasets for imitation learning. For now, three different dataset sources are available:

- **DefaultDataset**: Collected by the maintainers of LocoMuJoCo.
- `Lafan1Dataset <https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset>`__ : Humanoid motion capture dataset.
- `AMASSDataset <https://amass.is.tue.mpg.de/>`__ : Humanoid motion capture dataset.

All datasets will be automatically downloaded, retagerget for the respective humanoid and optionaly cached.
Due to licensing, the AMASS dataset needs to be downloaded manually. You can find more information on the installation
page of the documentation. The DefaultDataset and Lafan1Dataset are hosted on
`LocoMuJoCo's HuggingFace hub <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main>`__.


Task Factories
-----------

For easy dataset loading, LocoMuJoCo provides task factories that can be used to load the datasets and create the
corresponding tasks. Different sources of datasets can be mixed and matched to create a single task. Here is an example
of an imitation learning task using the DefaultDataset and Lafan1Dataset.

.. toctree::
    :hidden:

    ./loco_mujoco.task_factories.rst


.. literalinclude:: ../../examples/replay_datasets/example.py
    :language: python





