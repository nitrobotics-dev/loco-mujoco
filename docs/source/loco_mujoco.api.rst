API Documentation
====================

Core
-----------

.. toctree::
    :hidden:

    ./loco_mujoco.core.rst


Environments
-----------

.. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/73ca0cdd-3958-4d59-a1f7-0eba00fe373a


LocoMuJoCo focuses on *Loco*motion environments. This includes humanoids and quadrupeds, with
a strong focus on the latter. The environment is built on top of MuJoCo. The aim of LocoMuJoCo is to be
a simple and easy-to-use environment for imitation learning and reinforcement learning while shifting the focus towards
realistic and complex tasks crucial for real-world robotics. LocoMuJoCo strives to be simple and user-friendly, offering
an environment tailored for both imitation and reinforcement learning. Its main objective is to shift the focus away from
simplistic locomotion tasks often used as benchmarks for imitation and reinforcement learning algorithms, and instead
prioritize realistic and intricate tasks vital for real-world robotics applications.

For imitation learning, it is crucial to have a good and diverse datasets. LocoMuJoCo makes it very simple to generate
diverse datasets of different difficulty levels in a single line of code. This allows the user to focus on the learning
algorithm and not worry about the environment. Here is a simple example of how to generate a the environment and the dataset
for the Unitree H1 robot:

.. literalinclude:: ../../examples/simple_gymnasium_env/example_unitree_h1.py
    :language: python


.. note:: As can be seen in the example above *Task-IDs* (e.g., "UnitreeH1.run.real") are used to choose what environment,
    task and dataset type to use. The general structure of a Task-Id is `<environment>.<task>.<dataset_type>`.
    For the Task-ID, you have to choose *at least* the environment name. Missing
    information will be filled with default settings, which are "walk" for the task and "real" for the dataset type for the
    Unitree H1 robot. A list of all available *Task-IDs* in LocoMuJoCo is given in the :doc:`./loco_mujoco.environments`.
    Alternatively, you can use the following code:

    .. code-block:: python

        from loco_mujoco import LocoEnv

        task_ids = LocoEnv.get_all_task_names()




.. toctree::
    :hidden:

    ./loco_mujoco.environments.humanoids.rst
    ./loco_mujoco.environments.quadrupeds.rst
    ./loco_mujoco.environments.base.rst







