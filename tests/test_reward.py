from test_conf import *


@pytest.mark.parametrize("backend", ["numpy"])
def test_NoReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             reward_type="NoReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetXVelocityReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(target_velocity=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             reward_type="TargetXVelocityReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetVelocityGoalReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="TargetVelocityGoalReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_LocomotionReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="LocomotionReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetVelocityTrajReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             reward_type="TargetVelocityTrajReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_MimicReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             reward_type="MimicReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 999
        assert np.all(transitions.rewards == 0)
    else:
        assert False
