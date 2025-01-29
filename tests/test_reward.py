from math import isclose

from test_conf import *


@pytest.mark.parametrize("backend", ["numpy"])
def test_NoReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="NoReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        assert np.all(transitions.rewards == 0)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetXVelocityReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(target_velocity=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetXVelocityReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 11.4912405)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.097849689424038)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetVelocityGoalReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="TargetVelocityGoalReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 11.678762435913086)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.077435396611691)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_LocomotionReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(joint_position_limit_coeff=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="LocomotionReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 0.371385663747787)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.)
    else:
        assert False

@pytest.mark.parametrize("backend", ["numpy"])
def test_TargetVelocityTrajReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetVelocityTrajReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 27.210592269897461)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.002138426993042)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_MimicReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="MimicReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 28.109945297241211)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.274038851261139)
    else:
        assert False
