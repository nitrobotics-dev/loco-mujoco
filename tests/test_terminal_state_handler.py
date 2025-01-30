from test_conf import *


@pytest.mark.parametrize("backend", ["numpy"])
def test_HeightBasedTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="HeightBasedTerminalStateHandler")

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 922
        assert np.all(transitions.absorbings[0:77] == 0)
        assert np.all(transitions.absorbings[77:] == 1)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_NoTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="NoTerminalStateHandler")

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 0
        assert np.all(transitions.absorbings == 0)
    else:
        assert False


@pytest.mark.parametrize("backend", ["numpy"])
def test_RootPoseTrajTerminalStateHandler(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory


    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="RootPoseTrajTerminalStateHandler")

    if backend == "numpy":
        assert len(transitions.absorbings) == 999
        assert np.sum(transitions.absorbings) == 958
        assert np.all(transitions.absorbings[0:41] == 0)
        assert np.all(transitions.absorbings[41:] == 1)
    else:
        assert False