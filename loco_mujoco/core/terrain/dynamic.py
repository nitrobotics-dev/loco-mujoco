from loco_mujoco.core.terrain import Terrain


class DynamicTerrain(Terrain):

    def is_dynamic(self):
        return True
