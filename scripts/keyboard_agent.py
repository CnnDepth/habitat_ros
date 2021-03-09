import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import keyboard

class KeyboardAgent(habitat.Agent):
    def __init__(self, 
                 save_observations=True,
                 rgb_topic='/habitat/rgb/image',
                 depth_topic='/habitat/depth/image',
                 camera_info_topic='/habitat/rgb/camera_info',
                 path_topic='/true_path',
                 pose_topic='/true_pose',
                 odometry_topic='/true_odom',
                 publish_odom=False):
        self.speed = 0.
        self.twist = 1.
        self.time_of_publish = 0

    def reset(self):
        pass

    def get_actions_from_keyboard(self):
        keyboard_commands = [HabitatSimActions.MOVE_FORWARD] * int(self.speed)
        if keyboard.is_pressed('left'):
            keyboard_commands += [HabitatSimActions.TURN_LEFT] * max(int(self.twist), 1)
        if keyboard.is_pressed('right'):
            keyboard_commands += [HabitatSimActions.TURN_RIGHT] * max(int(self.twist), 1)
        if keyboard.is_pressed('up'):
            self.speed += 0.1
        if keyboard.is_pressed('down'):
            self.speed = max(self.speed - 0.2, 0)
        if keyboard.is_pressed('s'):
            self.speed = 0
        if keyboard.is_pressed('e'):
            self.twist += 0.2
        if keyboard.is_pressed('d'):
            self.twist = max(self.twist - 0.2, 0)
        return keyboard_commands

    def act(self, observations, env):
        # receive command from keyboard and move
        actions = self.get_actions_from_keyboard()
        if len(actions) > 0:
            for action in actions[:-1]:
                env.step(action)
        if len(actions) > 0:
            return actions[-1]
        else:
            return HabitatSimActions.STOP