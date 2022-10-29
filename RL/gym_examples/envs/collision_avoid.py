import gym
from gym import spaces
import pygame
import numpy as np


class CollisionAvoid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4, obstacle_num=0,mode='train'):
        self.size = size  # The size of the square grid
        self.window_size = 640  # The size of the PyGame window
        self.obstacle_num = obstacle_num    #number of obstacles
        self._target_location = [1, 1]
        self._mode = mode

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        self.observation_space = spaces.Box(0,size,shape=(2,),dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location.astype(np.float32)
        # return np.array([self._agent_location]).astype(np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self,options=None):
        # We need the following line to seed self.np_random
        # super().reset()
        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._agent_location = np.random.random_integers(0, self.size, size=2)
        # We will sample the target's location randomly until it does not coincide with the agent's location

        # obstacle location
        self._obstacle_location = []
        for _ in range(0,self.obstacle_num):
            self._obstacle_location.append(np.random.random_integers(0, self.size, size=2))
        self._obstacle_radius = self.window_size/self.size/2


        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = np.random.random_integers(
        #         0, self.size, size=2
        #     )

        # define distance
        self.pre_dis = np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self._mode == 'test':
            self._render_frame()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = int(action)
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target

        # terminated = np.array_equal(self._agent_location, self._target_location)
        # if terminated: reward = reward + 100
        # reward = reward + 0.5 * np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0

        # self.this_dis = np.linalg.norm(
        #     self._agent_location - self._target_location, ord=2
        # )
        # reward = reward -self.this_dis


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self._mode == 'test':
            self._render_frame()

        return observation, reward, terminated, info

    def render(self,mode='human'):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                np.multiply(pix_square_size , self._target_location),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            # [320,320],
            (self._agent_location+0.5)*pix_square_size,
            pix_square_size / 3,
        )

        # Draw obstacles
        # for i in range(self.obstacle_num):
        #     pygame.draw.circle(
        #         canvas,
        #         (255, 255, 0),
        #         (self._obstacle_location[i]+0.5)*pix_square_size,
        #         self._obstacle_radius,
        #     )

        #Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
