

import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 50.0
SIDE_ENGINE_POWER = 10.0

INITIAL_RANDOM = 500.0   # Set 1500 to make game harder

y_o = 2.656278982758522
LANDER_POLY =[
    (-14, +17-y_o), (-23, 0-y_o), (-20 ,-10-y_o),
    (+20, -10-y_o), (+23, 0-y_o), (+14, +17-y_o)
    ]

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

TARGET_Y = 0.55

WIND_SCALE = 1.0
WIND_MAX = 15
WIND_STEP = 0.3
WIND_INIT = 0

DELAY = 5
NOISE_SCALE = 1.0



class FlyDelayedEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.noise_scale = NOISE_SCALE
        self.wind_scale = WIND_SCALE
        self.delay = DELAY

        self.world = Box2D.b2World()
        self.lander = None

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.lander: return
        self.world.contactListener = None
        self.world.DestroyBody(self.lander)
        self.lander = None

    def reset(self):
        self._destroy()
        self.game_over = False
        self.prev_shaping = None
        self.wind = WIND_INIT

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        initial_y = np.random.uniform(2*VIEWPORT_H/SCALE/3, 4*VIEWPORT_H/SCALE/5)
        initial_x = np.random.uniform(VIEWPORT_W/SCALE/2 - VIEWPORT_W/SCALE/3 , VIEWPORT_W/SCALE/2 + VIEWPORT_W/SCALE/3)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)
        

        self.wing_angs = [0,0]
        self.wing_speed = [1,1]

        self.drawlist = [self.lander] 
        
        self.action_buffer = [np.array([0, 0])]*self.delay if self.continuous else [0]*self.delay

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def step(self, action):
        
        self.action_buffer.append(action)
        action = self.action_buffer.pop(0)
        
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.noise_scale * self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
            
        if (self.continuous) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = action[0]
            else:
                m_power = 1.0
            ox = (tip[0] * (4/SCALE + 2 * dispersion[0]) +
                  side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)
        
        self.lander.ApplyForce((self.wind,0),(self.lander.position[0], self.lander.position[1]), True)
        self.wind = np.clip(self.wind + WIND_STEP * (np.random.randint(3)-1),-WIND_MAX,WIND_MAX)
        
        s_power = 0.0
            
        if (self.continuous) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = abs(action[1])
            else:
                direction = action-2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17/SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)
            
        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y) / (VIEWPORT_H/SCALE),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            self.wind
            ]
        assert len(state) == 7

        reward = 0
        shaping = \
            - 5*(state[0]**2 + (state[1]-TARGET_Y)**2) \
            - 0*10*(state[2]*state[2] + state[3]*state[3]) \
            - 0*10*state[4]**2 

        reward = shaping

        reward -= 0*m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= 0*s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0 or abs(state[1]-0.5)>=0.6:
            done = True
            reward = -100

        for num,i in enumerate([-1,1]):
            self.wing_angs[num] += i* 0.7* self.wing_speed[num]
            if abs(self.wing_angs[num])>1:self.wing_speed[num]=-self.wing_speed[num]

        

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

            
        
        # Wind flag
        self.viewer.draw_polyline([(8.5*VIEWPORT_W/SCALE/10, 0), (8.5*VIEWPORT_W/SCALE/10, 1.5*VIEWPORT_H/SCALE/10)], color=(0, 0, 0))
        self.viewer.draw_polygon([(8.5*VIEWPORT_W/SCALE/10, 1.5*VIEWPORT_H/SCALE/10), 
                                 ((17+3*self.wind/WIND_MAX)*VIEWPORT_W/SCALE/20, 1.25*VIEWPORT_H/SCALE/10),
                                 (8.5*VIEWPORT_W/SCALE/10, 1*VIEWPORT_H/SCALE/10)], color=(0.85, 0.68, 0.0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        
        # Flapping wings
        l = self.lander; x = l.position.x; y = l.position.y; la = -l.angle; angs = [w_a + la for w_a in self.wing_angs]
        xo = 20/SCALE; yo = 15/SCALE;
        xs = x+xo*np.cos(la)+yo*np.sin(la); ys = y+yo*np.cos(la)-xo*np.sin(la)
        for i,a in zip([-1,1],angs):
            self.viewer.draw_polyline([( x+i*xo*np.cos(la)+yo*np.sin(la),y+yo*np.cos(la)-i*xo*np.sin(la)),
                                       ( x+i*xo*np.cos(la)+yo*np.sin(la)+i*20/SCALE*np.cos(a),
                                        y+yo*np.cos(la)-i*xo*np.sin(la)-i*20/SCALE*np.sin(a))],
                                      color=(0, 0, 0), linewidth=2)
            

        # Target reticle
        self.viewer.draw_polyline([(VIEWPORT_W/SCALE/2, TARGET_Y*VIEWPORT_H/SCALE-.4), (VIEWPORT_W/SCALE/2, TARGET_Y*VIEWPORT_H/SCALE+.4)], color=(1,0,.1))
        self.viewer.draw_polyline([(VIEWPORT_W/SCALE/2-0.4, TARGET_Y*VIEWPORT_H/SCALE), (VIEWPORT_W/SCALE/2+.4, TARGET_Y*VIEWPORT_H/SCALE)], color=(1,0,.1))     
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class FlyDelayedEnvContinuous(FlyDelayedEnv):
    continuous = True

