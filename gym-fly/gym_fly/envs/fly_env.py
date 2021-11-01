import sys, math
import numpy as np

import Box2D, pyglet
from Box2D.b2 import (fixtureDef, polygonShape)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 40.0   # affects how fast-paced the game is, forces should be adjusted as well

# The vertical/horizontal strength of wing flaps
VP = 100.0
HP = 50.0

INITIAL_RANDOM = 500.0   # Scale of initial random force, set higher for harder.

y_o = 2.656278982758522 #The offset to put CoM at center of object
LANDER_POLY =[
    (-14, +17-y_o), (-23, 0-y_o), (-20 ,-10-y_o),
    (+20, -10-y_o), (+23, 0-y_o), (+14, +17-y_o)
    ]

VIEWPORT_W = 600
VIEWPORT_H = 400

TARGET_Y = 0.05 # Where the system should stabilize the y coordinate

# Wind parameters
WIND_SCALE = 0.1
WIND_MAX = 15
WIND_STEP = 0.3
WIND_INIT = 0

DELAY = 5 # Note: delay = 0 is no delay

NOISE_SCALE = 0.1

WING_OFFSET = 0 # Set higher to move the location of the wing impulses off of the center of mass.


class FlyEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.noise_scale = NOISE_SCALE
        self.wind_scale = WIND_SCALE; self.wind_max = WIND_MAX; 
        self.wind_init = WIND_INIT; self.wind_step = WIND_STEP;
        self.delay = DELAY
        self.target_y = TARGET_Y
        self.wing_offset = WING_OFFSET

        self.world = Box2D.b2World()
        self.lander = None

        self.prev_reward = None

        # observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

        # Action is two floats [left-wing, right-wing].
        self.action_space = spaces.Box(-np.inf, +np.inf, (2,), dtype=np.float32)
       
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.lander: return
        self.world.DestroyBody(self.lander)
        self.lander = None

    def reset(self):
        self._destroy()
        self.game_over = False
        self.prev_shaping = None
        self.wind = self.wind_init

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        initial_y = self.np_random.uniform(2*VIEWPORT_H/SCALE/3, 4*VIEWPORT_H/SCALE/5)
        initial_x = self.np_random.uniform(VIEWPORT_W/SCALE/2 - VIEWPORT_W/SCALE/3 , VIEWPORT_W/SCALE/2 + VIEWPORT_W/SCALE/3)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=1.0,
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
        
        # In case of delay, set up a buffer of previous actions to go through.
        self.action_buffer = [np.array([0, 0])]*self.delay 
        
        return self.step(np.array([0, 0]))[0]

    def step(self, action):
        
        done = False # The environment never reports done.
        
        self.action_buffer.append(action)
        action = self.action_buffer.pop(0)
        
        # Fly angle calcs
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle)) 
        impulse_pos_offset = (self.wing_offset/100*tip[1],self.wing_offset/100*tip[0])
        
        # Left wing impulse
        self.lander.ApplyLinearImpulse(((-tip[0]*VP + tip[1]*HP)/SCALE * (action[0]) ,
                                        ( tip[1]*VP + tip[0]*HP)/SCALE * (action[0]) ),
                                        self.lander.position - impulse_pos_offset , True)

        # Right wing impulse
        self.lander.ApplyLinearImpulse(((-tip[0]*VP - tip[1]*HP)/SCALE * (action[1]) ,
                                        ( tip[1]*VP - tip[0]*HP)/SCALE * (action[1])),
                                        self.lander.position + impulse_pos_offset, True)

        # Noise impulse
        noise = [self.noise_scale * self.np_random.uniform(-1.0, +1.0) for _ in range(2)]
        self.lander.ApplyLinearImpulse((HP/SCALE * (noise[0]) ,
                                        VP/SCALE * (noise[1]) ),
                                        self.lander.position, True)
        
        # Wind force
        self.lander.ApplyForce((self.wind,0),(self.lander.position[0], self.lander.position[1]), True)
        self.wind = np.clip(self.wind + self.wind_scale*self.wind_step * (self.np_random.randint(3)-1),-self.wind_scale*self.wind_max,self.wind_scale*self.wind_max)
            
        # Environment step
        self.world.Step(1.0/FPS, 6*30, 2*30)

        # Preparing the reported state
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y) / (VIEWPORT_H/SCALE)- 0.55 + self.target_y,
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            self.wind
            ]

        # Calculation of the reward (distance to target, action)
        reward = - 10*np.sqrt(state[0]**2 + (state[1]-self.target_y)**2) \
                 - 2*np.sqrt(action[0]**2 + action[1]**2) 

        # Updating the location of the wings based on the action
        for num,i in enumerate([-1,1]):
            self.wing_angs[num] = np.clip( self.wing_angs[num] + i* (0.2+7*abs(action[num])) * self.wing_speed[num], -1, 1)
            if abs(self.wing_angs[num])==1:self.wing_speed[num]=-self.wing_speed[num]

        
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human', text_ul=None, text_ur=None):
        
        from gym.envs.classic_control import rendering
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
        
        # Wind flag
        self.viewer.draw_polyline([(8.5*VIEWPORT_W/SCALE/10, 0), (8.5*VIEWPORT_W/SCALE/10, 1.5*VIEWPORT_H/SCALE/10)], color=(0, 0, 0))
        self.viewer.draw_polygon([(8.5*VIEWPORT_W/SCALE/10, 1.5*VIEWPORT_H/SCALE/10), 
                                 ((17+5*self.wind/self.wind_max/self.wind_scale)*VIEWPORT_W/SCALE/20, 1.25*VIEWPORT_H/SCALE/10),
                                 (8.5*VIEWPORT_W/SCALE/10, 1*VIEWPORT_H/SCALE/10)], color=(0.85, 0.68, 0.0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        
        # Rendering the flapping wings
        l = self.lander; x = l.position.x; y = l.position.y; la = -l.angle; angs = [w_a + la for w_a in self.wing_angs]
        xo = 20/SCALE; yo = 15/SCALE;
        xs = x+xo*np.cos(la)+yo*np.sin(la); ys = y+yo*np.cos(la)-xo*np.sin(la)
        for i,a in zip([-1,1],angs):
            self.viewer.draw_polyline([( x+i*xo*np.cos(la)+yo*np.sin(la),y+yo*np.cos(la)-i*xo*np.sin(la)),
                                       ( x+i*xo*np.cos(la)+yo*np.sin(la)+i*20/SCALE*np.cos(a),
                                        y+yo*np.cos(la)-i*xo*np.sin(la)-i*20/SCALE*np.sin(a))],
                                      color=(0, 0, 0), linewidth=2)
            

        # Target reticle
        self.viewer.draw_polyline([(VIEWPORT_W/SCALE/2, 0.55*VIEWPORT_H/SCALE-.4), (VIEWPORT_W/SCALE/2, 0.55*VIEWPORT_H/SCALE+.4)], color=(1,0,.1))
        self.viewer.draw_polyline([(VIEWPORT_W/SCALE/2-0.4, 0.55*VIEWPORT_H/SCALE), (VIEWPORT_W/SCALE/2+.4, 0.55*VIEWPORT_H/SCALE)], color=(1,0,.1))     
        
        
        return self.do_render(return_rgb_array=mode == 'rgb_array', text_ul = text_ul, text_ur = text_ur)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
    def do_render(self, return_rgb_array, text_ul, text_ur):
        
        pyglet.gl.glClearColor(1,1,1,1)
    
        self.viewer.window.clear()
        self.viewer.window.switch_to()
        self.viewer.window.dispatch_events()
        self.viewer.transform.enable()
        for geom in self.viewer.geoms:
            geom.render()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.transform.disable()
        if text_ul!= None:
            label_ul = pyglet.text.Label(
                text_ul,
                font_size=22,
                x=10,
                y=360,
                anchor_x="left",
                anchor_y="bottom",
                color=(60, 60, 60, 255),
                )
            label_ul.draw()
        if text_ur!= None:
            label_ur = pyglet.text.Label(
                text_ur,
                font_size=22,
                x=590,
                y=360,
                anchor_x="right",
                anchor_y="bottom",
                color=(60, 60, 60, 255),
                )
            label_ur.draw()
        self.viewer.window.flip()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.viewer.onetime_geoms = []
        return arr if return_rgb_array else self.viewer.isopen