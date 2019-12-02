# -*- coding: utf-8 -*-
import numpy as np
import argparse
from distutils.util import strtobool
import pygame, sys
import cv2

from matchernet import MPCEnv, MovieWriter

from car import CarDynamics, CarRenderer
from obstacle_reward_system import ObstacleRewardSystem


BLACK = (0, 0, 0)
FPS = 60


class Display(object):
    def __init__(self,
                 display_size,
                 env,
                 recording=False):
        self.env = env
        
        pygame.init()
        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('MPCenv')
        
        if recording:
            self.writer = MovieWriter("out.mov", display_size, FPS)
        else:
            self.writer = None
            
    def update(self, left, right, up, down):
        self.surface.fill(BLACK)
        self.process(left, right, up, down)
        pygame.display.update()

    def show_image(self, state):
        state = (state * 255.0).astype(np.uint8)
        image = pygame.image.frombuffer(state, (256, 256), 'RGB')
        self.surface.blit(image, (0, 0))

        if self.writer is not None:
            self.writer.add_frame(state)
        
    def process(self, left, right, up, down):
        force = 0.0
        angle = 0.0
        if up:
            force += 0.2
        if down:
            force -= 0.2
        if left:
            angle -= 0.5
        if right:
            angle += 0.5
        action = [angle, force]
        state, reward = self.env.step(action)
        self.show_image(state)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=strtobool,
                        default="false")
    args = parser.parse_args()    
    
    recording = args.recording
    display_size = (256, 256)
    
    dt = 0.03
    dynamics = CarDynamics(dt)
    renderer = CarRenderer()
    reward_system = ObstacleRewardSystem()
    
    env = MPCEnv(dynamics, renderer, reward_system, use_visual_state=True)
    display = Display(display_size, env, recording=recording)
    
    clock = pygame.time.Clock()
    
    running = True
    
    left_pressed = False
    right_pressed = False
    up_pressed = False
    down_pressed = False
    esc_pressed = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    left_pressed = True
                elif event.key == pygame.K_RIGHT:
                    right_pressed = True
                elif event.key == pygame.K_UP:
                    up_pressed = True
                elif event.key == pygame.K_DOWN:
                    down_pressed = True
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.KEYUP :
                if event.key == pygame.K_LEFT:
                    left_pressed = False
                elif event.key == pygame.K_RIGHT:
                    right_pressed = False
                elif event.key == pygame.K_UP:
                    up_pressed = False
                elif event.key == pygame.K_DOWN:
                    down_pressed = False
            
        display.update(left_pressed, right_pressed, up_pressed, down_pressed)
        clock.tick(FPS)
        
    display.close()
    
if __name__ == '__main__':
    main()
