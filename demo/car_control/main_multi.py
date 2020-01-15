# -*- coding: utf-8 -*-
import numpy as np

from matchernet import iLQG, MovieWriter, AnimGIFWriter, MultiAgentDynamics, MultiAgentCost, MultiAgentRenderer, MultiAgentRewardSystem
from car import CarDynamics, CarCost, CarRenderer, CarObstacle

import cv2


def render_obstacles(image, obstacles, t):
    image_width = image.shape[1]
    render_scale = image_width / 2.0
    render_offset = image_width / 2.0
    
    for obstacle in obstacles:
        rx = int(obstacle.pos[0] * render_scale + render_offset)
        ry = int(obstacle.pos[1] * render_scale + render_offset)
        rate = obstacle.calc_rate(t)
        if rate > 1.0:
            rate = 1.0
        if obstacle.is_good:
            reward_color = (1, 1-rate, 1-rate)
        else:
            reward_color = (1-rate, 1-rate, 1-rate)
        radius = 0.1
        image = cv2.circle(image,
                           (rx, ry), int(radius * render_scale),
                           reward_color, -1)
    return image


def render_trajectory(image, x_list):
    image_width = image.shape[1]
    render_scale = image_width / 2.0
    render_offset = image_width / 2.0
    
    for i in range(len(x_list) - 1):
        x0 = int(x_list[i][0] * render_scale + render_offset)
        y0 = int(x_list[i][1] * render_scale + render_offset)
        x1 = int(x_list[i+1][0] * render_scale + render_offset)
        y1 = int(x_list[i+1][1] * render_scale + render_offset)
        image = cv2.line(image, (x0,y0), (x1,y1), (0.25, 0.5, 0.25), 1)


def render_time_step(image, time_step):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img=image,
                text="time={}".format(time_step),
                org=(5, 10),
                fontFace=font,
                fontScale=0.3,
                color=(0,0,0),
                thickness=1,
                lineType=cv2.LINE_AA)

    
def main():
    agent_size = 10
    
    dt = 0.03
    dynamics = MultiAgentDynamics(CarDynamics(), agent_size)
    renderer = MultiAgentRenderer(CarRenderer(image_width=256), agent_size)

    obstacles = []
    
    obstacle0 = CarObstacle(pos=np.array([0.5, 0.0], dtype=np.float32),
                            is_good=False)
    obstacles.append(obstacle0)
    obstacle1 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle1)
    obstacle2 = CarObstacle(pos=np.array([0.8, 0.6], dtype=np.float32),
                            is_good=False)
    obstacles.append(obstacle2)
    obstacle3 = CarObstacle(pos=np.array([0.6, 0.8], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle3)

    """
    obstacle0 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle0)
    obstacle1 = CarObstacle(pos=np.array([0.6, 0.8], dtype=np.float32),
                            is_good=True)
    """
    obstacles.append(obstacle1)
    
    cost = MultiAgentCost(CarCost(obstacles), agent_size)
    
    ilqg = iLQG(dynamics=dynamics, cost=cost, dt=dt)
    
    # Initial state
    x0 = np.zeros(4 * agent_size, dtype=np.float32)
    for i in range(agent_size):
        x0[4*i  ] = np.random.rand() * 2.0 - 1.0
        x0[4*i+1] = np.random.rand() * 2.0 - 1.0
        x0[4*i+2] = np.random.rand() * 2.0 * np.pi
    
    movie = MovieWriter("car_ilqg0.mov", (256, 256), 60)
    anim_gif = AnimGIFWriter("car_ilqg0.gif", 60)

    T = 50
    receding_T = 10

    #T = 2
    #receding_T = 1

    # Initial control sequence
    u0 = np.zeros((T, 2*agent_size), dtype=np.float32)

    time_step = 0

    for j in range(20):
        iter_max = 10
        
        x_list, u_list, K_list = ilqg.optimize(x0,
                                               u0,
                                               T,
                                               start_time_step=time_step,
                                               iter_max=iter_max)
        
        x = x0
        
        for i in range(receding_T):
            x_t = x_list[i]
            u_t = u_list[i]
            K_t = K_list[i]
            u = u_t + K_t @ (x - x_t)
            xdot = dynamics.value(x, u)
            next_x = x + xdot * dt
            
            cost.apply_state(x_t, time_step * dt)
            
            x = next_x

            image = renderer.render(x, u)
            render_obstacles(image, obstacles, time_step * dt)

            render_trajectory(image, x_list[i:])
            render_time_step(image, time_step)
            
            image = (image * 255.0).astype(np.uint8)
            movie.add_frame(image)
            anim_gif.add_frame(image)

            time_step += 1

        # Set next initial state
        x0 = x

        # Set next initial control signals by copying remaining control signals.
        u0 = np.zeros((T, 2*agent_size), dtype=np.float32)
        u0[:T-receding_T,:] = np.array(u_list, dtype=np.float32)[receding_T:,:]
        
    movie.close()
    anim_gif.close()

if __name__ == '__main__':
    main()
