#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Tuple, List

X_LIMITS = (75, 525)
Y_LIMITS = (75, 525)


MIN_DIST = 2

class Particle:
    def __init__(self, vel: np.array, position: np.array, r: int, mass: int,
                 color: Tuple[int]):
        self.vel = vel
        self.pos = position  # centre of the particle
        self.r = r  # radius around the centre
        self.mass = mass
        self.color = color

        # keep track of particles this particle has already had collisions
        # accounted for with in this frame
        # ensures that each interparticle collision is only simulated once
        # per frame
        self._collided_with = []

    def update_position(self, other_particles: List[Particle]) -> None:
        # reset list of particles this particle has collided with, as we are
        # in a new frame
        self._collided_with = []

        # check for collision with other particles and update velocity
        self._check_particle_collisions(other_particles)

        # check for collision w/ wall and update velocity accordingly
        self._check_wall_collision()

        # enforce wall collisions if new velocity of particle causes it to go
        # beyond a wall container in the next frame
        self._check_go_past_wall()

        # update position using updated velocity, and assuming time increment of
        # 1 unit
        self.pos = np.array(
            [self._get_x() + self.vel[0], self._get_y() + self.vel[1]]
        )

    def get_position(self) -> Tuple[float]:
        return self._get_x(), self._get_y()

    def _get_x(self) -> float:
        return self.pos[0]

    def _get_y(self) -> float:
        return self.pos[1]

    def _check_wall_collision(self) -> bool:
        """Checks if the particle has collided with any of the container walls,
        and adjusts the particle's velocity to collide elastically if so.

        A collision counts as the particle being within 2 pixels of a wall,
        including the radius of the particle.
        """
        # check for collision with left/right container borders
        # count as collision if particle is MIN_DIST + radius units away from
        # border,
        dx_left = self._get_x() - (X_LIMITS[0] + MIN_DIST + self.r)
        dx_right = (X_LIMITS[1] - MIN_DIST - self.r) - self._get_x()
        dy_top = self._get_y() - (Y_LIMITS[0] + MIN_DIST + self.r)
        dy_bottom = (Y_LIMITS[1] - MIN_DIST - self.r) - self._get_y()
        
        # collided with left or right border
        if dx_left <=0 or dx_right <= 0:
            # particle's velocity is perpendicular to left/right border, so
            # reverse its velocity
            if np.dot(self.vel, np.array([0, 1])) == 0:
                self.vel = -self.vel
            else:
                # otherwise, flip velocity vector and negate new y velocity,
                # to bounce off right angle from wall
                self.vel = np.array([self.vel[1], -self.vel[0]])
        
        elif dy_top <= 0 or dy_bottom <= 0:
            # particle velocity perpendicular to top/bottom border
            if np.dot(self.vel, np.array([1, 0])) == 0:
                self.vel = -self.vel
            else:
                self.vel = np.array([self.vel[1], -self.vel[0]])

    def _check_particle_collisions(self, other_particles: List[Particle]) -> None:
        """Check if particle has collided with some other particle.
        If so, adjust the velocity of both particles accordingly, assuming
        a perfect elastic collision.
        """
        for p in other_particles:
            # distance between 2 particles = distance between centres minus radii
            # of both particles
            if p in self._collided_with:
                break  # collision was already accounted for by this particle

            dist = np.linalg.norm(self.pos - p.pos) - self.r - p.r

            if dist <= MIN_DIST:
                # only let particles collide if they're moving towards each other
                # if they're moving away from each other, then they probably
                # already collided in a previous frame
                moving_toward = np.dot(p.pos - self.pos, self.vel - p.vel) > 0
                if moving_toward:
                    self._collide_particles(p)
                    self._collided_with.append(p)
    
    def _collide_particles(self, other: Particle) -> None:
        """Adjust velocities of this particle and some other particle after
        they've undergone a 2D collision.
        """
        a1 = (2 * other.mass) / (self.mass + other.mass)
        b1 = np.dot((self.vel - other.vel), (self.pos - other.pos)) / np.linalg.norm(self.pos - other.pos) ** 2
        c1 = self.pos - other.pos

        a2 = (2 * self.mass) / (self.mass + other.mass)
        b2 = np.dot((other.vel - self.vel), (other.pos - self.pos)) / np.linalg.norm(other.pos - self.pos) ** 2
        c2 = other.pos - self.pos

        self.vel = self.vel - (a1 * b1 * c1)
        other.vel = other.vel - (a2 * b2 * c2)
    
    def _check_go_past_wall(self) -> None:
        """Checks whether the particle's new velocity, after colliding with wall
        and/or other particles will cause it to go beyond the walls in the next
        frame.

        Make the particle "collide" with the wall it is about to pass if this
        is the case.
        """
        future_x = self._get_x() + self.vel[0]
        future_y = self._get_y() + self.vel[1]

        dx_left_future = future_x - (X_LIMITS[0] + MIN_DIST + self.r)
        dx_right_future = (X_LIMITS[1] - MIN_DIST - self.r) - future_x
        dy_top_future = future_y - (Y_LIMITS[0] + MIN_DIST + self.r)
        dy_bottom_future = (Y_LIMITS[1] - MIN_DIST - self.r) - future_y

        if dx_left_future <= 0 or dx_right_future <= 0:
            # particle's velocity is perpendicular to left/right border, so
            # reverse its velocity
            if np.dot(self.vel, np.array([0, 1])) == 0:
                self.vel = -self.vel
            else:
                # otherwise, flip velocity vector and negate new y velocity,
                # to bounce off right angle from wall
                self.vel = np.array([self.vel[1], -self.vel[0]])

        elif dy_top_future <= 0 or dy_bottom_future <= 0:
            # particle velocity perpendicular to top/bottom border
            if np.dot(self.vel, np.array([1, 0])) == 0:
                self.vel = -self.vel
            else:
                self.vel = np.array([self.vel[1], -self.vel[0]])
