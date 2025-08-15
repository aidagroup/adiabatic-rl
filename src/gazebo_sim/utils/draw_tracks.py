"""
draws albedo maps for Gazebo ground planes etc. Produced SVG file can be converted to PNG using ImageMagick.
"""

from svg_turtle import *

import os, sys
import numpy as np

# mapping 100x100m to 5000x5000 pixels
# factor is 5

EPS = 1e-3
FACTOR = 50
# in m
WIDTH, HEIGHT = 100, 100 # in m
LINEWIDTH=10

DEBUG=True

t = SvgTurtle((WIDTH * FACTOR) + 1, (HEIGHT * FACTOR) + 1)
t.width(LINEWIDTH)


# scale from m with coords in [-width//2, width//2] to pixel coords with scale [0,width]
def scale(*args):
    return np.multiply(args, FACTOR) # +FACTOR*np.array([WIDTH//2, HEIGHT//2,0.])

def animate(flag=None):
    if flag is None: s.tracer()
    else: s.tracer(flag, 10)

def reset(x, y, o):
    #print(dir(t))
    t.penup()
    t.home()
    t.setposition(x, y)
    if o == 'x': t.left(0)
    if o == 'y': t.left(90)
    t.pendown()

def check(x, y, o, l, c):
    if o == 'x':
        i = 0
        v = x + l
    if o == 'y':
        i = 1
        v = y + l

    if c == 'gt': return t.position()[i] - EPS >= v
    if c == 'lt': return t.position()[i] + EPS <= v

def grid(e):
    e, = scale(e)

    t.pencolor('yellow')
    for x in range(-WIDTH, +WIDTH + 1, e):
        straight(x, -HEIGHT, 2 * HEIGHT, 'y')
    for y in range(-HEIGHT, +HEIGHT + 1, e):
        straight(-WIDTH, y, 2 * WIDTH, 'x')
    t.pencolor('black')

    straight(-e, 0, 2 * e, 'x')
    straight(0, -e, 2 * e, 'y')


# x,y: start in pixels
# l: length in pixels
# o: orientation x or y
def straight(x, y, o, l):
    x, y, l = scale(x, y, l)

    reset(x, y, o)
    t.forward(l)

def circle(x, y, o, r, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    t.circle(r, steps=s)

def ellipsis(x, y, o, rx, ry, s=None):
    x, y, rx, ry = scale(x, y, rx, ry)

    reset(x, y, o)
    if o == 'x':
        t.right(45)
        for _ in range(2):
            t.circle(rx, 90, steps=s)
            t.circle(ry, 90, steps=s)
    if o == 'y':
        t.right(45)
        for _ in range(2):
            t.circle(ry, 90, steps=s)
            t.circle(rx, 90, steps=s)

def polygon(x, y, o, k, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(k)
        t.left(360/d)

# == skyline
def flower(x, y, o, n, r, a, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.circle(r, a, steps=s)
        t.left(360/n-a)

def flower_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.circle(r, 180, steps=s)
        t.right(170)

# == slalom
def gear(x, y, o, n, r1, r2, a1, a2, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.circle(+r1, a1, steps=s)
        t.circle(-r2, a2, steps=s)

def gear_poly(x, y, o, n, k1, k2, a1, a2, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

# == signal
def rounded(x, y, o, n, r, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.forward(r*np.pi/n)
        t.circle(r, 360/n, steps=s)

def rounded_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

# == up_n_down
def puzzle(x, y, o, n, k, r, a, s=None):
    x, y, k, r = scale(x, y, k, r)

    reset(x, y, o)
    for _ in range(n):
        t.forward(r*np.pi/n)
        t.circle(r, 360/n, steps=s)

def puzzle_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

def skyline(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)

def skyline_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': t.right(360/d)
    while True:
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.left(360/d)

def slalom(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def slalom_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': t.right(360/d)
    while True:
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.left(360/d)
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.right(360/d)

def signal(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    a = 200
    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a/2)
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)
        t.right(a/2)

def signal_poly(x, y, o, l, k, a, d):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)
        t.right(a)

def up_n_down(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.left(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        t.left(a)

def up_n_down_poly(x, y, o, l, k, a, d):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.left(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        t.left(a)

def snake(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/2)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/2)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)

def snake_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.left(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.left(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.right(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.right(90)

def wave_lr(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.left(52)
    # t.left(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)

def wave_rl(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.right(52)
    # t.right(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def wave_lr_poly(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.left(52)
    # t.left(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)

def wave_rl_poly(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.right(52)
    # t.right(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def mesh(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)

def mesh_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def spiral_in_circle(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)

    # a = 0
    # while a < 360 * 5:
    #     r = a / (2 * np.pi)
    #     a += 1
    #     # t.circle(r / (1 + np.log(i)), 100 / (1 + np.log(i)))
    #     # t.circle(r / (1 + i / 1e1), 100 / (1 + i / 1e1))
    #     t.circle(r, np.radians(a))

    d = 0.5
    a = 0

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10:
        r = d * a / (2 * np.pi)
        t.goto(r * np.cos(np.radians(a)) + x, r * np.sin(np.radians(a)) + y)
        a += 1

    # a = 0
    # d = 25
    # while r > 0:
    #     t.circle(r, a)
    #     r -= d
    #     a += 10

def spiral_in_ellipsis(x, y, o, f, rx, ry, s=None):
    x, y, f, rx, ry = scale(x, y, f, rx, ry)

    reset(x, y, o)
    pass

def spiral_in_polygon(x, y, o, f, k, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    while True:
        for _ in range(d):
            t.forward(l)
            t.left(360/d)
            l -= s
            if l <= 0: return

def spiral_in_flower(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_flower_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_gear(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_gear_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_rounded(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_rounded_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_puzzle(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_puzzle_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_circle(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)

    # a = 0
    # while a < 360 * 5:
    #     r = a / (2 * np.pi)
    #     a += 1
    #     # t.circle(r / (1 + math.log(i)), 100 / (1 + math.log(i)))
    #     # t.circle(r / (1 + i / 1e1), 100 / (1 + i / 1e1))
    #     t.circle(r, np.radians(a))

    # d = 1
    # a = 0
    # while a < 360 * 10:
    #     r = d * a / (2 * math.pi)
    #     t.goto(r * math.cos(math.radians(a)) + x, r * math.sin(math.radians(a)) - y)
    #     a += 1

    # a = 0
    # d = 25
    # while r > 0:
    #     t.circle(r, a)
    #     r -= d
    #     a += 10

def spiral_out_ellipsis(x, y, o, f, rx, ry, s=None):
    x, y, f, rx, ry = scale(x, y, f, rx, ry)

    reset(x, y, o)
    pass

def spiral_out_polygon(x, y, o, f, k, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    while True:
        for _ in range(d):
            t.forward(l)
            t.left(360/d)
            l -= s
            if l <= 0: return

def spiral_out_flower(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_flower_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_gear(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_gear_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_rounded(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_rounded_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_puzzle(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_puzzle_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)


def slalom_special(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        r -= 3.8
        if r < 0: break
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def slalom_special_double(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    org_r = r
    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    flag = True
    while True:
        if flag: r -= 8.05
        if r < 8.05: flag = False
        if not flag: r += 8.05
        if r > org_r: break
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def spiral_in_circle_special_r(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    d = 0.5
    a = 0

    sign = +1
    if r < 0: sign = -1

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10 + 1:
        r = d * a / (2 * np.pi)
        t.goto(r * np.cos(np.radians(sign*a)) + x, r * np.sin(np.radians(sign*a)) + y)
        a += 1

def spiral_in_circle_special_l(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    d = 0.5
    a = 0

    sign = +1
    if r < 0: sign = -1

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10 + 1:
        r = d * a / (2 * np.pi)
        t.goto(x + r * np.cos(np.radians(sign*(a + 180))), y + r * np.sin(np.radians(sign*(a + 180))))
        a += 1

def spiral_in_circle_special_double_r(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # # Setting the angle between the spiral arms
    # phi = 137.508

    # # Loop to draw the spiral
    # for i in range(200):
    #     angle = i * phi
    #     x = (r * np.sqrt(i)) * np.cos(np.radians(angle))
    #     y = (r * np.sqrt(i)) * np.sin(np.radians(angle))
    #     turtle.goto(x, y)
    #     turtle.dot(5 + i // 5)

    # r = 1
    # theta_max = 10 * 360
    # delta_theta = 10
    # theta = 0

    # turtle.penup()

    # while theta < theta_max:
    #     # Convert polar to cartesian coordinates
    #     x = r * np.sqrt(theta) * np.cos(np.radians(theta))
    #     y = r * np.sqrt(theta) * np.sin(np.radians(theta))

    #     print(r)
    #     print(x, y)

    #     turtle.goto(x, y)
    #     turtle.pendown()

    #     # Increase radius for next point
    #     r += delta_theta
    #     theta += delta_theta

    length = 500
    turns = 5

    t.penup()   # Lift the pen up off the paper
    for i in range(-360 * turns, 360 * turns):  # Loop through the desired number of degrees
        r = length * np.sqrt(abs(i) / (360*turns))  # Calculate the radius at the current angle
        theta = np.radians(i)  # Convert the angle from degrees to radians
        if i < 0: theta = np.pi - theta
        xx = r * np.cos(theta) + x  # Calculate the x-coordinate
        yy = r * np.sin(theta) + y  # Calculate the y-coordinate
        t.goto(xx, yy)  # Move the turtle to the new location
        t.pendown()   # Put the pen back down on the paper

def spiral_in_circle_special_double_l(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # # Setting the angle between the spiral arms
    # phi = 137.508

    # # Loop to draw the spiral
    # for i in range(200):
    #     angle = i * phi
    #     x = (r * np.sqrt(i)) * np.cos(np.radians(angle))
    #     y = (r * np.sqrt(i)) * np.sin(np.radians(angle))
    #     turtle.goto(x, y)
    #     turtle.dot(5 + i // 5)

    # r = 1
    # theta_max = 10 * 360
    # delta_theta = 10
    # theta = 0

    # turtle.penup()

    # while theta < theta_max:
    #     # Convert polar to cartesian coordinates
    #     x = r * np.sqrt(theta) * np.cos(np.radians(theta))
    #     y = r * np.sqrt(theta) * np.sin(np.radians(theta))

    #     print(r)
    #     print(x, y)

    #     turtle.goto(x, y)
    #     turtle.pendown()

    #     # Increase radius for next point
    #     r += delta_theta
    #     theta += delta_theta

    length = 500
    turns = 5

    t.penup()   # Lift the pen up off the paper
    for i in range(-360 * turns, 360 * turns):  # Loop through the desired number of degrees
        r = length * np.sqrt(abs(i) / (360*turns))  # Calculate the radius at the current angle
        theta = np.radians(i)  # Convert the angle from degrees to radians
        if i < 0: theta = np.pi - theta
        xx = r * np.cos(-theta) + x  # Calculate the x-coordinate
        yy = r * np.sin(-theta) + y  # Calculate the y-coordinate
        t.goto(xx, yy)  # Move the turtle to the new location
        t.pendown()   # Put the pen back down on the paper

def spiral_in_rounded_special_r(x, y, o, f, n, r, a, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # d = 2
    # a = 0
    # t.penup()
    # t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    # t.pendown()
    # while a < 360 * 10:
    #     r = d * a / (2 * np.pi)
    #     t.goto(r * np.cos(np.radians(a)) + x, r * np.sin(np.radians(a)) + y)
    #     a += 1
    sign = 1
    if r < 0:
        t.right(180)
        sign = -1
        r = -r
    d = 0.9 * FACTOR
    for _ in range(10):
        if r < 0: break
        t.forward(r*np.pi/n)
        t.circle(sign*r, 180, steps=s)
        r -= d * FACTOR

        if r < 0: break
        t.forward(r*np.pi/n)
        t.circle(sign*r, 180, steps=s)
        r -= d * FACTOR

def spiral_in_rounded_special_l(x, y, o, f, n, r, a, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    sign = 1
    if r < 0:
        t.right(180)
        sign = -1
        r = -r
    d = 0.9 * FACTOR
    for _ in range(10):
        if r < 0: break
        t.forward(-r*np.pi/n)
        t.circle(sign*-r, -180, steps=s)
        r -= d * FACTOR

        if r < 0: break
        t.forward(-r*np.pi/n)
        t.circle(sign*-r, -180, steps=s)
        r -= d * FACTOR


def rect(x,y,w,h,fillcolor=None):
  t.width(0) ;
  if fillcolor is not None:
    t.fillcolor(fillcolor) ;
    t.begin_fill() ;
  x,y,w,h = scale(x,y,w,h) ;
  t.setposition(x,y)  ;
  t.setheading(90) ;
  t.pendown() ;
  t.forward(w) ;
  t.right(90) ;
  t.forward(h) ;
  t.right(90) ;
  t.forward(w) ;
  t.right(90) ;
  t.forward(h) ;

  if fillcolor is not None:
    t.end_fill() ;
  t.width(LINEWIDTH) ;


# DRAW Circle Tracks

def draw_circle_tracks():
    tracks = [
        'circle_red', 'circle_green', 'circle_blue', 'circle_yellow', 'circle_white'
    ]

    #t.fillcolor("green") 
    #rect(0,0,10,10,"green")
    t.penup()
    rect(-50,-10,30,20,'red')
    t.penup()
    rect(-30,-10,30,20,'green')
    t.penup()
    rect(-10,-10,30,20,'blue')
    t.penup()
    rect(10,-10,30,20,'yellow')
    for i, name in enumerate(tracks):
        t.penup()
        #print(name)
        
        if i == 0: circle(-40, 0, "x", 4) ; print("RED: [pos,rot] [-40.0,0.0]"); # The starting value for drawing the circle is the actual value in the sim!
        elif i == 1: circle(-20, 0, "x", 4) ; print("GREEN: [pos,rot] [-20.0,0.0]");
        elif i == 2: circle(0, 0, "x", 4) ; print("BLUE: [pos,rot] [0.0,0.0]");
        elif i == 3: circle(20, 0, "x", 4) ; print("YELLOW: [pos,rot] [20.0,0.0]");
        elif i == 4: circle(40, 0, "x", 4) ; print("WHITE: [pos,rot] [40.0,0.0]");

def draw_arena(ground_color='white'):
    t.penup()
    rect(-2.5,-2.5,5,5,'black')
    rect(-2.3,-2.3,4.6,4.6,ground_color)


# -------------------- MAIN MAIN MAIN ----------------------------------------------
if True:
    t.screen.bgcolor("white")
    #draw_circle_tracks()
    draw_arena()

    t.save_as('temp.svg')

