"""Aleksander Viiret
Rekursioonipuu"""

import turtle
import random
from colour import Color

varvid = []
varvid0 = list(Color("red").range_to(Color("violet"), 512))
for x in varvid0:
    varvid.append(x.hex)
varv = 0

kk = turtle.Turtle()
kk.hideturtle()
kk.left(90)
kk.speed(0)
kk.pensize(2)
kk.penup()
kk.goto(0, -200)


def puu(i):
    global varv
    if i >= 10:
        kk.color(varvid[varv])
        if varv != 511:
            varv += 1
        else:
            pass
        kk.pensize(0.1*i)
        kk.pendown()
        kk.forward(i)
        nurk = random.randint(10, 40)
        kk.left(nurk)
        puu(i*random.uniform(0.6, 0.9))
        kk.right(nurk*2)
        puu(i*random.uniform(0.6, 0.9))
        kk.left(nurk)
        kk.penup()
        kk.backward(i)
    else:
        kk.begin_fill()
        kk.circle(2)
        kk.end_fill()


puu(100)
turtle.done()
