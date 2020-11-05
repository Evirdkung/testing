import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

num1=(-3,-1,1,3)
verticies =[]
for x in num1:
    for y in num1:
        for z in num1:
            a=[x,y,z]
            verticies.append(a)
edges = []
for x in range (0,63):
    if x%4!=3:
        a=(x,x+1)
        edges.append(a)
    if x%16<=11:
        a=(x,x+4)
        edges.append(a)
    if x%64<=47:
        a=(x,x+16)
        edges.append(a) 
colors = (
(1,0,0),#red0
(0,1,0),#green1
(0,0,1),#blue2
(1,1,1),#white3
(1,1,0),#yellow4
(1,0.5,0),#orange5
(0,0,0)#black6
)

surfaces_left=[]
for x in (0,1,2,4,5,6,8,9,10):
    a=(x,x+1,x+5,x+4)
    surfaces_left.append(a)
surfaces_right=[]
for x in (48,49,50,52,53,54,56,57,58):
    a=(x,x+1,x+5,x+4)
    surfaces_right.append(a)

#copy start
def Cube():
    glBegin(GL_QUADS)
    for surface in surfaces_left[0:9]:
        for vertex in surface:
            glColor3fv(colors[2])
            glVertex3fv(verticies[vertex])
    for surface in surfaces_right[0:9]:
        for vertex in surface:
            glColor3fv(colors[1])
            glVertex3fv(verticies[vertex])


    glColor3fv(colors[3])#make line black      
    glEnd()
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
#copy end

def main():
    print("Test")
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(75, (display[0]/display[1]), 0.1, 500.0)
    glTranslatef(0.0,0.0, -10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #glRotatef(1, 3, 1, 1)
        glRotatef(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)
main()
