import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
import time

move=[]
code=[]
solution=[]
rubik = [[["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]]
        ,[["b1","b2","b3"],["b4","b5","b6"],["b7","b8","b9"]]
        ,[["r1","r2","r3"],["r4","r5","r6"],["r7","r8","r9"]]
        ,[["g1","g2","g3"],["g4","g5","g6"],["g7","g8","g9"]]
        ,[["o1","o2","o3"],["o4","o5","o6"],["o7","o8","o9"]]
        ,[["w1","w2","w3"],["w4","w5","w6"],["w7","w8","w9"]],move,code]

turn_cube= [[["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]]
        ,[["b1","b2","b3"],["b4","b5","b6"],["b7","b8","b9"]]
        ,[["r1","r2","r3"],["r4","r5","r6"],["r7","r8","r9"]]
        ,[["g1","g2","g3"],["g4","g5","g6"],["g7","g8","g9"]]
        ,[["o1","o2","o3"],["o4","o5","o6"],["o7","o8","o9"]]
        ,[["w1","w2","w3"],["w4","w5","w6"],["w7","w8","w9"]],move,code]

solve = [[["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]]
        ,[["b1","b2","b3"],["b4","b5","b6"],["b7","b8","b9"]]
        ,[["r1","r2","r3"],["r4","r5","r6"],["r7","r8","r9"]]
        ,[["g1","g2","g3"],["g4","g5","g6"],["g7","g8","g9"]]
        ,[["o1","o2","o3"],["o4","o5","o6"],["o7","o8","o9"]]
        ,[["w1","w2","w3"],["w4","w5","w6"],["w7","w8","w9"]],move,code]

    def _solve_cube(self, *args,rubik_1=rubik):
        move=[]
        rubik_1[6]=[]
        solve_the_rubik(rubik_1)

        if len(rubik_1[6])==0:
            print("The rubik is already solve!!!")
            return rubik_1
        
        print("Solution is",rubik_1[6])
        print("Code is",rubik_1[7])
        sol=[]

        while len(rubik_1[6])!=0:
            a = rubik_1[6].pop(0)
            
            if a in ["(T)","(Jb)","(Ja)","(Y)","(Ra)","(x)","(y)","(z)","(x2)","(y2)","(z2)","(x3)","(y3)","(z3)"]:
                b={"(T)":["R","U","R3","U3","R3","F","R2","U3","R3","U3","R","U","R3","F3"],
                   "(Jb)":["R2","D","R","D3","R","F2","Rw3","F","Rw","F2"],
                   "(Ja)":["R","U","R3","F3","R","U","R3","U3","R3","F","R2","U3","R3","U3"],
                   "(Y)":["F","R","U3","R3","U3","R","U","R3","F3","R","U","R3","U3","R3","F","R","F3"],
                   "(Ra)":["R","U","R3","F3","R","U2","R3","U2","R3","F","R","U","R","U2","R3","U3"],
                   "(x)":["Rw","L3"],
                   "(x2)":["Rw2","L2"],
                   "(x3)":["Rw3","L"],
                   "(y)":["Uw","D3"],
                   "(y2)":["Uw2","D2"],
                   "(y3)":["Uw3","D"],
                   "(z)":["Fw","B3"],
                   "(z2)":["Fw2","B2"],
                   "(z3)":["Fw3","B"],
                   }
               
                for c in b[a]:
                    sol.append(c)                
            else:
                sol.append(a)

        while len(sol)!=0:
            i = sol.pop(0)
            if i in ("U","L","F","R","B","D"):
                self.rotate_face(i)
            if i in ["U2","L2","F2","R2","B2","D2"]:
                self.rotate_face(i[0],2)
            if i in ["U3","L3","F3","R3","B3","D3"]:        
                self.rotate_face(i[0], -1)
            if i in ("Uw","Lw","Fw","Rw","Bw","Dw"):
                self.rotate_face(i[0])
                self.rotate_face(i[0],layer=1)
            if i in ["Uw2","Lw2","Fw2","Rw2","Bw2","Dw2"]:
                self.rotate_face(i[0],2)
                self.rotate_face(i[0],2,layer=1)
            if i in ["Uw3","Lw3","Fw3","Rw3","Bw3","Dw3"]:        
                self.rotate_face(i[0], -1)
                self.rotate_face(i[0], -1,layer=1)
            #print("move is",i)

        code.clear()       
        solution.clear()

        rubik.clear()
        rubik.append([["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]])
        rubik.append([["b1","b2","b3"],["b4","b5","b6"],["b7","b8","b9"]])
        rubik.append([["r1","r2","r3"],["r4","r5","r6"],["r7","r8","r9"]])
        rubik.append([["g1","g2","g3"],["g4","g5","g6"],["g7","g8","g9"]])
        rubik.append([["o1","o2","o3"],["o4","o5","o6"],["o7","o8","o9"]])
        rubik.append([["w1","w2","w3"],["w4","w5","w6"],["w7","w8","w9"]])
        rubik.append(move)
        rubik.append(code)

        self.cube._move_list = []
        return rubik

####################################################################################
import random

code_colector = []
swap_position=[]
 
def read_rubik(rubik):
    print("")
    for i in range(0,3):
        print(" "*9,end="")
        for j in range(0,3):
            print(rubik[0][i][j],end=" ")
        print("")
    for i in range(0,3):
        for j in range(0,3):
            print(rubik[1][i][j],end=" ")
        for j in range(0,3):
            print(rubik[2][i][j],end=" ")
        for j in range(0,3):
            print(rubik[3][i][j],end=" ")
        for j in range(0,3):
            print(rubik[4][i][j],end=" ")
        print("")
    for i in range(0,3):
        print(" "*9,end="")
        for j in range(0,3):
            print(rubik[5][i][j],end=" ")
        print("")
    print("")

def turns(a,b=0):
    a=a.split(" ")
    for i in a:
        turn(i,b)

def turn(a,b=0):
    #read_rubik(rubik)
    if a=="U":
        U_turn(rubik)
    if a=="U2":
        U2_turn(rubik)
    if a=="U3":
        U3_turn(rubik)
    if a=="Uw":
        Uw_turn(rubik)
    if a=="Uw2":
        Uw2_turn(rubik)
    if a=="Uw3":
        Uw3_turn(rubik)
    
    if a=="L":
        L_turn(rubik)
    if a=="L2":
        L2_turn(rubik)
    if a=="L3":
        L3_turn(rubik)
    if a=="Lw":
        Lw_turn(rubik)
    if a=="Lw2":
        Lw2_turn(rubik)
    if a=="Lw3":
        Lw3_turn(rubik)
        
    if a=="F":
        F_turn(rubik)
    if a=="F2":
        F2_turn(rubik)
    if a=="F3":
        F3_turn(rubik)
    if a=="Fw":
        Fw_turn(rubik)
    if a=="Fw2":
        Fw2_turn(rubik)
    if a=="Fw3":
        Fw3_turn(rubik)
        
    if a=="R":
        R_turn(rubik)
    if a=="R2":
        R2_turn(rubik)
    if a=="R3":
        R3_turn(rubik)
    if a=="Rw":
        Rw_turn(rubik)
    if a=="Rw2":
        Rw2_turn(rubik)
    if a=="Rw3":
        Rw3_turn(rubik)
        
    if a=="B":
        B_turn(rubik)
    if a=="B2":
        B2_turn(rubik)
    if a=="B3":
        B3_turn(rubik)
    if a=="Bw":
        Bw_turn(rubik)
    if a=="Bw2":
        Bw2_turn(rubik)
    if a=="Bw3":
        Bw3_turn(rubik)
        
    if a=="D":
        D_turn(rubik)
    if a=="D2":
        D2_turn(rubik)
    if a=="D3":
        D3_turn(rubik)
    if a=="Dw":
        Dw_turn(rubik)
    if a=="Dw2":
        Dw2_turn(rubik)
    if a=="Dw3":
        Dw3_turn(rubik)
        
    if a=="M":
        M_turn(rubik)
    if a=="M2":
        M2_turn(rubik)
    if a=="M3":
        M3_turn(rubik)
    if a=="S":
        S_turn(rubik)
    if a=="S2":
        S2_turn(rubik)
    if a=="S3":
        S3_turn(rubik)
    if a=="E":
        E_turn(rubik)
    if a=="E2":
        E2_turn(rubik)
    if a=="E3":
        E3_turn(rubik)
        
    if a=="x":
        a="(x)"
        solution.append(a)
        print(a,end=" ")
        b=2
        x_rotation(rubik)  
    if a=="x2":
        a="(x2)"
        solution.append(a)
        print(a,end=" ")
        b=2
        x2_rotation(rubik)
    if a=="x3":
        a="(x3)"
        solution.append(a)
        print(a,end=" ")
        b=2
        x3_rotation(rubik)
    if a=="y":
        a="(y)"
        solution.append(a)
        print(a,end=" ")
        b=2
        y_rotation(rubik) 
    if a=="y2":
        a="(y2)"
        solution.append(a)
        print(a,end=" ")
        b=2
        y2_rotation(rubik)
    if a=="y3":
        a="(y3)"
        solution.append(a)
        print(a,end=" ")
        b=2
        y3_rotation(rubik)
    if a=="z":
        a="(z)"
        solution.append(a)
        print(a,end=" ")
        b=2
        z_rotation(rubik)
    if a=="z2":
        a="(z2)"
        solution.append(a)
        print(a,end=" ")
        b=2
        z2_rotation(rubik)
    if a=="z3":
        a="(z3)"
        solution.append(a)
        print(a,end=" ")
        b=2
        z3_rotation(rubik)
        
    if a=="T":
        a="(T)"
        solution.append(a)
        b=2
        T_perm(rubik)    
    if a=="Ja":
        a="(Ja)"
        solution.append(a)
        b=2
        Ja_perm(rubik)
    if a=="Jb":
        a="(Jb)"
        solution.append(a)
        b=2
        Jb_perm(rubik)
    if a=="Y":
        a="(Y)"
        solution.append(a)
        b=2
        Y_perm(rubik)
    if a=="Ra":
        a="(Ra)"
        solution.append(a)
        b=2
        Ra_perm(rubik)
        
    if a=="new":
        scramble(rubik)
        b=1
    if a=="sol":
        b=1
        solution.clear()
        solve_the_rubik(rubik)
    if a=="ok":
        b=1
        print("start")
        
    
    if b==0:
        solution.append(a)
    if b==1:
        solution.clear()
    if b==2:
        return solution
    return solution


### face
def face(rubik,x):
    new=rubik
    a=rubik[x][0][2]
    new[x][0][2]=rubik[x][0][0]
    new[x][0][0]=rubik[x][2][0]
    new[x][2][0]=rubik[x][2][2]
    new[x][2][2]=a
    b=rubik[x][1][2]
    new[x][1][2]=rubik[x][0][1]
    new[x][0][1]=rubik[x][1][0]
    new[x][1][0]=rubik[x][2][1]
    new[x][2][1]=b
    return new
### turn  
def U_turn(rubik):
    new=rubik
    face(rubik,0)
    a=rubik[1][0][0]
    b=rubik[1][0][1]
    c=rubik[1][0][2]
    new[1][0][0]=rubik[2][0][0]
    new[1][0][1]=rubik[2][0][1]
    new[1][0][2]=rubik[2][0][2]
 
    new[2][0][0]=rubik[3][0][0]
    new[2][0][1]=rubik[3][0][1]
    new[2][0][2]=rubik[3][0][2]
    
    new[3][0][0]=rubik[4][0][0]
    new[3][0][1]=rubik[4][0][1]
    new[3][0][2]=rubik[4][0][2]
    
    new[4][0][0]=a
    new[4][0][1]=b
    new[4][0][2]=c
    rubik = new
    return rubik

def U2_turn(rubik):
    U_turn(rubik)
    U_turn(rubik)
    return rubik

def U3_turn(rubik):
    U_turn(rubik)
    U_turn(rubik)
    U_turn(rubik)
    return rubik

def Uw_turn(rubik):
    U_turn(rubik)
    E3_turn(rubik)
    return rubik

def Uw2_turn(rubik):
    Uw_turn(rubik)
    Uw_turn(rubik)
    return rubik

def Uw3_turn(rubik):
    Uw_turn(rubik)
    Uw_turn(rubik)
    Uw_turn(rubik)
    return rubik

def L_turn(rubik):
    new=rubik
    face(rubik,1)

    a=rubik[0][0][0]
    b=rubik[0][1][0]
    c=rubik[0][2][0]
    
    new[0][0][0]=rubik[4][2][2]
    new[0][1][0]=rubik[4][1][2]
    new[0][2][0]=rubik[4][0][2]
    
    new[4][2][2]=rubik[5][0][0]
    new[4][1][2]=rubik[5][1][0]
    new[4][0][2]=rubik[5][2][0]
    
    new[5][0][0]=rubik[2][0][0]
    new[5][1][0]=rubik[2][1][0]
    new[5][2][0]=rubik[2][2][0]
    
    new[2][0][0]=a
    new[2][1][0]=b
    new[2][2][0]=c
    rubik = new
    return rubik

def L2_turn(rubik):
    L_turn(rubik)
    L_turn(rubik)
    return rubik

def L3_turn(rubik):
    L_turn(rubik)
    L_turn(rubik)
    L_turn(rubik)
    return rubik

def Lw_turn(rubik):
    L_turn(rubik)
    M_turn(rubik)
    return rubik

def Lw2_turn(rubik):
    Lw_turn(rubik)
    Lw_turn(rubik)
    return rubik

def Lw3_turn(rubik):
    Lw_turn(rubik)
    Lw_turn(rubik)
    Lw_turn(rubik)
    return rubik

def F_turn(rubik):
    new=rubik
    face(rubik,2)

    a=rubik[0][2][0]
    b=rubik[0][2][1]
    c=rubik[0][2][2]
    
    new[0][2][0]=rubik[1][2][2]
    new[0][2][1]=rubik[1][1][2]
    new[0][2][2]=rubik[1][0][2]
    
    new[1][2][2]=rubik[5][0][2]
    new[1][1][2]=rubik[5][0][1]
    new[1][0][2]=rubik[5][0][0]
    
    new[5][0][2]=rubik[3][0][0]
    new[5][0][1]=rubik[3][1][0]
    new[5][0][0]=rubik[3][2][0]
    
    new[3][0][0]=a
    new[3][1][0]=b
    new[3][2][0]=c
    rubik = new
    return rubik

def F2_turn(rubik):
    F_turn(rubik)
    F_turn(rubik)
    return rubik

def F3_turn(rubik):
    F_turn(rubik)
    F_turn(rubik)
    F_turn(rubik)
    return rubik

def Fw_turn(rubik):
    F_turn(rubik)
    S_turn(rubik)
    return rubik

def Fw2_turn(rubik):
    Fw_turn(rubik)
    Fw_turn(rubik)
    return rubik

def Fw3_turn(rubik):
    Fw_turn(rubik)
    Fw_turn(rubik)
    Fw_turn(rubik)
    return rubik

def R_turn(rubik):
    new=rubik
    face(rubik,3)

    a=rubik[0][0][2]
    b=rubik[0][1][2]
    c=rubik[0][2][2]
    
    new[0][0][2]=rubik[2][0][2]
    new[0][1][2]=rubik[2][1][2]
    new[0][2][2]=rubik[2][2][2]
    
    new[2][0][2]=rubik[5][0][2]
    new[2][1][2]=rubik[5][1][2]
    new[2][2][2]=rubik[5][2][2]
    
    new[5][0][2]=rubik[4][2][0]
    new[5][1][2]=rubik[4][1][0]
    new[5][2][2]=rubik[4][0][0]
    
    new[4][2][0]=a
    new[4][1][0]=b
    new[4][0][0]=c
    rubik = new
    return rubik

def R2_turn(rubik):
    R_turn(rubik)
    R_turn(rubik)
    return rubik

def R3_turn(rubik):
    R_turn(rubik)
    R_turn(rubik)
    R_turn(rubik)
    return rubik

def Rw_turn(rubik):
    R_turn(rubik)
    M3_turn(rubik)
    return rubik

def Rw2_turn(rubik):
    Rw_turn(rubik)
    Rw_turn(rubik)
    return rubik

def Rw3_turn(rubik):
    Rw_turn(rubik)
    Rw_turn(rubik)
    Rw_turn(rubik)
    return rubik

def B_turn(rubik):
    new=rubik
    face(rubik,4)

    a=rubik[0][0][0]
    b=rubik[0][0][1]
    c=rubik[0][0][2]
    
    new[0][0][0]=rubik[3][0][2]
    new[0][0][1]=rubik[3][1][2]
    new[0][0][2]=rubik[3][2][2]
    
    new[3][0][2]=rubik[5][2][2]
    new[3][1][2]=rubik[5][2][1]
    new[3][2][2]=rubik[5][2][0]
    
    new[5][2][2]=rubik[1][2][0]
    new[5][2][1]=rubik[1][1][0]
    new[5][2][0]=rubik[1][0][0]
    
    new[1][2][0]=a
    new[1][1][0]=b
    new[1][0][0]=c
    rubik = new
    return rubik

def B2_turn(rubik):
    B_turn(rubik)
    B_turn(rubik)
    return rubik

def B3_turn(rubik):
    B_turn(rubik)
    B_turn(rubik)
    B_turn(rubik)
    return rubik

def Bw_turn(rubik):
    B_turn(rubik)
    S3_turn(rubik)
    return rubik

def Bw2_turn(rubik):
    Bw_turn(rubik)
    Bw_turn(rubik)
    return rubik

def Bw3_turn(rubik):
    Bw_turn(rubik)
    Bw_turn(rubik)
    Bw_turn(rubik)
    return rubik

def D_turn(rubik):
    new=rubik
    face(rubik,5)

    a=rubik[1][2][0]
    b=rubik[1][2][1]
    c=rubik[1][2][2]
    
    new[1][2][0]=rubik[4][2][0]
    new[1][2][1]=rubik[4][2][1]
    new[1][2][2]=rubik[4][2][2]
    
    new[4][2][0]=rubik[3][2][0]
    new[4][2][1]=rubik[3][2][1]
    new[4][2][2]=rubik[3][2][2]
    
    new[3][2][0]=rubik[2][2][0]
    new[3][2][1]=rubik[2][2][1]
    new[3][2][2]=rubik[2][2][2]
    
    new[2][2][0]=a
    new[2][2][1]=b
    new[2][2][2]=c
    rubik = new
    return rubik

def D2_turn(rubik):
    D_turn(rubik)
    D_turn(rubik)
    return rubik

def D3_turn(rubik):
    D_turn(rubik)
    D_turn(rubik)
    D_turn(rubik)
    return rubik

def Dw_turn(rubik):
    D_turn(rubik)
    E_turn(rubik)
    return rubik

def Dw2_turn(rubik):
    Dw_turn(rubik)
    Dw_turn(rubik)
    return rubik

def Dw3_turn(rubik):
    Dw_turn(rubik)
    Dw_turn(rubik)
    Dw_turn(rubik)
    return rubik

def M_turn(rubik):

    new=rubik
   
    a=rubik[4][2][1]
    b=rubik[4][1][1]
    c=rubik[4][0][1]
    
    new[4][2][1]=rubik[5][0][1]
    new[4][1][1]=rubik[5][1][1]
    new[4][0][1]=rubik[5][2][1]
    
    new[5][0][1]=rubik[2][0][1]
    new[5][1][1]=rubik[2][1][1]
    new[5][2][1]=rubik[2][2][1]
    
    new[2][0][1]=rubik[0][0][1]
    new[2][1][1]=rubik[0][1][1]
    new[2][2][1]=rubik[0][2][1]
    
    new[0][0][1]=a
    new[0][1][1]=b
    new[0][2][1]=c
    rubik = new
    return rubik

def M2_turn(rubik):
    M_turn(rubik)
    M_turn(rubik)
    return rubik

def M3_turn(rubik):
    M_turn(rubik)
    M_turn(rubik)
    M_turn(rubik)
    return rubik

def E_turn(rubik):
    new=rubik
    
    a=rubik[1][1][0]
    b=rubik[1][1][1]
    c=rubik[1][1][2]
    
    new[1][1][0]=rubik[4][1][0]
    new[1][1][1]=rubik[4][1][1]
    new[1][1][2]=rubik[4][1][2]
    
    new[4][1][0]=rubik[3][1][0]
    new[4][1][1]=rubik[3][1][1]
    new[4][1][2]=rubik[3][1][2]
    
    new[3][1][0]=rubik[2][1][0]
    new[3][1][1]=rubik[2][1][1]
    new[3][1][2]=rubik[2][1][2]
    
    new[2][1][0]=a
    new[2][1][1]=b
    new[2][1][2]=c
    rubik = new
    return rubik

def E2_turn(rubik):
    E_turn(rubik)
    E_turn(rubik)
    return rubik

def E3_turn(rubik):
    E_turn(rubik)
    E_turn(rubik)
    E_turn(rubik)
    return rubik

def S_turn(rubik):
    new=rubik

    a=rubik[0][1][0]
    b=rubik[0][1][1]
    c=rubik[0][1][2]
    
    new[0][1][0]=rubik[1][2][1]
    new[0][1][1]=rubik[1][1][1]
    new[0][1][2]=rubik[1][0][1]
    
    new[1][2][1]=rubik[5][1][2]
    new[1][1][1]=rubik[5][1][1]
    new[1][0][1]=rubik[5][1][0]
    
    new[5][1][2]=rubik[3][0][1]
    new[5][1][1]=rubik[3][1][1]
    new[5][1][0]=rubik[3][2][1]
    
    new[3][0][1]=a
    new[3][1][1]=b
    new[3][2][1]=c
    rubik = new
    return rubik

def S2_turn(rubik):
    S_turn(rubik)
    S_turn(rubik)
    return rubik

def S3_turn(rubik):
    S_turn(rubik)
    S_turn(rubik)
    S_turn(rubik)
    return rubik

def x_rotation(rubik):
    M3_turn(rubik)
    R_turn(rubik)
    L3_turn(rubik)
    return rubik

def x2_rotation(rubik):
    x_rotation(rubik)
    x_rotation(rubik)
    return rubik

def x3_rotation(rubik):
    x_rotation(rubik)
    x_rotation(rubik)
    x_rotation(rubik)
    return rubik

def y_rotation(rubik):
    E3_turn(rubik)
    U_turn(rubik)
    D3_turn(rubik)
    return rubik

def y2_rotation(rubik):
    y_rotation(rubik)
    y_rotation(rubik)
    return rubik

def y3_rotation(rubik):
    y_rotation(rubik)
    y_rotation(rubik)
    y_rotation(rubik)
    return rubik

def z_rotation(rubik):
    S_turn(rubik)
    F_turn(rubik)
    B3_turn(rubik)
    return rubik

def z2_rotation(rubik):
    z_rotation(rubik)
    z_rotation(rubik)
    return rubik

def z3_rotation(rubik):
    z_rotation(rubik)
    z_rotation(rubik)
    z_rotation(rubik)
    return rubik

def T_perm(rubik):
    a = "R U R3 U3 R3 F R2 U3 R3 U3 R U R3 F3"
    turns(a,b=2)
    return rubik

def Jb_perm(rubik):
    a = "R2 D R D3 R F2 Rw3 F Rw F2"
    turns(a,b=2)
    return rubik

def Ja_perm(rubik):
    a = "R U R3 F3 R U R3 U3 R3 F R2 U3 R3 U3"
    turns(a,b=2)
    return rubik

def Y_perm(rubik):
    a = "F R U3 R3 U3 R U R3 F3 R U R3 U3 R3 F R F3"
    turns(a,b=2)
    return rubik

def Ra_perm(rubik):
    a = "R U R3 F3 R U2 R3 U2 R3 F R U R U2 R3 U3"
    turns(a,b=2)
    return rubik
###
def scramble(rubik):
    print("Let scramble : ",end="")
    i=1
    c=0
    d=0
    while i<=20:
        while c==d:
            c = random.randint(1,3)
        r = random.randint(1,6)
        if c==1:
            if r==1:
                x="U"
            if r==2:
                x="U2"
            if r==3:
                x="U3"
            if r==4:
                x="D"
            if r==5:
                x="D2"
            if r==6:
                x="D3"
        if c==2:
            if r==1:
                x="L"
            if r==2:
                x="L2"
            if r==3:
                x="L3"
            if r==4:
                x="R"
            if r==5:
                x="R2"
            if r==6:
                x="R3"
        if c==3:
            if r==1:
                x="F"
            if r==2:
                x="F2"
            if r==3:
                x="F3"
            if r==4:
                x="B"
            if r==5:
                x="B2"
            if r==6:
                x="B3"
        turn(x)
        i+=1
        d=c
        rubik[6].append(x)
    print("\n")
    return rubik

###Solve Binndford Function reader
def check_digit(sticker,rubik):
    for i in range(0,6):
        for j in range(0,3):
            for k in range(0,3):
                if sticker == rubik[i][j][k]:
                    digit=[i,j,k]
                    return digit
                
def check_sticker(digit,rubik):
    sticker=rubik[digit[0]][digit[1]][digit[2]]
    return sticker

###Solve Binndford center
def solve_center(rubik):
    if "y5"==rubik[1][1][1]:
        turns("z")
    if "y5"==rubik[2][1][1]:
        turns("x")
    if "y5"==rubik[3][1][1]:
        turns("z3")
    if "y5"==rubik[4][1][1]:
        turns("x3")
    if "y5"==rubik[5][1][1]:
        turns("x2")
    if "r5"==rubik[1][1][1]:
        turns("y3")
    if "r5"==rubik[3][1][1]:
        turns("y")
    if "r5"==rubik[4][1][1]:
        turns("y2")
    return rubik
    
###Solve Binndford edge
def solve_edge(buffer_edge,rubik):
    buffer_edge_digit=[0,1,2]
    buffer_edge=check_sticker(buffer_edge_digit,rubik)
    edge_swap=0
    while True :
        correct_edge=0        
        for i in range(0,6):
            if solve[i][0][1] == rubik[i][0][1]:
                correct_edge+=1
            if solve[i][1][0] == rubik[i][1][0]:
                correct_edge+=1
            if solve[i][1][2] == rubik[i][1][2]:
                correct_edge+=1
            if solve[i][2][1] == rubik[i][2][1]:
                correct_edge+=1
        if correct_edge==24:
            print("Total number of edge swap is\n",edge_swap)
            if edge_swap%2==1:
                print("Have parity do Ra_perm\n")
                turn("Ra")
                code.append("(Parity)")
            else:
                print("No parity\n")
                return rubik
            return rubik
        
        if buffer_edge=="y6" or buffer_edge=="g2":
            for i in range(0,6):
                if  solve[i][0][1] != rubik[i][0][1]:
                    if i!=3:
                        buffer_edge=rubik[i][0][1]
                        break
                if  solve[i][1][0] != rubik[i][1][0]:
                    buffer_edge=rubik[i][1][0]
                    break
                if  solve[i][1][2] != rubik[i][1][2]:
                    if i!=0:
                        buffer_edge=rubik[i][1][2]
                        break
                if  solve[i][2][1] != rubik[i][2][1]:
                    buffer_edge=rubik[i][2][1]
                    break

        if buffer_edge[0]=="y":
            if buffer_edge=="y2":
                turns("Jb")
            if buffer_edge=="y4":
                turns("T")
            if buffer_edge=="y8":
                turns("Ja")

        if buffer_edge[0]=="b":   
            if buffer_edge=="b2":
                turns("L3 Dw L3 T L Dw3 L")
            if buffer_edge=="b4":
                turns("Dw L3 T L Dw3")
            if buffer_edge=="b6":
                turns("Dw3 L T L3 Dw")    
            if buffer_edge=="b8":
                turns("L Dw L3 T L Dw3 L3")

        if buffer_edge[0]=="r":           
            if buffer_edge=="r2":
                turns("Lw3 Jb Lw")
            if buffer_edge=="r4":
                turns("L3 T L")
            if buffer_edge=="r6":
                turns("Dw2 L T L3 Dw2")
            if buffer_edge=="r8":
                turns("Lw3 Ja Lw")

        if buffer_edge[0]=="g":           
            if buffer_edge=="g4":
                turns("Dw3 L3 T L Dw") 
            if buffer_edge=="g6":
                turns("Dw L T L3 Dw3")        
            if buffer_edge=="g8":
                turns("D3 Lw3 Ja Lw D")

        if buffer_edge[0]=="o":           
            if buffer_edge=="o2":
                turns("Lw Ja Lw3")   
            if buffer_edge=="o4":
                turns("Dw2 L3 T L Dw2")
            if buffer_edge=="o6":
                turns("L T L3")
            if buffer_edge=="o8":
                turns("Lw Jb Lw3")

        if buffer_edge[0]=="w":           
            if buffer_edge=="w2":
                turns("D3 L2 T L2 D")
            if buffer_edge=="w4":
                turns("L2 T L2")
            if buffer_edge=="w6":
                turns("D2 L2 T L2 D2")  
            if buffer_edge=="w8":
                turns("D L2 T L2 D3")
        edge_swap+=1
        read_rubik(rubik)
        buffer_edge=check_sticker(buffer_edge_digit,rubik)
        code.append(buffer_edge)
        print("Number of edge swap is",edge_swap)

    return rubik
###Solve Binndford conner
def solve_corner(buffer_corner,rubik):
    corner_swap=0
    buffer_corner_digit=[1,0,0]
    buffer_corner=check_sticker(buffer_corner_digit,rubik)
    while True:
        correct=0
        for i in range(0,6):
            if solve[i][0][0] == rubik[i][0][0]:
                correct+=1
            if solve[i][0][2] == rubik[i][0][2]:
                correct+=1
            if solve[i][2][0] == rubik[i][2][0]:
                correct+=1
            if solve[i][2][2] == rubik[i][2][2]:
                correct+=1
        if correct==24:
            print("Total number of corner swap is\n",corner_swap)
            break
    
        if buffer_corner=="y1" or buffer_corner=="b1" or buffer_corner=="o3":
            for i in range(0,6):
                if  solve[i][0][0] != rubik[i][0][0]:
                    if i!=0 and i!=1 :
                        buffer_corner=rubik[i][0][0]
                        break
                if  solve[i][0][2] != rubik[i][0][2]:
                    if i!=4:
                        buffer_corner=rubik[i][0][2]
                        break
                if  solve[i][2][0] != rubik[i][2][0]:
                    buffer_corner=rubik[i][2][0]
                    break
                if  solve[i][2][2] != rubik[i][2][2]:
                    buffer_corner=rubik[i][2][2]
                    break           
        if buffer_corner[0]=="y":
            if buffer_corner=="y3":
                turns("R2 F3 Y F R2")
            if buffer_corner=="y7":
                turns("F Y F3")
            if buffer_corner=="y9":
                turns("F R Y R3 F3")
                
        if buffer_corner[0]=="b":    
            if buffer_corner=="b3":
                turns("F2 R Y R3 F2")   
            if buffer_corner=="b7":
                turns("D2 R Y R3 D2")
            if buffer_corner=="b9":
                turns("F2 Y F2")

        if buffer_corner[0]=="r":        
            if buffer_corner=="r1":
                turns("F3 D R Y R3 D3 F")
            if buffer_corner=="r3":
                turns("R3 F3 Y F R")
            if buffer_corner=="r7":
                turns("D R Y R3 D3")
            if buffer_corner=="r9":
                turns("D R2 Y R2 D3")

        if buffer_corner[0]=="g":        
            if buffer_corner=="g1":
                turns("Y")
            if buffer_corner=="g3":
                turns("R3 Y R")
            if buffer_corner=="g7":
                turns("R Y R3")
            if buffer_corner=="g9":
                turns("R2 Y R2")

        if buffer_corner[0]=="o":
            if buffer_corner=="o1":
                turns("R3 F R Y R3 F3 R")
            if buffer_corner=="o7":
                turns("D3 R Y R3 D")
            if buffer_corner=="o9":
                turns("D3 R2 Y R2 D")
                
        if buffer_corner[0]=="w":
            if buffer_corner=="w1":
                turns("F3 R Y R3 F")
            if buffer_corner=="w3":
                turns("F3 Y F")
            if buffer_corner=="w7":
                turns("D F3 R Y R3 F D3")
            if buffer_corner=="w9":
                turns("D3 F3 Y F D")
                
        corner_swap+=1
        read_rubik(rubik)
        buffer_corner=check_sticker(buffer_corner_digit,rubik)
        code.append(buffer_corner)
        print("Number of corner_swap is",corner_swap)
    return rubik

###solve
def solve_the_rubik(rubik):
    solve_center(rubik)
    solve_edge(buffer_edge,rubik)
    solve_corner(buffer_corner,rubik)
    move=0

    while len(solution)!=0:
        a=solution.pop(0)
        rubik[6].append(a)
    solution.clear()
    return rubik

def reset_rubik(rubik):
    rubik=new_rubik[new]
    return rubik

###Main
def main(rubik):
    read_rubik(rubik)
    while True:
        print("Start!!!\n")
        print("Input turn (U/L/F/R/B/D/S/M with suffix (non/w/2/3) for turn the rubik")
        print("or x/y/z for rotation the rubik")
        print("or Ja/Jb/Ra/Y for permutation the top layer")
        print("or new for random scumble")
        print("or sol for solution")
        print(rubik[6])
        a = input("Input : ")
        turns(a)
        if a== "sol":
            solve_the_rubik(rubik)
            break
        if a== "ok":
            break
        read_rubik(rubik)
        if a!="new":
            a=a.split(" ")
            for i in a:
                move.append(i)

    if __name__ == '__main__':
        import sys
        try:
            N = int(sys.argv[1])
        except:
            N = 3
        c = Cube(N)

    for a in rubik[6]:
        if a[0] in ("U","L","F","R","B","D"):
            if a in ("U","L","F","R","B","D"):
                c.rotate_face(a)
            if a in ["U2","L2","F2","R2","B2","D2"]:
                c.rotate_face(a[0])
                c.rotate_face(a[0])
            if a in ["U3","L3","F3","R3","B3","D3"]:        
                c.rotate_face(a[0], -1)
            if a in ("Uw","Lw","Fw","Rw","Bw","Dw"):
                c.rotate_face(a[0])
                c.rotate_face(a[0],layer=1)
            if a in ("Uw2","Lw2","Fw2","Rw2","Bw2","Dw2"):
                c.rotate_face(a[0])
                c.rotate_face(a[0])
                c.rotate_face(a[0],layer=1)
                c.rotate_face(a[0],layer=1)
            if a in ("Uw3","Lw3","Fw3","Rw3","Bw3","Dw3"):
                c.rotate_face(a[0],-1)
                c.rotate_face(a[0],-1,layer=1)
                
        if a[0] in ("x","y","z"):  
            if a in ("x","y","z"):
                b={"x":"R","y":"U","z":"F"}
                c.rotate_face(b[a])
                c.rotate_face(b[a],layer=1)
                c.rotate_face(b[a],layer=2)
            if a in ("x2","y2","z2"):
                b={"x":"R","y":"U","z":"F"}
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=2)
                c.rotate_face(b[a[0]],layer=2)
            if a in ("x3","y3","z3"):
                b={"x":"R","y":"U","z":"F"}
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]],-1,layer=1)
                c.rotate_face(b[a[0]],-1,layer=2)

    c.draw_interactive()
    plt.show()   
    
###Solve Binndford Pochmann Method
buffer_edge_digit=[0,1,2]
buffer_edge=check_sticker(buffer_edge_digit,rubik)
buffer_corner_digit=[1,0,0]
buffer_corner=check_sticker(buffer_corner_digit,rubik)
###Start
main(rubik)
####################################################################

