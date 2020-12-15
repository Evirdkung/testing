#----------------------------------------------------------------------
# Matplotlib Rubik's cube simulator
# Written by Jake Vanderplas
# Adapted from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets

import random


move=[]
collector=[]
code=[]
solution=[]
rubik = [[["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]]
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


"""
Sticker representation
----------------------
Each face is represented by a length [5, 3] array:
  [v1, v2, v3, v4, v1]
Each sticker is represented by a length [9, 3] array:
  [v1a, v1b, v2a, v2b, v3a, v3b, v4a, v4b, v1a]
In both cases, the first point is repeated to close the polygon.
Each face also has a centroid, with the face number appended
at the end in order to sort correctly using lexsort.
The centroid is equal to sum_i[vi].
Colors are accounted for using color indices and a look-up table.
With all faces in an NxNxN cube, then, we have three arrays:
  centroids.shape = (6 * N * N, 4)
  faces.shape = (6 * N * N, 5, 3)
  stickers.shape = (6 * N * N, 9, 3)
  colors.shape = (6 * N * N,)
The canonical order is found by doing
  ind = np.lexsort(centroids.T)
After any rotation, this can be used to quickly restore the cube to
canonical position.
"""
###
class Quaternion:
    """Quaternion Rotation:
    Class to aid in representing 3D rotations via quaternions.
    """
    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternions from unit vectors v and rotation angles theta
        Parameters
        ----------
        v : array_like
            array of vectors, last dimension 3. Vectors will be normalized.
        theta : array_like
            array of rotation angles in radians, shape = v.shape[:-1].
        Returns
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        # we don't implement multiplication by a scalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1]
                         - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0]
                         + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3]
                         + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2]
                         - prod[2, 1] + prod[3, 0])],
                       dtype=np.float,
                       order='F').T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points):
        M = self.as_rotation_matrix()
        return np.dot(points, M.T)


def project_points(points, q, view, vertical=[0, 1, 0]):
    """Project points using a quaternion q and a view v
    Parameters
    ----------
    points : array_like
        array of last-dimension 3
    q : Quaternion
        quaternion representation of the rotation
    view : array_like
        length-3 vector giving the point of view
    vertical : array_like
        direction of y-axis for view.  An error will be raised if it
        is parallel to the view.
    Returns
    -------
    proj: array_like
        array of projected points: same shape as points.
    """
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponing to vertical
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotate the points
    R = q.as_rotation_matrix()
    Rpts = np.dot(points, R.T)

    # project the points onto the view
    dpoint = Rpts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans =  list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir),
                     -np.dot(dpoint, zdir)]).transpose(trans)
###
class Cube:
    """Magic Cube Representation"""
    # define some attribues
    default_plastic_color = 'black'
    default_face_colors = ["yellow", "w",
                           "#00008f", "#009f0f",
                           "#ff6f00", "#cf0000",
                           "gray", "none"]
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    stickerwidth = 0.9
    stickermargin = 0.5 * (1. - stickerwidth)
    stickerthickness = 0.001
    (d1, d2, d3) = (1 - stickermargin,
                    1 - 2 * stickermargin,
                    1 + stickerthickness)
    base_sticker = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_sticker_centroid = np.array([[0, 0, 1 + stickerthickness]])

    # Define rotation angles and axes for the six sides of the cube
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(np.eye(3)[0], theta)
    for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(np.eye(3)[1], theta)
    for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    # define face movements
    facesdict = dict(F=z, B=-z,
                     R=x, L=-x,
                     U=y, D=-y)

    def __init__(self, N=3, plastic_color=None, face_colors=None):
        self.N = N
        if plastic_color is None:
            self.plastic_color = self.default_plastic_color
        else:
            self.plastic_color = plastic_color

        if face_colors is None:
            self.face_colors = self.default_face_colors
        else:
            self.face_colors = face_colors

        self._move_list = []
        self._initialize_arrays()

    def _initialize_arrays(self):
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, stickers, and colors
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            M = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, M.T)
            stickers_t = np.dot(factor * self.base_sticker
                                + translations, M.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, M.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid
                                         + translations, M.T)
            colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)
            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)
        self._colors = np.concatenate(colors)

        self._sort_faces()

    def _sort_faces(self):
        # use lexsort on the centroids to put faces in a standard order.
        ind = np.lexsort(self._face_centroids.T)
        self._face_centroids = self._face_centroids[ind]
        self._sticker_centroids = self._sticker_centroids[ind]
        self._stickers = self._stickers[ind]
        self._colors = self._colors[ind]
        self._faces = self._faces[ind]

    def rotate_face(self, f, n=1, layer=0):
        """Rotate Face"""
        if layer < 0 or layer >= self.N:
            raise ValueError('layer should be between 0 and N-1')

        try:
            f_last, n_last, layer_last = self._move_list[-1]
        except:
            f_last, n_last, layer_last = None, None, None

        if (f == f_last) and (layer == layer_last):
            ntot = (n_last + n) % 4
            if abs(ntot - 4) < abs(ntot):
                ntot = ntot - 4
            if np.allclose(ntot, 0):
                self._move_list = self._move_list[:-1]
            else:
                self._move_list[-1] = (f, ntot, layer)
        else:
            self._move_list.append((f, n, layer))
        
        v = self.facesdict[f]
        r = Quaternion.from_v_theta(v, n * np.pi / 2)
        M = r.as_rotation_matrix()

        proj = np.dot(self._face_centroids[:, :3], v)
        cubie_width = 2. / self.N
        flag = ((proj > 0.9 - (layer + 1) * cubie_width) &
                (proj < 1.1 - layer * cubie_width))

        for x in [self._stickers, self._sticker_centroids,
                  self._faces]:
            x[flag] = np.dot(x[flag], M.T)
        self._face_centroids[flag, :3] = np.dot(self._face_centroids[flag, :3],
                                                M.T)

    def draw_interactive(self):
        fig = plt.figure(figsize=(5, 5))
        fig.add_axes(InteractiveCube(self))
        return fig


class InteractiveCube(plt.Axes):
    def __init__(self, cube=None,
                 interactive=True,
                 view=(0, 0, 10),
                 fig=None, rect=[0, 0.16, 1, 0.84],
                 **kwargs):
        if cube is None:
            self.cube = Cube(3)
        elif isinstance(cube, Cube):
            self.cube = cube
        else:
            self.cube = Cube(cube)

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0),
                                                  -np.pi / 6)

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-2.0, 2.0)),
                           ylim=kwargs.get('ylim', (-2.0, 2.0)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy = None  # store xy position of mouse event
        self._shift = False  # shift key pressed
        self._digit_flags = np.zeros(10, dtype=bool)  # digits 0-9 pressed

        self._current_rot = self._start_rot  #current rotation state
        self._face_polys = None
        self._sticker_polys = None

        self._draw_cube()

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)
        self.figure.canvas.mpl_connect('key_release_event',
                                       self._key_release)

        self._initialize_widgets()

        # write some instructions
        self.figure.text(0.05, 0.05,
                         "Mouse/arrow keys adjust view\n"
                         "U/D/L/R/B/F keys turn faces\n"
                         "1/2 for the layer of the faces\n"
                         "(hold shift for counter-clockwise)",
                         size=10)

    def _initialize_widgets(self):
        self._ax_reset = self.figure.add_axes([0.75, 0.05, 0.2, 0.075])
        self._btn_reset = widgets.Button(self._ax_reset, 'Reset View')
        self._btn_reset.on_clicked(self._reset_view)

        self._ax_solve = self.figure.add_axes([0.55, 0.05, 0.2, 0.075])
        self._btn_solve = widgets.Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube)

    def _project(self, pts):
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self):
        stickers = self._project(self.cube._stickers)[:, :, :2]
        faces = self._project(self.cube._faces)[:, :, :2]
        face_centroids = self._project(self.cube._face_centroids[:, :3])
        sticker_centroids = self._project(self.cube._sticker_centroids[:, :3])

        plastic_color = self.cube.plastic_color
        colors = np.asarray(self.cube.face_colors)[self.cube._colors]
        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color,
                                 zorder=face_zorders[i])
                sp = plt.Polygon(stickers[i], facecolor=colors[i],
                                 zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def rotate_face(self, face, turns=1, layer=0, steps=3):
###
        if not np.allclose(turns, 0):
            for i in range(0,steps):
                self.cube.rotate_face(face, turns * 1. / steps,
                                      layer=layer)
                self._draw_cube()
                          
        
        if turns==1:
            if layer==0:
                turn(face,b=2)
            if layer==1:
                a={"U":"E3","L":"M3","F":"S3","R":"M","B":"S","D":"E"}
                turn(a[face],b=2)
            if layer==2:
                a={"U":"D3","L":"R3","F":"B3","R":"L3","B":"F3","D":"U3"}
                turn(a[face],b=2)
            
        if turns==-1:
            if layer==0:
                turn(face+"3",b=2)
            if layer==1:
                a={"U":"E","L":"M","F":"S","R":"M3","B":"S3","D":"E3"}
                turn(face,b=2)
            if layer==2:
                a={"U":"D3","L":"R3","F":"B3","R":"L3","B":"F3","D":"U3"}
                turn(a[face],b=2)
        
        
###
    def _reset_view(self, *args):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()


    def _solve_cube(self, *args,rubik_1=rubik):
        move=[]
        rubik_1[6]=[]
        
        if rubik_1[0:6]==solve[0:6]:
            print("\nThe rubik is already solve!!!")
            return rubik_1

        solve_the_rubik(rubik_1)
        
        print("\nSolution is\n",rubik_1[6])
        print("\nCode is\n",rubik_1[7])
        print("\nT_perm\tis R U R3 U3 R3 F R2 U3 R3 U3 R U R3 F3")
        print("Ja_perm\tis R U R3 F3 R U R3 U3 R3 F R2 U3 R3 U3")
        print("Jb_perm\tis R2 D R D3 R F2 Rw3 F Rw F2")
        print("Y_perm\tis F R U3 R3 U3 R U R3 F3 R U R3 U3 R3 F R F3")
        print("Ra_perm\tis R U R3 F3 R U2 R3 U2 R3 F R U R U2 R3 U3")
        
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
                   "(z3)":["Fw3","B"]}
               
                for c in b[a]:
                    sol.append(c)                
            else:
                sol.append(a)

        while len(sol)!=0:
            i = sol.pop(0)
            if i in ("U","L","F","R","B","D"):
                self.rotate_face(i)
                continue
            if i[1] in ["2","3"]:
                if i in ["U2","L2","F2","R2","B2","D2"]:
                    self.rotate_face(i[0],2)
                if i in ["U3","L3","F3","R3","B3","D3"]:        
                    self.rotate_face(i[0], -1)
            if i[1] in ["w"]:
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
        collector.clear()

        rubik.clear()
        rubik.append([["y1","y2","y3"],["y4","y5","y6"],["y7","y8","y9"]])
        rubik.append([["b1","b2","b3"],["b4","b5","b6"],["b7","b8","b9"]])
        rubik.append([["r1","r2","r3"],["r4","r5","r6"],["r7","r8","r9"]])
        rubik.append([["g1","g2","g3"],["g4","g5","g6"],["g7","g8","g9"]])
        rubik.append([["o1","o2","o3"],["o4","o5","o6"],["o7","o8","o9"]])
        rubik.append([["w1","w2","w3"],["w4","w5","w6"],["w7","w8","w9"]])
        rubik.append(move)
        rubik.append(code)

        move.clear()
        
        self.cube._move_list = []
        return rubik

    def _key_press(self, event):
        """Handler for key press events"""
        
        if event.key == 'shift':
            self._shift = True
        elif event.key.isdigit():
            self._digit_flags[int(event.key)] = 1
        elif event.key == 'right':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                5 * self._step_LR))
        elif event.key == 'left':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                -5 * self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                5 * self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                -5 * self._step_UD))
        elif event.key.upper() in 'LRUDBF':
            if self._shift:
                direction = -1
            else:
                direction = 1
            N=3
            if np.any(self._digit_flags[:N]):
                for d in np.arange(N)[self._digit_flags[:N]]:
                    self.rotate_face(event.key.upper(), direction, layer=d)
                    
            else:
                self.rotate_face(event.key.upper(), direction)       
        self._draw_cube()
        

    def _key_release(self, event):
        """Handler for key release event"""
        if event.key == 'shift':
            self._shift = False
        elif event.key.isdigit():
            self._digit_flags[int(event.key)] = 0

    def _mouse_press(self, event):
        """Handler for mouse button press"""
        self._event_xy = (event.x, event.y)
        if event.button == 1:
            self._button1 = True
        elif event.button == 3:
            self._button2 = True

    def _mouse_release(self, event):
        """Handler for mouse button release"""
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event):
        """Handler for mouse motion"""
        if self._button1 or self._button2:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)

            if self._button1:
                if self._shift:
                    ax_LR = self._ax_LR_alt
                else:
                    ax_LR = self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD,
                                               self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_LR,
                                               self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()
####################################################################################

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
    if a[0]=="U":
        if a=="U":
            U_turn(rubik)
        elif a=="U2":
            U2_turn(rubik)
        elif a=="U3":
            U3_turn(rubik)
        elif a=="Uw":
            Uw_turn(rubik)
        elif a=="Uw2":
            Uw2_turn(rubik)
        elif a=="Uw3":
            Uw3_turn(rubik)
    elif a[0]=="L":
        if a=="L":
            L_turn(rubik)
        elif a=="L2":
            L2_turn(rubik)
        elif a=="L3":
            L3_turn(rubik)
        elif a=="Lw":
            Lw_turn(rubik)
        elif a=="Lw2":
            Lw2_turn(rubik)
        elif a=="Lw3":
            Lw3_turn(rubik)
    elif a[0]=="F":
        if a=="F":
            F_turn(rubik)
        elif a=="F2":
            F2_turn(rubik)
        elif a=="F3":
            F3_turn(rubik)
        elif a=="Fw":
            Fw_turn(rubik)
        elif a=="Fw2":
            Fw2_turn(rubik)
        elif a=="Fw3":
            Fw3_turn(rubik)
    elif a[0]=="R":
        if a=="R":
            R_turn(rubik)
        elif a=="R2":
            R2_turn(rubik)
        elif a=="R3":
            R3_turn(rubik)
        elif a=="Rw":
            Rw_turn(rubik)
        elif a=="Rw2":
            Rw2_turn(rubik)
        elif a=="Rw3":
            Rw3_turn(rubik)
    elif a[0]=="B":
        if a=="B":
            B_turn(rubik)
        elif a=="B2":
            B2_turn(rubik)
        elif a=="B3":
            B3_turn(rubik)
        elif a=="Bw":
            Bw_turn(rubik)
        elif a=="Bw2":
            Bw2_turn(rubik)
        elif a=="Bw3":
            Bw3_turn(rubik)
    elif a[0]=="D":    
        if a=="D":
            D_turn(rubik)
        elif a=="D2":
            D2_turn(rubik)
        elif a=="D3":
            D3_turn(rubik)
        elif a=="Dw":
            Dw_turn(rubik)
        elif a=="Dw2":
            Dw2_turn(rubik)
        elif a=="Dw3":
            Dw3_turn(rubik)
    elif a[0] in ["M","S","E"]:    
        if a=="M":
            M_turn(rubik)
        elif a=="M2":
            M2_turn(rubik)
        elif a=="M3":
            M3_turn(rubik)
        elif a=="S":
            S_turn(rubik)
        elif a=="S2":
            S2_turn(rubik)
        elif a=="S3":
            S3_turn(rubik)
        elif a=="E":
            E_turn(rubik)
        elif a=="E2":
            E2_turn(rubik)
        elif a=="E3":
            E3_turn(rubik)
    elif a[0] in ["x","y","z"]:  
        if a=="x":
            a="(x)"
            solution.append(a)
            print(a,end=" ")
            b=2
            x_rotation(rubik)  
        elif a=="x2":
            a="(x2)"
            solution.append(a)
            print(a,end=" ")
            b=2
            x2_rotation(rubik)
        elif a=="x3":
            a="(x3)"
            solution.append(a)
            print(a,end=" ")
            b=2
            x3_rotation(rubik)
        elif a=="y":
            a="(y)"
            solution.append(a)
            print(a,end=" ")
            b=2
            y_rotation(rubik) 
        elif a=="y2":
            a="(y2)"
            solution.append(a)
            print(a,end=" ")
            b=2
            y2_rotation(rubik)
        elif a=="y3":
            a="(y3)"
            solution.append(a)
            print(a,end=" ")
            b=2
            y3_rotation(rubik)
        elif a=="z":
            a="(z)"
            solution.append(a)
            print(a,end=" ")
            b=2
            z_rotation(rubik)
        elif a=="z2":
            a="(z2)"
            solution.append(a)
            print(a,end=" ")
            b=2
            z2_rotation(rubik)
        elif a=="z3":
            a="(z3)"
            solution.append(a)
            print(a,end=" ")
            b=2
            z3_rotation(rubik)

    if a in ["T","Ja","Jb","Y","Ra"]:
        if a=="T":
            a="(T)"
            solution.append(a)
            b=2
            T_perm(rubik)    
        elif a=="Ja":
            a="(Ja)"
            solution.append(a)
            b=2
            Ja_perm(rubik)
        elif a=="Jb":
            a="(Jb)"
            solution.append(a)
            b=2
            Jb_perm(rubik)
        elif a=="Y":
            a="(Y)"
            solution.append(a)
            b=2
            Y_perm(rubik)
        elif a=="Ra":
            a="(Ra)"
            solution.append(a)
            b=2
            Ra_perm(rubik)
        
    elif a=="new":
        scramble(rubik)
        b=1
    elif a=="sol":
        b=1
        print("start")
        
    if b==0:
        solution.append(a)
    elif b==1:
        solution.clear()
    elif b==2:
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
    while i<=15:
        while c==d:
            c = random.randint(1,3)
        r = random.randint(1,6)
        if c==1:
            if r==1:
                x="U"
            elif r==2:
                x="U2"
            elif r==3:
                x="U3"
            elif r==4:
                x="D"
            elif r==5:
                x="D2"
            elif r==6:
                x="D3"
        elif c==2:
            if r==1:
                x="L"
            elif r==2:
                x="L2"
            elif r==3:
                x="L3"
            elif r==4:
                x="R"
            elif r==5:
                x="R2"
            elif r==6:
                x="R3"
        elif c==3:
            if r==1:
                x="F"
            elif r==2:
                x="F2"
            elif r==3:
                x="F3"
            elif r==4:
                x="B"
            elif r==5:
                x="B2"
            elif r==6:
                x="B3"
        turn(x)
        i+=1
        d=c
        rubik[6].append(x)
        collector.append(x)
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
            print("\nTotal number of edge swap is",edge_swap)
            if edge_swap%2==1:
                print("Have parity do Ra_perm")
                turn("Ra")
                code.append("*Ra*")
            else:
                print("No parity")
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
            elif buffer_edge=="y4":
                turns("T")
            elif buffer_edge=="y8":
                turns("Ja")

        elif buffer_edge[0]=="b":   
            if buffer_edge=="b2":
                turns("L3 Dw L3 T L Dw3 L")
            elif buffer_edge=="b4":
                turns("Dw L3 T L Dw3")
            elif buffer_edge=="b6":
                turns("Dw3 L T L3 Dw")    
            elif buffer_edge=="b8":
                turns("L Dw L3 T L Dw3 L3")

        elif buffer_edge[0]=="r":           
            if buffer_edge=="r2":
                turns("Lw3 Jb Lw")
            elif buffer_edge=="r4":
                turns("L3 T L")
            elif buffer_edge=="r6":
                turns("Dw2 L T L3 Dw2")
            elif buffer_edge=="r8":
                turns("Lw3 Ja Lw")

        elif buffer_edge[0]=="g":           
            if buffer_edge=="g4":
                turns("Dw3 L3 T L Dw") 
            elif buffer_edge=="g6":
                turns("Dw L T L3 Dw3")        
            elif buffer_edge=="g8":
                turns("D3 Lw3 Ja Lw D")

        elif buffer_edge[0]=="o":           
            if buffer_edge=="o2":
                turns("Lw Ja Lw3")   
            elif buffer_edge=="o4":
                turns("Dw2 L3 T L Dw2")
            elif buffer_edge=="o6":
                turns("L T L3")
            elif buffer_edge=="o8":
                turns("Lw Jb Lw3")

        elif buffer_edge[0]=="w":           
            if buffer_edge=="w2":
                turns("D3 L2 T L2 D")
            elif buffer_edge=="w4":
                turns("L2 T L2")
            elif buffer_edge=="w6":
                turns("D2 L2 T L2 D2")  
            elif buffer_edge=="w8":
                turns("D L2 T L2 D3")
        edge_swap+=1
#        read_rubik(rubik)
        buffer_edge=check_sticker(buffer_edge_digit,rubik)
        code.append(buffer_edge)
#        print("Number of edge swap is",edge_swap)

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
            print("Total number of corner swap is",corner_swap)
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
            elif buffer_corner=="y7":
                turns("F Y F3")
            elif buffer_corner=="y9":
                turns("F R Y R3 F3")
                
        elif buffer_corner[0]=="b":    
            if buffer_corner=="b3":
                turns("F2 R Y R3 F2")   
            elif buffer_corner=="b7":
                turns("D2 R Y R3 D2")
            elif buffer_corner=="b9":
                turns("F2 Y F2")

        elif buffer_corner[0]=="r":        
            if buffer_corner=="r1":
                turns("F3 D R Y R3 D3 F")
            elif buffer_corner=="r3":
                turns("R3 F3 Y F R")
            elif buffer_corner=="r7":
                turns("D R Y R3 D3")
            elif buffer_corner=="r9":
                turns("D R2 Y R2 D3")

        elif buffer_corner[0]=="g":        
            if buffer_corner=="g1":
                turns("Y")
            elif buffer_corner=="g3":
                turns("R3 Y R")
            elif buffer_corner=="g7":
                turns("R Y R3")
            elif buffer_corner=="g9":
                turns("R2 Y R2")

        elif buffer_corner[0]=="o":
            if buffer_corner=="o1":
                turns("R3 F R Y R3 F3 R")
            elif buffer_corner=="o7":
                turns("D3 R Y R3 D")
            elif buffer_corner=="o9":
                turns("D3 R2 Y R2 D")
                
        elif buffer_corner[0]=="w":
            if buffer_corner=="w1":
                turns("F3 R Y R3 F")
            elif buffer_corner=="w3":
                turns("F3 Y F")
            elif buffer_corner=="w7":
                turns("D F3 R Y R3 F D3")
            elif buffer_corner=="w9":
                turns("D3 F3 Y F D")
                
        corner_swap+=1
#        read_rubik(rubik)
        buffer_corner=check_sticker(buffer_corner_digit,rubik)
        code.append(buffer_corner)
#        print("Number of corner_swap is",corner_swap)
    return rubik

###solve
def solve_the_rubik(rubik):
    solve_center(rubik)
    solve_edge(buffer_edge,rubik)
    solve_corner(buffer_corner,rubik)

    while len(solution)!=0:
        a=solution.pop(0)
        rubik[6].append(a)
    solution.clear()
    return rubik

###Main
def main(rubik):
    read_rubik(rubik)
    print("\nStart!!!\n")
    while True:
        print("\nInput(U/L/F/R/B/D/S/M/E with suffix (non/w/2/3) for turn the rubik")
        print("or x/y/z for rotation the rubik")
        print("or Ja/Jb/Ra/Y/T for permutation the top layer")
        print("or new for random scumble")
        print("or sol for solution")
        print(collector)
        a = input("Input : ")
        print("\nLet Start!!!\n")
        
        if a not in ["T","Ja","Jb","Ra","Y"]:
            turns(a)
            
        b=a.split(" ")
        
        if a== "sol":
            break
        read_rubik(rubik)
        if a!="new":
            a=a.split(" ")
            for i in a:
                rubik[6].append(i)
        for i in b:
            if i in ["U","L","F","R","B","D","S","M","E",
                     "U2","L2","F2","R2","B2","D2","S2","M2","E2",
                     "U3","L3","F3","R3","B3","D3","S3","M3","E3",
                     "Uw","Lw","Fw","Rw","Bw","Dw",
                     "Uw2","Lw2","Fw2","Rw2","Bw2","Dw2",
                     "Uw3","Lw3","Fw3","Rw3","Bw3","Dw3",
                     "x","y","z","x2","y2","z2","x3","y3","z3"
                     "T","Ja","Jb","Ra","Y"]:
                collector.append(i)
            else:
                print("The invalid input :",i)


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
                continue
            elif a[1] in ["2","3"]:
                if a in ["U2","L2","F2","R2","B2","D2"]:
                    c.rotate_face(a[0])
                    c.rotate_face(a[0])
                    continue
                elif a in ["U3","L3","F3","R3","B3","D3"]:        
                    c.rotate_face(a[0], -1)
                    continue
            elif a[1] in ("w"):
                if a in ("Uw","Lw","Fw","Rw","Bw","Dw"):
                    c.rotate_face(a[0])
                    c.rotate_face(a[0],layer=1)
                    continue
                elif a in ("Uw2","Lw2","Fw2","Rw2","Bw2","Dw2"):
                    c.rotate_face(a[0])
                    c.rotate_face(a[0])
                    c.rotate_face(a[0],layer=1)
                    c.rotate_face(a[0],layer=1)
                    continue
                elif a in ("Uw3","Lw3","Fw3","Rw3","Bw3","Dw3"):
                    c.rotate_face(a[0],-1)
                    c.rotate_face(a[0],-1,layer=1)
                    continue
                
        elif a[0] in ("x","y","z"):
            b={"x":"R","y":"U","z":"F"}
            if a in ("x","y","z"):
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=2)
                continue
            elif a in ("x2","y2","z2"):
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=2)
                c.rotate_face(b[a[0]],layer=2)
                continue
            elif a in ("x3","y3","z3"):
                c.rotate_face(b[a[0]])
                c.rotate_face(b[a[0]],-1,layer=1)
                c.rotate_face(b[a[0]],-1,layer=2)
                continue
        elif a[0] in ("M","S","E"):
            b={"M":"L","S":"F","E":"D"}
            if a in ("M","S","E"):
                c.rotate_face(b[a[0]],layer=1)
                continue
            elif a in ("M2","S2","E2"):
                c.rotate_face(b[a[0]],layer=1)
                c.rotate_face(b[a[0]],layer=1)
                continue
            elif a in ("M3","S3","E3"):
                c.rotate_face(b[a[0]],-1,layer=1)
                continue              
                
    c.draw_interactive()
    plt.show()
    main(rubik)
###Solve Binndford Pochmann Method
buffer_edge_digit=[0,1,2]
buffer_edge=check_sticker(buffer_edge_digit,rubik)
buffer_corner_digit=[1,0,0]
buffer_corner=check_sticker(buffer_corner_digit,rubik)
###Start
main(rubik)
####################################################################

