# -*- coding: utf-8 -*-
#
#    Project: Fitting and Intergation routine speciallised for azimuthal integration of ion track diffraction pattern
#
#    Copyright (C) Australian National University, Canberra, Australia
#
#    Authors: Christian Notthoff <christian.notthoff@anu.edu.au>
#             Patrick Kluth <patrick.kluth@anu.edu.au>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
    pyITCfit is a script that allows automated data reduction
    for Small Angle Scattering of ion tracks.
    """
__author__ = "Christian Notthoff, Patrick Kluth"
__contact__ = "christian.notthoff@anu.edu.au"
__license__ = "GPLv3+"
__copyright__ = "Australian National University, Canberra, Australia"
__date__ = "31/05/2019"
__status__ = "development"

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import sys

from PIL import Image

def beam_stop_threshold_mask(img,maxValue,minValue,x_beam,y_beam,bsc_size,bs_alpha,bs_w2,bs_l1,bs_l2,bs_x_off,bs_y_off,mask_file=None,pyFAI_mask=None):
    if mask_file:
        mask = np.array(Image.open(mask_file))
    else:
        mask = None
    if pyFAI_mask is not None:
        mask = pyFAI_mask
    selectionMask = np.zeros(img.shape,dtype=np.uint8)
    tmpData = np.array(img, copy=True)
#    tmpData[True - np.isfinite(img)] = maxValue
    tmpData[np.logical_not(np.isfinite(img))] = maxValue
    selectionMask[tmpData >= maxValue] = 1
    selectionMask[tmpData <= minValue] = 1

    x_beam=np.copy(x_beam)
    y_beam=np.copy(y_beam)
    x_beam=x_beam+bs_x_off
    y_beam=y_beam+bs_y_off

    im_w,im_h = img.shape
    row, col = np.ogrid[:im_w, :im_h]

    disk_mask = ((col - x_beam)**2 + (row - y_beam)**2 < (bsc_size)**2)
    selectionMask[disk_mask]=1

    e_x1=np.cos(np.radians(bs_alpha))
    e_x2=np.sin(np.radians(bs_alpha))
    e_y1=np.cos(np.radians(bs_alpha+90))
    e_y2=np.sin(np.radians(bs_alpha+90))

    bs_w1=bsc_size #center_mask

    x0=(bs_w1*e_x1+x_beam,bs_w1*e_x2+(y_beam))
    x1=((bs_w2-bs_w1)*e_x1+x0[0]+bs_l1*e_y1,(bs_w2-bs_w1)*e_x2+bs_l1*e_y2+x0[1])
    x2=(x1[0]+e_y1*bs_l2,x1[1]+e_y2*bs_l2)
    x3=(x2[0]-e_x1*2*bs_w2,x2[1]-e_x2*2*bs_w2)
    x4=(x3[0]-bs_l2*e_y1,x3[1]-bs_l2*e_y2)
    x5=(-bs_w1*e_x1+x_beam,-bs_w1*e_x2+(y_beam))

    poly_verts1 = [x0,x1,x2,x3,x4,x5]

    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    points = np.vstack((X,Y)).T

    path = Path(poly_verts1)
    grid = path.contains_points(points)
    grid = grid.reshape(img.shape)

    selectionMask[grid > 0]=1
    if mask_file:
        selectionMask[mask > 0]=1
    if pyFAI_mask is not None:
        selectionMask[pyFAI_mask > 0]=1
    return selectionMask

if __name__ == "__main__":
    img =np.array(Image.open("./a-SiO2_03_0020.tif"))
    
    maxValue=1000
    minValue=10
    x_beam=587.184
    y_beam=452.159

    selectionMask = beam_stop_threshold_mask(img,maxValue,minValue,x_beam,y_beam)

    img[selectionMask > 0]=10000
#    img[mask > 0]=0

    plt.imshow(img)
    plt.show()
