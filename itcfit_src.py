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

import pyFAI, pyFAI.azimuthalIntegrator
from pyFAI.method_registry import IntegrationMethod

pyFAI.use_opencl = False

from scipy.optimize import curve_fit
from PIL import Image
import sys

import matplotlib.pyplot as plt

def xy(r,phi,x0,y0):
    return r*np.cos(phi)+x0, r*np.sin(phi)+y0

def qr_conv(config):
    EHC = 2.*np.pi*(config["energy"])*1e9/1.23984
    lam=1.23984/(config["energy"])*1e-9

    theta_pix=np.arctan(config["pixel1"]/config["dist"])
    qpix=4.*np.pi*np.sin(theta_pix)/lam
    rr=2.*EHC
    k=rr/qpix

    return k,qpix

def do_integration1d_x(img,alpha,gamma,config,k,qpix,mask=None):
    #    mask_img = np.array(Image.open("./SAXS_masks.tiff"))
    mask_img = np.zeros(img.shape,dtype=np.uint8)
    if mask is not None:
        mask_img[mask>0] = 1
#    mask_img = mask

    rnpts=1
    chinpts=500
    alpha=alpha%360
    im_w,im_h=img.shape
    x_beam=config["Beam_x"]#+0.5
    y_beam=config["Beam_y"]#+0.5
    d0 =k*np.tan(np.radians(gamma))
    qmax=config["qmax"]*1e9
    kx=qpix*0.5

    x0=d0*np.cos(np.radians(alpha))
    y0=d0*np.sin(np.radians(-alpha))

    width=config["radial_int"]

    row, col = np.ogrid[:im_w, :im_h]
    disk_mask = ((col - (x_beam+x0))**2 + (row - (y0+y_beam))**2 < (d0-width*0.5)**2)
    mask_img[disk_mask] = 1
    disk_mask = ((col - (x_beam+x0))**2 + (row - (y0+y_beam))**2 > (d0+width*0.5)**2)
    mask_img[disk_mask] = 1
    
    #plt.imshow(mask_img)
    #plt.show()
    p1=((y_beam))*config["pixel1"] # point of normal incidenc at the detector in meter
    p2=(x_beam)*config["pixel1"] #point of normal incidenc at the detector in meter

    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=config["dist"], pixel1=config["pixel1"], pixel2=config["pixel2"],poni1=p1,poni2=p2)
    ai.wavelength=12.398419292/config['energy']*1e-10

    amin=0#(alpha0-dalpha)
    amax=np.radians(360)#(alpha0+dalpha)

#    method="bbox"
    method="splitpixel"

    chinpts=np.floor((qmax/kx-1)*1.0)+1

    chi1,I1,sig = ai.integrate1d(img, chinpts, unit="q_nm^-1",method=method, radial_range=(kx*1e-9,qmax*1e-9),correctSolidAngle=True,dummy=-1,delta_dummy=2,mask=mask_img,error_model='poisson')

    return chi1,chi1,I1,sig

def do_integration1d(img,alpha,gamma,config,k,qpix,mask=None):
#    mask_img = np.array(Image.open("./SAXS_masks.tiff"))
#    mask_img = mask
    mask_img = np.zeros(img.shape,dtype=np.uint8)
    if mask is not None:
        mask_img[mask>0] = 1

    rnpts=1
    chinpts=500
    alpha=alpha%360
    im_w,im_h=img.shape
    x_beam=config["Beam_x"]#+0.5
    y_beam=config["Beam_y"]#+0.5
    d0 =k*np.tan(np.radians(gamma))
    qmax=config["qmax"]*1e9
    kx=qpix*0.5
    chinpts=np.floor((qmax/kx-1)*1.0)+1
    dalpha=np.pi*0.75
    if 4.*d0**2*kx**2 > qmax**2:
        dalpha=np.arctan2(np.sqrt(4.*d0**2*kx**2-qmax**2)*qmax/(kx**2*d0**2),(2*d0**2*kx**2-qmax**2)/(kx**2*d0**2))
    dalpha=np.degrees(dalpha)
    if dalpha > 90:
        dalpha=90

    x0=d0*np.cos(np.radians(alpha))
    y0=d0*np.sin(np.radians(alpha))

    p1=(y_beam-y0)*config["pixel1"] # point of normal incidenc at the detector in meter
    p2=(x0+x_beam)*config["pixel1"] #point of normal incidenc at the detector in meter
    d0=d0*config["pixel1"]*1e+3 # radius of the streak circle in mm
        
    width=config["radial_int"]*config["pixel1"]*1e+3*rnpts
        
    alpha0=180-alpha

    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=config["dist"], pixel1=config["pixel1"], pixel2=config["pixel2"],poni1=p1,poni2=p2)
    ai.wavelength=12.398419292/config['energy']*1e-10

    if alpha <= dalpha or alpha > (360 - dalpha):
        ai.setChiDiscAtZero()
    #        ai.setChiDiscAtPi()
    else:
        ai.setChiDiscAtPi()
    #        ai.setChiDiscAtZero()

    amin=(alpha0-dalpha)
    amax=(alpha0+dalpha)

    method="bbox"
    #method="splitpixel"
#    method=IntegrationMethod.select_one_available("no", dim=2)
#    print(method)
#    sys.exit(0)
    #res = ai._integrate2d_ng(img, rnpts, chinpts, unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(amin,amax),correctSolidAngle=False,dummy=1,delta_dummy=2,mask=mask_img,error_model = "poisson")
    
#    res = ai._integrate2d_ng(img, rnpts, chinpts, unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(amin,amax),correctSolidAngle=False,mask=mask_img,error_model = "poisson")

#    res = ai._integrate2d_ng(img, rnpts, int(chinpts), unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(amin,amax),correctSolidAngle=False,mask=mask_img,error_model = "poisson")
    res = ai._integrate2d_ng(img, rnpts, int(chinpts), unit="r_mm",method=method,radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(amin,amax),correctSolidAngle=False,mask=mask_img,error_model = "poisson",dummy=-1)

    I1 = res.intensity
    tth1 = res.radial
    chi1 = res.azimuthal
    sig1 = res.sigma

    xbeam=d0*np.cos(np.radians(alpha0))
    ybeam=d0*np.sin(np.radians(alpha0))
        
    ky=qpix/config["pixel1"]*0.5e-12

    abschi=np.copy(chi1)

    for i in range(0,len(chi1)):
        x0=d0*np.cos(np.radians(chi1[i])) # distances are all in mm here
        y0=d0*np.sin(np.radians(chi1[i]))
        abschi[i]=np.sqrt((x0-xbeam)**2+(y0-ybeam)**2)*ky
        if (chi1[i]-alpha0) < 0 or (chi1[i]-alpha0) > 360:
        # calculate the radial distance from the beam center to a point on the streak circle and convert it to q in nm-1. rtoq is in m-1 and dist is in mm which gives 1e-6 to get nm-1
            chi1[i]=-abschi[i]
        else:
            chi1[i]=abschi[i]
    column = 0 #np.argmin(np.abs(tth1-d0))
    return abschi,chi1,I1[:,column],sig1[:,column]


def func_lin(x, a, b):
    return a * x + b

def fit_circ_pyFAI(img,gamma,alpha,width,config,mask_img=None):
    alpha=alpha%360
    rnpts=100
    chinpts=40
    thesh = config['fit_thresh']
    x_beam=config['Beam_x']
    y_beam=config['Beam_y']

    k,qpix= qr_conv(config)
    
    d0 =k*np.tan(np.radians(gamma))
    qmax=config["fit_q_max"]*1e9
    kx=qpix*0.5
    dalpha=np.pi*0.75
    if 4.*d0**2*kx**2 > qmax**2:
        dalpha=np.arctan2(np.sqrt(4.*d0**2*kx**2-qmax**2)*qmax/(kx**2*d0**2),(2*d0**2*kx**2-qmax**2)/(kx**2*d0**2))
    
    dalpha=np.degrees(dalpha)
    dalpha_max=dalpha#*0.2

    qmax=config["fit_q_min"]*1e9
    dalpha=np.pi*0.75
    if 4.*d0**2*kx**2 > qmax**2:
        dalpha=np.arctan2(np.sqrt(4.*d0**2*kx**2-qmax**2)*qmax/(kx**2*d0**2),(2*d0**2*kx**2-qmax**2)/(kx**2*d0**2))
    dalpha=np.degrees(dalpha)
    dalpha_min=dalpha#*0.04

    x0=d0*np.cos(np.radians(alpha))
    y0=d0*np.sin(np.radians(alpha))

    im_w,im_h=img.shape
    
    p1=((y_beam-y0))*config["pixel1"] # point of normal incidenc at the detector in meter
    p2=(x0+x_beam)*config["pixel1"] #point of normal incidenc at the detector in meter
    d0=d0*config["pixel1"]*1e+3 # radius of the streak circle in mm

    alpha0=180-alpha

    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=config["dist"], pixel1=config["pixel1"], pixel2=config["pixel2"],poni1=p1,poni2=p2)
    
    if alpha <= dalpha or alpha > (360 - dalpha):
        ai.setChiDiscAtZero()
    else:
        ai.setChiDiscAtPi()

    im_w,im_h = img.shape
    im=np.copy(img)

    dy = np.zeros(2*chinpts,dtype=float)
    dx = np.zeros(2*chinpts,dtype=float)
    
    Ix, tthx, chix = ai._integrate2d_ng(im, rnpts, chinpts, unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(alpha0-dalpha_max,alpha0-dalpha_min),correctSolidAngle=False,mask=mask_img)
    
    xbeam=d0*np.cos(np.radians(alpha0))
    ybeam=d0*np.sin(np.radians(alpha0))

    for i in range(0,len(chix)):
        i_max=Ix[i,:].argmax()
        if Ix[i,i_max] > thesh:
            dx[i] = tthx[i_max]-d0
            x0=d0*np.cos(np.radians(chix[i])) # distances are all in mm here
            y0=d0*np.sin(np.radians(chix[i]))
            if(chix[i]-alpha0 < 0 ):
                dy[i] = -np.sqrt((x0-xbeam)**2+(y0-ybeam)**2)
            else:
                dy[i] = np.sqrt((x0-xbeam)**2+(y0-ybeam)**2)
        else:
            dx[i] = 0
            dy[i] = 0

    Ix, tthx, chix = ai._integrate2d_ng(im, rnpts, chinpts, unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(alpha0+dalpha_min,alpha0+dalpha_max),correctSolidAngle=False,dummy=-1,delta_dummy=2,mask=mask_img)

    for i in range(0,len(chix)):
        i_max=Ix[i,:].argmax()
        if Ix[i,i_max] > thesh:
            dx[i+chinpts] = tthx[i_max]-d0
            x0=d0*np.cos(np.radians(chix[i])) # distances are all in mm here
            y0=d0*np.sin(np.radians(chix[i]))
            if (chix[i]-alpha0) < 0 or (chix[i]-alpha0) > 360:
                dy[i+chinpts] = -np.sqrt((x0-xbeam)**2+(y0-ybeam)**2)
            else:
                dy[i+chinpts] = np.sqrt((x0-xbeam)**2+(y0-ybeam)**2)
        else:
            dx[i+chinpts] = 0
            dy[i+chinpts] = 0
# Constrain the optimization to the region of ``0 < a < 3``, ``0 < b < 2`` and ``0 < c < 1``:
#popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 2., 1.]))
    popt, pcov = curve_fit(func_lin, dy, dx)
    print(np.degrees(popt[0]))
    return np.degrees(np.arcsin(popt[0])),gamma,k,qpix


def show_2D_cake(img,gamma,alpha,width,config):
    rnpts=1000
    chinpts=1000
    alpha=alpha%360
    thesh = config['fit_thresh']
    x_beam=config['Beam_x']#+0.5
    y_beam=config['Beam_y']#+0.5

    EHC = 2.*np.pi*(config["energy"])*1e9/1.23984
    lam=1.23984/(config["energy"])*1e-9
    
    theta_pix=np.arctan(config["pixel1"]/config["dist"])
    qpix=4.*np.pi*np.sin(theta_pix)/lam
    rr=2.*EHC
    k=rr/qpix
    
    d0 =k*np.tan(np.radians(gamma))
    qmax=config["fit_q_max"]*1e9
    kx=qpix*0.5
    dalpha=np.pi*0.75
    if 4.*d0**2*kx**2 > qmax**2:
        dalpha=np.arctan2(np.sqrt(4.*d0**2*kx**2-qmax**2)*qmax/(kx**2*d0**2),(2*d0**2*kx**2-qmax**2)/(kx**2*d0**2))

    dalpha=np.degrees(dalpha)
    dalpha_max=dalpha#*0.2

    qmax=config["fit_q_min"]*1e9
    dalpha=np.pi*0.75
    if 4.*d0**2*kx**2 > qmax**2:
        dalpha=np.arctan2(np.sqrt(4.*d0**2*kx**2-qmax**2)*qmax/(kx**2*d0**2),(2*d0**2*kx**2-qmax**2)/(kx**2*d0**2))
    dalpha=np.degrees(dalpha)
    dalpha_min=dalpha#*0.04
    
    x0=d0*np.cos(np.radians(alpha))
    y0=d0*np.sin(np.radians(alpha))

    im_w,im_h=img.shape
    
    p1=((y0+y_beam))*config["pixel1"] # point of normal incidenc at the detector in meter
    p2=(x0+x_beam)*config["pixel1"] #point of normal incidenc at the detector in meter
    d0=d0*config["pixel1"]*1e+3 # radius of the streak circle in mm
    
    alpha0=180-alpha

    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=config["dist"], pixel1=config["pixel1"], pixel2=config["pixel2"],poni1=p1,poni2=p2)
    if alpha <= dalpha_max or alpha > (360 - dalpha_max):
        ai.setChiDiscAtZero()
#        ai.setChiDiscAtPi()
    else:
        ai.setChiDiscAtPi()
#        ai.setChiDiscAtZero()

    im_w,im_h = img.shape
    im=np.copy(img)
    
    Ix1, tthx1, chix1 = ai._integrate2d_ng(im, rnpts, chinpts, unit="r_mm",method="bbox",radial_range=(d0-width*0.5,d0+width*0.5),azimuth_range=(alpha0-dalpha_max,alpha0+dalpha_max),correctSolidAngle=False)

    return Ix1,tthx1,chix1
