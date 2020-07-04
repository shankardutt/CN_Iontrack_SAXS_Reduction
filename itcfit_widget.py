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

from PyQt5.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar)

from silx.gui import qt
from pyFAI.utils import float_, int_, str_, get_ui_file

from mod.drawmask import *

import os,sys
import os.path as op
import numpy as np
import json
from PIL import Image
from matplotlib import cm
import matplotlib as mpl

from itcfit_src import *
from DetectorParams import DetectorParams as DetPar
import LoadExperiments as exp
from beam_stop_mask import *

import fabio

if sys.version_info[0] >= 3:
    unicode = str

def get_ui_file_ITC(filename):
    """get the full path of a user-interface file
        
        :return: the full path of the ui
        """
    return os.path.join(".", filename)
#    return _get_data_path_ITC(os.path.join("gui", filename))

UIC = get_ui_file_ITC("ITCfit_view.ui")

Ui_MainWindow, QMainWindow = loadUiType(UIC)

class MainITCfit(QMainWindow, Ui_MainWindow):
    def __init__(self,file_path_=None,exp_path_=None,outpath_=None,json_file=None,cakeit=False,_mask_file=None):
        super(MainITCfit,self).__init__()
        self.setupUi(self)
                
        self.cakeit=cakeit
        self.json_file=json_file
        self.fig_dict = {}
        self.img=None
        self.img_orig=None
        self.file_path=None
        self.exp_path=None
        self.exp_load=None
        self.outpath=outpath_
        self.active_fig=None
        self.last_exp_path=None
        self.subp=None
        self.k=None
        self.qpix=None
        self.config=None
        self.bs_mask=None
        self.mask_file=_mask_file

        default_dict={
                        "dist": 1.0,
                        "energy": 12.0,
                        "c_max_s": 1000.0,
                        "outpath": "/",
                        "c_max": 1000.0,
                        "gamma_delta": 0,
                        "pixel2": 0.000172,
                        "pixel1": 0.000172,
                        "fit_tol": 0.1,
                        "radial_int": 20,
                        "c_min": 0.0,
                        "fit_thresh": 200.0,
                        "bkg_angle": 10.0,
                        "damping": 0.5,
                        "c_min_s": 0.0,
                        "exp_path": " ",
                        "val_dummy": -1,
                        "d_radius": 200,
                        "fit_q_min": 0.8,
                        "alpha_ini": 144,
                        "delta_dummy": 1,
                        "exp_load": 0,
                        "fit_q_max": 2,
                        "alpha_delta": 10,
                        "qmax": 6.0,
                        "save_img": True,
                        "file_path": " ",
                        "cmap": "jet",
                        "gamma_ini": 3.55,
                        "Beam_x": 587,
                        "Beam_y": 591,
                        "Bsc_size" : 16,
                        "Bs_alpha": -59.5,
                        "Bs_w2": 9,
                        "Bs_l1": 40,
                        "Bs_l2": 5000,
                        "Bs_x_off": 0,
                        "Bs_y_off": 0
                    }
        default_dict["file_path"] = os.path.join(os.getcwd(),'test_data','a-SiO2_01_0003.tif')
        default_dict["outpath"] = os.path.join(os.getcwd(),'test_data_reduced')

        ok=False
        if self.json_file is not None:
            print ("loading parameter ", self.json_file)
            ok=self.restore(self.json_file)
        if not ok:
                print ("loading default itcfit.json")
                ok=self.restore(".itcfit.json")
        if not ok:
            print ("no file to load, using the default dict")
            self.set_config(default_dict)
        print ("json_file: ", self.json_file)

        ok=False
        if file_path_:
            if op.isfile(file_path_):
                self.file_path=file_path_
                ok=True
        if not ok:
            if self.config["file_path"]:
                if op.isfile(self.config["file_path"]):
                    self.file_path=self.config["file_path"]
                    ok=True
        print ("file_path: ",self.file_path)

        ok=False
        self.exp_load=self.config["exp_load"]
        if self.exp_load !=0:
            if exp_path_:
                if op.isfile(exp_path_):
                    self.exp_path=exp_path_
                    ok=True
            if not ok:
                if self.config["exp_path"]:
                    if op.isfile(self.config["exp_path"]):
                        self.exp_path=self.config["exp_path"]
                        ok=True
        if not ok:
            self.exp_load=0
        print ("exp_path: ",self.exp_path)

        self.k,self.qpix = qr_conv(self.config)
        self.phis=np.arange(0,6.3,0.01)

        self.exp_cb.addItem("ScatterBrain")
        self.exp_cb.addItem("pyFAI")
        self.exp_cb.currentIndexChanged.connect(self.exp_cb_change)
        self.cb_logscale.setChecked(True)
        self.cb_logscale.stateChanged.connect(self.update_view)
        self.cb_log_x_scale.stateChanged.connect(self.update_view)
        self.dsb_alpha.valueChanged.connect(self.update_view)
        self.dsb_gamma.valueChanged.connect(self.update_view)
        self.dsb_bkg_angle.editingFinished.connect(self.update_view)
        self.dsb_qmax.editingFinished.connect(self.update_view)
        self.dsb_radial_int.editingFinished.connect(self.update_view)
        self.exp_btn.pressed.connect(self.exp_btn_handl)
        self.mask_btn.pressed.connect(self.mask_btn_handl)
        self.subfolder_btn.pressed.connect(self.subfolder_btn_handl)
        self.file_path_btn.pressed.connect(self.file_path_btn_handl)
        self.fit_it_btn.pressed.connect(self.fit_it_btn_handl)
        self.mplfigs.itemClicked.connect(self.changefig)
        self.btn_reset_clim.clicked.connect(self.btn_reset_handl)
        self.qfile_path.editingFinished.connect(self.qfile_path_handl)
        self.dsb_max.editingFinished.connect(self.dsb_max_handl)#CN todo: change to pressed enter
        self.dsb_min.editingFinished.connect(self.dsb_min_handl)
#        self.slider_max.sliderReleased.connect(self.slider_max_handl)
        self.slider_max.valueChanged.connect(self.slider_max_handl)
        self.slider_min.valueChanged.connect(self.slider_max_handl)
        self.save_btn.pressed.connect(self.save_config)
        self.save_as_btn.pressed.connect(self.save_as_config)
        self.sub_folder_cb.stateChanged.connect(self.sub_folder_cb_handl)
        self.qexp_path.setEnabled(False)
        fig = Figure()
        self.addmpl((fig,None))
        if self.exp_path:
            self.last_exp_path=self.exp_path
#            self.qexp_path.setText(self.exp_path)
            self.exp_path=None
        self.exp_cb_change(self.exp_load)
        if self.file_path:
            self.qfile_path.setText(self.file_path)
            if '.tif' not in self.file_path:
                img = fabio.edfimage.EdfImage()
                img.read(self.file_path)
                self.img_orig = np.copy(img.data)#,'float64')
            else:
                self.img_orig=np.array(Image.open(self.file_path))
            
            self.img=np.copy(self.img_orig)
            self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),bsc_size=float(self.config["Bsc_size"]),bs_alpha=float(self.config["Bs_alpha"]),bs_w2=float(self.config["Bs_w2"]),bs_l1=float(self.config["Bs_l1"]),bs_l2=float(self.config["Bs_l2"]),bs_x_off=float(self.config["Bs_x_off"]),bs_y_off=float(self.config["Bs_y_off"]),mask_file=self.mask_file)
            self.img[self.bs_mask > 0]=0
            self.update_view()
        if self.outpath:
            self.sub_folder_path.setText(self.outpath)
        self.exp_cb.setCurrentIndex(self.exp_load)
        self.sub_folder_cb_handl()

    def hide_fit(self):
        self.dsb_fit_q_max.hide()
        self.label_4.hide()
        self.dsb_fit_q_min.hide()
        self.label_5.hide()
        self.dsb_fit_thresh.hide()
        self.label_6.hide()                
        self.dsb_damping.hide()
        self.label_10.hide()                
        self.dsb_fit_tol.hide()
        self.label_8.hide()                
        self.dsb_d_radius.hide()
        self.label_12.hide()                
        self.fit_it_btn.hide()
        
    def update_view(self):
        self.k,self.qpix = qr_conv(self.config)
        self.cmap= cm.get_cmap(self.config['cmap'])
        if not self.active_fig and self.img is not None:#initzialize figures
            fig1  = Figure()
            ax1f1 = fig1.add_subplot(111)
            self.subp=ax1f1
            im1=ax1f1.imshow(self.img,zorder=1,cmap=self.cmap)
            fig1.colorbar(im1)
            ax=ax1f1.axis()
            im_w,im_h=self.img.shape
            x_beam=self.config["Beam_x"]#+0.5
            y_beam=float(self.config["Beam_y"])#+0.5
            k=self.k
            alpha = float(self.dsb_alpha.text())
            gamma = float(self.dsb_gamma.text())
            r =k*np.tan(np.radians(gamma))
            x0=r*np.cos(np.radians(alpha))+x_beam
            y0=r*np.sin(-np.radians(alpha))+y_beam
            self.circ1 = ax1f1.plot( *(xy)(r,self.phis,x0,y0), c='r',ls='-',zorder=2)
            self.circ3 = ax1f1.plot( *(xy)(r+0.5*int(self.dsb_radial_int.text()),self.phis,x0,y0), c='r',ls='-',zorder=4)
            self.circ4 = ax1f1.plot( *(xy)(r-0.5*int(self.dsb_radial_int.text()),self.phis,x0,y0), c='r',ls='-',zorder=5)
            
            alpha+=float(self.dsb_bkg_angle.text())
            x0=r*np.cos(np.radians(alpha))+x_beam
            y0=r*np.sin(-np.radians(alpha))+y_beam
            self.circ2 = ax1f1.plot( *(xy)(r,self.phis,x0,y0), c='g',ls='-',zorder=3)
            ax1f1.axis(ax)
            self.addfig('orig. image', [fig1,im1,float(self.config["c_max"]),float(self.config["c_min"]),float(self.config["c_max_s"]),float(self.config["c_min_s"])])
            if self.cakeit:
                fig5  = Figure()
                ax1f5 = fig5.add_subplot(111)
                im5=ax1f5.imshow(self.img,zorder=1)
                self.addfig('cake', [fig5,im5,float(self.config["c_max"]),float(self.config["c_min"]),float(self.config["c_max_s"]),float(self.config["c_min_s"])])
            abschi,chi,I1d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
            I1d[I1d<0]=1
            alpha+=float(self.dsb_bkg_angle.text())
            abschi2,chi2,I2d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
            I2d[I2d<0]=1
            fig2= Figure()
            ax1f2 = fig2.add_subplot(111)
            ax1f2.plot(chi,np.power(I1d,0.2))
            self.addfig('1d reduced', [fig2,None])
            fig3= Figure()
            ax1f3 = fig3.add_subplot(111)
            ax1f3.plot(abschi,np.power(I1d,0.2))
            self.addfig('1d reduced |q|', [fig3,None])
            fig4= Figure()
            ax1f4 = fig4.add_subplot(111)
#            ax1f4.plot(chi,np.power(I2d,0.2))
            ax1f4.plot(chi,I2d)
            self.addfig('1d sub', [fig4,None])
            fig5= Figure()
            ax1f5 = fig5.add_subplot(111)
            ax1f5.plot(abschi,np.power(I2d,0.2))
            self.addfig('1d sub |q|', [fig5,None])
            if self.cb_logscale.isChecked():
#                im1.set_norm(mpl.colors.LogNorm())
                ax1f2.set_yscale('log')
                ax1f3.set_yscale('log')
                ax1f4.set_yscale('log')
                ax1f5.set_yscale('log')
            self.rmmpl()
            self.addmpl(self.fig_dict['orig. image'])
            self.active_fig='orig. image'
        elif self.img is not None:
            if self.active_fig == 'orig. image':
                alpha = float(self.dsb_alpha.text())
                gamma = float(self.dsb_gamma.text())
                fig=self.fig_dict['orig. image'][0]
                im=self.fig_dict['orig. image'][1]
#                print(im.get_extent())
#                print(self.img.shape)
#                im.set_extent((-0.5,self.img.shape[1]-0.5,self.img.shape[0]-0.5,-0.5))
#                print(im.get_extent())
                self.subp.lines.pop(0)
                self.subp.lines.pop(0)
                self.subp.lines.pop(0)
                self.subp.lines.pop(0)
                
                ax=self.subp.axis()
                
#                if self.cb_logscale.isChecked():
#                    im.set_norm(mpl.colors.LogNorm())
#                else:
#                    im.set_norm(mpl.colors.LogNorm())
                im_w,im_h=self.img.shape
                x_beam=self.config["Beam_x"]#+0.5
                y_beam=self.config["Beam_y"]#+0.5
                k=self.k
                r =k*np.tan(np.radians(gamma))
                x0=r*np.cos(np.radians(alpha))+x_beam
                y0=r*np.sin(-np.radians(alpha))+y_beam
                self.circ1 = self.subp.plot( *(xy)(r,self.phis,x0,y0), c='r',ls='-',zorder=2)
                self.circ3 = self.subp.plot( *(xy)(r+0.5*int(self.dsb_radial_int.text()),self.phis,x0,y0), c='r',ls='-',zorder=4)
                self.circ4 = self.subp.plot( *(xy)(r-0.5*int(self.dsb_radial_int.text()),self.phis,x0,y0), c='r',ls='-',zorder=5)
            
                alpha+=float(self.dsb_bkg_angle.text())
                x0=r*np.cos(np.radians(alpha))+x_beam
                y0=r*np.sin(-np.radians(alpha))+y_beam
                self.circ2 = self.subp.plot( *(xy)(r,self.phis,x0,y0), c='g',ls='-',zorder=3)
                self.subp.axis(ax)
                if fig.canvas:
                    fig.canvas.draw()
            else:
                self.get_config()
                alpha=float(self.dsb_alpha.text())
                gamma=float(self.dsb_gamma.text())
                if self.active_fig == '1d reduced':
                    fig = self.fig_dict['1d reduced'][0]
                    abschi,chi,I1d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I1d[I1d<=0]=1
                    pchi=chi
                    pI=I1d
                    pI_err=I1d*0
                elif self.active_fig == '1d reduced |q|':
                    fig = self.fig_dict['1d reduced |q|'][0]
#                    abschi,chi,I1d,pI_err = do_integration1d_x(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    abschi,chi,I1d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I1d[I1d<=0]=1
                    pchi=abschi
                    pI=I1d
                elif self.active_fig == '1d sub':
                    fig = self.fig_dict['1d sub'][0]
                    abschi,chi,I1d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I1d[I1d<=0]=1
                    alpha+=float(self.dsb_bkg_angle.text())
                    abschi,chi,I2d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I2d[I2d<=0]=1
                    pchi=chi
                    pI=I1d-I2d
                    pI_err=I1d*0
                elif self.active_fig == '1d sub |q|':
                    fig = self.fig_dict['1d sub |q|'][0]
                    abschi,chi,I1d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I1d[I1d<=0]=1
                    alpha+=float(self.dsb_bkg_angle.text())
                    abschi,chi,I2d,_ = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                    I2d[I2d<=0]=1
                    pchi=abschi
                    pI=I1d-I2d
                    pI_err=I1d*0
                #ax1 = fig.add_subplot(111)
                ax1 = fig.gca()
                
                ax1.lines.pop(0)
                #                ax1.plot(pchi,np.power(pI,0.2),c='b')
                ax1.plot(pchi,pI,c='b')
                #ax1.errorbar(pchi,pI,yerr=pI_err, fmt='o',c='b')
                if self.cb_logscale.isChecked():
                    ax1.set_yscale('log')
                else:
                    ax1.set_yscale('linear')
                if self.cb_log_x_scale.isChecked():
                    ax1.set_xscale('log')
                else:
                    ax1.set_xscale('linear')
                if fig.canvas:
                    fig.canvas.draw()

    def load_img(self,file_path):
        if file_path != self.file_path:
            if op.isfile(file_path):
                self.file_path=file_path
                self.get_config()
                if '.tif' not in self.file_path:
                    img = fabio.edfimage.EdfImage()
                    img.read(self.file_path)
                    self.img_orig = np.copy(img.data)#,'float64')
                else:
                    self.img_orig=np.array(Image.open(self.file_path))
                self.img=np.copy(self.img_orig)
#                self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),self.mask_file)
                self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),bsc_size=float(self.config["Bsc_size"]),bs_alpha=float(self.config["Bs_alpha"]),bs_w2=float(self.config["Bs_w2"]),bs_l1=float(self.config["Bs_l1"]),bs_l2=float(self.config["Bs_l2"]),bs_x_off=float(self.config["Bs_x_off"]),bs_y_off=float(self.config["Bs_y_off"]),mask_file=self.mask_file)
                self.img[self.bs_mask > 0]=0
                if self.active_fig is not None:
                    self.fig_dict['orig. image'][1].set_data(self.img)
                    self.fig_dict['orig. image'][0].gca().set_autoscale_on(True)
                    self.fig_dict['orig. image'][1].set_extent((-0.5,self.img.shape[1]-0.5,self.img.shape[0]-0.5,-0.5))
#                    self.fig_dict['orig. image'][0].gca().set_autoscale_on(False)
            self.update_view()
        self.qfile_path.setText(self.file_path)
        
    def file_path_btn_handl(self):
        file_path,_=(qt.QFileDialog.getOpenFileName())
        if file_path:
            self.load_img(file_path)
            
    def qfile_path_handl(self):
        self.load_img(str_(self.qfile_path.text()))
#            self.update_1d(img)
#            self.mplfigs.setCurrentRow(0)
#            item=self.mplfigs.selectedItems()[0]
#            self.changefig(item)

    def fit_it_btn_handl(self):
        self.get_config()
        alpha_corr=100
        alpha=float(self.dsb_alpha.text())
        gamma=float(self.dsb_gamma.text())
        max_iter=20
        i=0
        img=self.fig_dict['orig. image'][1].get_array()
        fit_tol=self.config["fit_tol"]
        fit_r_range=self.config['d_radius']*self.config['pixel1']*1e+3
        while abs(alpha_corr) > fit_tol and i < max_iter:
            alpha_corr,gamma,k,qpix=fit_circ_pyFAI(img,gamma,alpha,fit_r_range,self.config,self.bs_mask)
            
            alpha+=alpha_corr*self.config['damping']
            if alpha_corr == 90:
                print('error while fitting')
                i = max_iter
            print ("a=",alpha_corr)
            #print ("al=",alpha)
            #print ("d=",self.config['damping'])
            print ("i=",i)
            i+=1
        
        alpha=alpha%360
        if alpha < 0:
            alpha+=360
        self.dsb_alpha.setValue(alpha)
        self.update_view()
        
    def sub_folder_cb_handl(self):
        if self.sub_folder_cb.isChecked():
            self.sub_folder_path.setEnabled(True)
            self.subfolder_btn.setEnabled(True)
        else:
            self.sub_folder_path.setEnabled(False)
            self.subfolder_btn.setEnabled(False)
            
    def exp_load_set(self,a):
        self.exp_load=int(a)
        
    def set_config(self, dico):
        """Setup the widget from its description
            
            :param dico: dictionary with description of the widget
            :type dico: dict
        """
        self.config=dico
        setup_data = {
                    "exp_load": lambda a: self.exp_load_set(a),
                    "exp_path": self.qexp_path.setText,
                    "file_path": self.qfile_path.setText,
                    "outpath": self.sub_folder_path.setText,
                    "save_img": self.sub_folder_cb.setChecked,
                    "alpha_ini": self.dsb_alpha.setValue,
                    "gamma_ini": self.dsb_gamma.setValue,
                    "fit_q_max": self.dsb_fit_q_max.setValue,
                    "fit_q_min": self.dsb_fit_q_min.setValue,
                    "fit_thresh": self.dsb_fit_thresh.setValue,
                    "fit_tol": self.dsb_fit_tol.setValue,
                    "qmax": self.dsb_qmax.setValue,
                    "radial_int": self.dsb_radial_int.setValue,
                    "damping": self.dsb_damping.setValue,
                    "bkg_angle": self.dsb_bkg_angle.setValue,
                    "d_radius": self.dsb_d_radius.setValue,
                    "c_min": self.dsb_min.setValue,
                    "c_max": self.dsb_max.setValue
#                    "c_max_s": self.slider_max.setValue,
#                    "c_min_s": self.slider_min.setValue
                    #"poni": self.poni.setText,
                    # "detector": self.all_detectors[self.detector.getCurrentIndex()],
                    #"wavelength": lambda a: self.wavelength.setText(str_(a)),
                    #"splineFile": lambda a: self.splineFile.setText(str_(a)),
                    #"pixel1": lambda a: self.pixel1.setText(str_(a)),
                    #"pixel2": lambda a: self.pixel2.setText(str_(a)),
                    #"dist": lambda a: self.dist.setText(str_(a)),
                    #"poni1": lambda a: self.poni1.setText(str_(a)),
                    #"poni2": lambda a: self.poni2.setText(str_(a)),
                    #"rot1": lambda a: self.rot1.setText(str_(a)),
                    #"rot2": lambda a: self.rot2.setText(str_(a)),
                    #"rot3": lambda a: self.rot3.setText(str_(a)),
                    #"do_dummy": self.do_dummy.setChecked,
                    #"do_dark": self.do_dark.setChecked,
                    #"do_flat": self.do_flat.setChecked,
                    #"do_polarization": self.do_polarization.setChecked,
                    #"val_dummy": lambda a: self.val_dummy.setText(str_(a)),
                    #"delta_dummy": lambda a: self.delta_dummy.setText(str_(a)),
                    #"do_mask": self.do_mask.setChecked,
                    #"mask_file": lambda a: self.mask_file.setText(str_(a)),
                    #"dark_current": lambda a: self.dark_current.setText(str_(a)),
                    #"flat_field": lambda a: self.flat_field.setText(str_(a)),
                    #"polarization_factor": self.polarization_factor.setValue,
                    #"nbpt_rad": lambda a: self.nbpt_rad.setText(str_(a)),
                    #"do_2D": self.do_2D.setChecked,
                    #"nbpt_azim": lambda a: self.nbpt_azim.setText(str_(a)),
                    #"chi_discontinuity_at_0": self.chi_discontinuity_at_0.setChecked,
                    #"do_radial_range": self.do_radial_range.setChecked,
                    #"do_azimuthal_range": self.do_azimuthal_range.setChecked,
                    #"do_poisson": self.do_poisson.setChecked,
                    #"radial_range_min": lambda a: self.radial_range_min.setText(str_(a)),
                    #"radial_range_max": lambda a: self.radial_range_max.setText(str_(a)),
                    #"azimuth_range_min": lambda a: self.azimuth_range_min.setText(str_(a)),
                    #"azimuth_range_max": lambda a: self.azimuth_range_max.setText(str_(a)),
                    #"do_solid_angle": self.do_solid_angle.setChecked
                    }
        for key, value in setup_data.items():
            if key in dico and (value is not None):
                value(dico[key])
                
    def restore(self,filename):
        """Restore from JSON file the status of the current widget
            
            :param filename: path where the config was saved
            :type filename: str
            """
#        logger.debug("Restore from %s", filename)
        if not op.isfile(filename):
#            logger.error("No such file: %s", filename)
            return False
        data = json.load(open(filename))
        self.set_config(data)
        return True
    
    def save_as_config(self):
#        logger.debug("save_config")
        self.get_config()
        json_file,_ = (qt.QFileDialog.getSaveFileName(caption="Save configuration as json",
                                                        directory=self.outpath,
                                                        filter="Config (*.json)"))
        if json_file:
            self.json_file=json_file
            self.dump(json_file)
            self.dump(".itcfit.json")
            
    def save_config(self):
        self.get_config()
        if self.sub_folder_cb.isChecked():
            self.outpath=self.sub_folder_path.text()
            print ("saveing to:", self.outpath)
            if not os.path.isdir(self.outpath):#making the subfolder, if it doesn't exist
                os.makedirs(self.outpath)
            alpha = float(self.dsb_alpha.text())
            gamma = float(self.dsb_gamma.text())
#            abschi,chi1,I1,sig1 = do_integration1d_x(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
            abschi,chi1,I1,sig1 = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
#            chi1 = I1 = sig1 = np.arange(0.0,5.0,1.0)
#            data = np.array(zip(chi1,I1,sig1))
            data = np.array([chi1,I1,sig1]).transpose()
 
            basename=os.path.basename(self.file_path)
            np.savetxt(self.outpath+"/"+basename+"_sig.xy", data, delimiter="\t",header="q\tI\tsig")
            if self.config["bkg_angle"]!=0:
                print ("bkg=",self.config["bkg_angle"])
                alpha+=float(self.config["bkg_angle"])
#                abschi,chi2,I2,sig2 = do_integration1d_x(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                abschi,chi2,I2,sig2 = do_integration1d(self.img,alpha,gamma,self.config,self.k,self.qpix,self.bs_mask)
                I3=I1-I2
                sig3=sig1*sig1+sig2*sig2
                sig3=np.sqrt(sig3)
#                data = np.array(zip(chi2,I2,sig2))
                data = np.array([chi2,I2,sig2]).transpose()
                np.savetxt(self.outpath+"/"+basename+"_bkg.xy", data, delimiter="\t",header="q\tI\tsig")
#                data = np.array(zip(chi2,I3,sig3))
                data = np.array([chi2,I3,sig3]).transpose()
                
                np.savetxt(self.outpath+"/"+basename+"_sub.xy", data, delimiter="\t",header="q\tI\tsig")
        self.dump()
#        self.dump("itcfit.json")

    def get_config(self):
        """Read the configuration of the plugin and returns it as a dictionary
            
            :return: dict with all information.
            """
        self.config["alpha_ini"]=float(self.dsb_alpha.text())
        self.config["gamma_ini"]=float(self.dsb_gamma.text())
        self.config["qmax"]=float(self.dsb_qmax.text())
        self.config["fit_q_max"]=float(self.dsb_fit_q_max.text())
        self.config["fit_q_min"]=float(self.dsb_fit_q_min.text())
        self.config["exp_load"]=self.exp_cb.currentIndex()
        if self.config["exp_load"] == 0:
            self.config["exp_path"]=None
        else:
            self.config["exp_path"]=self.qexp_path.text()
        self.config["file_path"]=self.qfile_path.text()
        self.config["outpath"]=self.sub_folder_path.text()
        self.config["save_img"]=self.sub_folder_cb.isChecked()
        self.config["fit_thresh"]=float(self.dsb_fit_thresh.text())
        self.config["fit_tol"]=float(self.dsb_fit_tol.text())
        self.config["radial_int"]=int(self.dsb_radial_int.text())
        self.config["d_radius"]=int(self.dsb_d_radius.text())
        self.config["damping"]=float(self.dsb_damping.text())
        self.config["bkg_angle"]=float(self.dsb_bkg_angle.text())
        self.config["c_max"]=float(self.dsb_max.text())
        self.config["c_min"]=float(self.dsb_min.text())
        self.config["c_max_s"]=float(self.slider_max.value())
        self.config["c_min_s"]=float(self.slider_min.value())
#        to_save = {
#                "poni": str_(self.poni.text()).strip(),
#                "detector": str_(self.detector.currentText()).lower(),
#                "wavelength": self._float("wavelength", None),
#                "splineFile": str_(self.splineFile.text()).strip(),
#                "pixel1": self._float("pixel1", None),
#                "pixel2": self._float("pixel2", None),
#                "dist": self._float("dist", None),
#                "poni1": self._float("poni1", None),
#                "poni2": self._float("poni2", None),
#                "rot1": self._float("rot1", None),
#                "rot2": self._float("rot2", None),
#                "rot3": self._float("rot3", None),
#                "do_dummy": bool(self.do_dummy.isChecked()),
#                "do_mask": bool(self.do_mask.isChecked()),
#                "do_dark": bool(self.do_dark.isChecked()),
#                "do_flat": bool(self.do_flat.isChecked()),
#                "do_polarization": bool(self.do_polarization.isChecked()),
#                "val_dummy": self._float("val_dummy", None),
#                "delta_dummy": self._float("delta_dummy", None),
#                "mask_file": str_(self.mask_file.text()).strip(),
#                "dark_current": str_(self.dark_current.text()).strip(),
#                "flat_field": str_(self.flat_field.text()).strip(),
#                "polarization_factor": float_(self.polarization_factor.value()),
#                "nbpt_rad": int_(self.nbpt_rad.text()),
#                "do_2D": bool(self.do_2D.isChecked()),
#                "nbpt_azim": int_(self.nbpt_azim.text()),
#                "chi_discontinuity_at_0": bool(self.chi_discontinuity_at_0.isChecked()),
#                "do_solid_angle": bool(self.do_solid_angle.isChecked()),
#                "do_radial_range": bool(self.do_radial_range.isChecked()),
#                "do_azimuthal_range": bool(self.do_azimuthal_range.isChecked()),
#                "do_poisson": bool(self.do_poisson.isChecked()),
#                "radial_range_min": self._float("radial_range_min", None),
#                "radial_range_max": self._float("radial_range_max", None),
#                "azimuth_range_min": self._float("azimuth_range_min", None),
#                "azimuth_range_max": self._float("azimuth_range_max", None),
#                "do_OpenCL": bool(self.do_OpenCL.isChecked())
#                }
        return self.config
    
    def dump(self, filename=None):
        """
        Dump the status of the current widget to a file in JSON
        
        :param filename: path where to save the config
        :type filename: string
        :return: dict with configuration
        """
        to_save = self.get_config()
        if filename is None:
            filename = '.itcfit.json' #self.json_file
        print ("saving to: ",filename)
        if filename is not None:
#            logger.info("Dump to %s", filename)
            try:
                with open(filename, "w") as myFile:
                    json.dump(to_save, myFile, indent=4)
            except IOError as error:
                print ("Error while saving config: ", error)
#                logger.error("Error while saving config: %s", error)
            else:
                print ("Saved")
#                logger.debug("Saved")
        return to_save

    def update_cake(self):
        alpha = float(self.dsb_alpha.text())
        gamma = float(self.dsb_gamma.text())
        fig=self.fig_dict['cake'][0]
        im=self.fig_dict['cake'][1]
#        img=self.fig_dict['orig. image'][1].get_array()
        fit_r_range=self.config['d_radius']*self.config['pixel1']*1e+3
        I1,th1,chi1=show_2D_cake(self.img,gamma,alpha,fit_r_range,self.config)
        im.set_data(I1)
#        if self.active_fig != self.fig_dict['orig. image']:
#            img=self.fig_dict['orig. image'][1].get_array()
#            self.update_1d(img)
#            item=self.mplfigs.selectedItems()[0]
#            self.changefig(item)
        if fig.canvas:
            fig.canvas.draw()
    def subfolder_btn_handl(self):
        file_path=(qt.QFileDialog.getExistingDirectory(self, "Select Directory"))
        print (file_path)
        if file_path:
            self.outpath=file_path
            self.sub_folder_path.setText(self.outpath)
    def load_exp(self,exp_path):
        self.config["exp_path"]=exp_path
        self.config["exp_load"]=self.exp_load
        loader=exp.LoadExperimet(self.config)
        self.exp_path=exp_path # do this only if success
        self.k,self.qpix = qr_conv(self.config)
        if self.img_orig is not None:
            self.img=np.copy(self.img_orig)
            self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),bsc_size=float(self.config["Bsc_size"]),bs_alpha=float(self.config["Bs_alpha"]),bs_w2=float(self.config["Bs_w2"]),bs_l1=float(self.config["Bs_l1"]),bs_l2=float(self.config["Bs_l2"]),bs_x_off=float(self.config["Bs_x_off"]),bs_y_off=float(self.config["Bs_y_off"]),mask_file=self.mask_file)
            #beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),mask_file=self.mask_file)
            self.img[self.bs_mask > 0]=0
            if self.active_fig is not None:
                self.fig_dict['orig. image'][1].set_data(self.img)

    def mask_btn_handl(self):
        if self.img_orig is not None:
            window = MaskImageWidget.runx(self.img_orig,self)

    def set_mask(self,mask):
        self.pyFAI_mask = np.copy(mask)
        if self.img_orig is not None:
            self.img=np.copy(self.img_orig)
            self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),bsc_size=float(self.config["Bsc_size"]),bs_alpha=float(self.config["Bs_alpha"]),bs_w2=float(self.config["Bs_w2"]),bs_l1=float(self.config["Bs_l1"]),bs_l2=float(self.config["Bs_l2"]),bs_x_off=float(self.config["Bs_x_off"]),bs_y_off=float(self.config["Bs_y_off"]),mask_file=None,pyFAI_mask=self.pyFAI_mask)

            self.img[self.bs_mask > 0]=0
            if self.active_fig is not None:
                self.fig_dict['orig. image'][1].set_data(self.img)
        self.update_view()

    def exp_btn_handl(self):
        if self.exp_cb.currentIndex() != 0:
            if self.exp_cb.currentIndex() == 1:
                exp_path,_ = (qt.QFileDialog.getOpenFileName(caption="Open Experiment configuration",
                                                           filter="Config (*.xml)"))#str_(qt.QFileDialog.getOpenFileName())
            if self.exp_cb.currentIndex() == 2:
                exp_path,_ = (qt.QFileDialog.getOpenFileName(caption="Open Experiment configuration",
                                                             filter="Config (*.poni)"))#str_(qt.QFileDialog.getOpenFileName())
            
            if exp_path:
                self.last_exp_path=self.exp_path
                self.load_exp(exp_path)
                self.qexp_path.setText(exp_path)
        else:
            data,ok=DetPar.getParams(self.config)
            if ok:
                self.config=data
                self.exp_path=None
        if self.img_orig is None:
            return
        self.img=np.copy(self.img_orig)
        self.bs_mask = beam_stop_threshold_mask(self.img,self.img.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),bsc_size=float(self.config["Bsc_size"]),bs_alpha=float(self.config["Bs_alpha"]),bs_w2=float(self.config["Bs_w2"]),bs_l1=float(self.config["Bs_l1"]),bs_l2=float(self.config["Bs_l2"]),bs_x_off=float(self.config["Bs_x_off"]),bs_y_off=float(self.config["Bs_y_off"]),mask_file=self.mask_file)
        #beam_stop_threshold_mask(self.img,self.img_orig.max()-10,0,float(self.config["Beam_x"]),float(self.config["Beam_y"]),mask_file=self.mask_file)
        self.img[self.bs_mask > 0]=0
        if self.active_fig is not None:
            self.fig_dict['orig. image'][1].set_data(self.img)
        self.update_view()
    def exp_cb_change(self,i):
        self.exp_load=i
        if i == 0:
            self.last_exp_path=self.exp_path
            self.qexp_path.setText("<--- select experiment loader or use button to set parameter manualy --->")
        elif self.last_exp_path:
                if self.last_exp_path != self.exp_path:
                    self.load_exp(self.last_exp_path)
                    self.update_view()
                self.qexp_path.setText(self.exp_path)
    def slider_max_handl(self):
        if self.active_fig:
            max=float(self.slider_max.value())
            min=float(self.slider_min.value())
            #print(self.active_fig)
            #self.fig_dict[self.active_fig][4]=max
            #self.fig_dict[self.active_fig][5]=min
            #self.fig_dict[self.active_fig][1].set_clim([min,max])
            self.fig_dict['orig. image'][4]=max
            self.fig_dict['orig. image'][5]=min
            self.fig_dict['orig. image'][1].set_clim([min,max])
            self.canvas.draw()
    def dsb_max_handl(self):
        max = float(self.dsb_max.text())
        min = float(self.dsb_min.text())
        curr_max=self.slider_max.value()
        curr_min=self.slider_min.value()
        if max < curr_max:
            curr_max=max*0.5
            self.slider_max.setValue(curr_max)
        if curr_min > max:
            curr_min=max*0.5
            self.slider_min.setValue(curr_min)
        self.slider_min.setRange(min,max)
        self.slider_max.setRange(min,max)
    def dsb_min_handl(self):
        max = float(self.dsb_max.text())
        min = float(self.dsb_min.text())
        curr_max=self.slider_max.value()
        curr_min=self.slider_min.value()
        if min > curr_min:
            curr_min=min
            self.slider_min.setValue(curr_min)
        if curr_max < min:
            curr_max=min
            self.slider_max.setValue(curr_max)
        self.slider_min.setRange(min,max)
        self.slider_max.setRange(min,max)
    def btn_reset_handl(self):
        if self.active_fig and self.fig_dict[self.active_fig][1]:
            max=self.img.max()
            min=self.img.min()
            self.dsb_max.setValue(max)
            self.dsb_min.setValue(min)
            self.slider_min.setRange(min,max)
            self.slider_max.setRange(min,max)
    def changefig(self,item):
        text = unicode(item.text())
        self.rmmpl()
        self.addmpl(self.fig_dict[text])
        self.active_fig=text
        self.update_view()
    def addfig(self, name, fig):
        self.fig_dict[name] = fig
        self.mplfigs.addItem(name)
    def addmpl(self,fig):
        self.canvas = FigureCanvas(fig[0])
        self.mplvl.addWidget(self.canvas)
        if fig[1]:
#            max=fig[1].get_array().max()
            max=fig[2]
            min=fig[3]
            actual_max=fig[4]
            actual_min=fig[5]
            self.dsb_max.setValue(max)
            self.dsb_min.setValue(min)
            self.slider_max.setRange(min,max)
            self.slider_max.setValue(actual_max)
            self.slider_min.setRange(min,max)
            self.slider_min.setValue(actual_min)
            fig[1].set_clim([actual_min,actual_max])
        
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
#        self.addToolBar(self.toolbar)
    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np
    from PIL import Image
    
    file_path_string = 'a-SiO2_03_0020.tif'
    img=np.array(Image.open(file_path_string))
    fig1  = Figure()
    ax1f1 = fig1.add_subplot(111)
    im1=ax1f1.imshow(img)
    fig1.colorbar(im1)
#    ax1f1.plot(np.random.rand(5))

    fig2 = Figure()
    ax1f2 = fig2.add_subplot(121)
    ax1f2.plot(np.random.rand(5))
    ax2f2 = fig2.add_subplot(122)
    ax2f2.plot(np.random.rand(10))

    fig3 = Figure()
    axf3 = fig3.add_subplot(111)
    axf3.pcolormesh(np.random.rand(20,20))

    fig4 = Figure()
    ax1f4 = fig4.add_subplot(111)
    ax1f4.plot(np.random.rand(5))
    
    app = QtGui.QApplication(sys.argv)
    main = MainITCfit()
    #main.addmpl(fig1)
#    main.addfig('orig. image', (fig1,im1,img.max(),img.min(),1000,0))
    main.addfig('orig. image', [fig1,im1,float(2000),float(img.min()),float(1000),float(0)])
#    main.addfig('circle', fig3)
    main.addfig('cake', (fig3,None))
    main.addfig('1d reduced', (fig4,None))
    main.addfig('1d reduced |q|', (fig2,None))
    main.show()
#    input()
#    main.rmmpl()
#    main.addmpl(fig2)
    sys.exit(app.exec_())
