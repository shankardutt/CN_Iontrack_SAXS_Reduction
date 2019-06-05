#!/bin/python
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
__date__ = "05/06/2019"
__status__ = "development"

pyITCfit_version = "0.6.0"
pyITCfit_date = "05/06/2019"

import sys, logging, json, os, time, types, threading
import os.path
import numpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyITCfit")

import pyFAI.utils
from pyFAI.utils.shell import ProgressBar

import LoadExperiments as exp
from beam_stop_mask import *

import ctypes
from PIL import Image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.debug("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")
pyFAI.use_opencl = False
from itcfit_src import *
##for debug
#import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib.figure import Figure
#from matplotlib import cm
#import numpy as np
##for debug end

def integrate_gui(options, args):
    from silx.gui import qt
    from itcfit_widget import MainITCfit
    app = qt.QApplication([])
    if args:
        file=args[0]
    else:
        file=None
    window = MainITCfit(file_path_=file,exp_path_=options.exp_path,outpath_=options.output,json_file=options.json,cakeit=options.cake,_mask_file=options.mask_file)
    window.hide_fit()
    window.show()
    return app.exec_()

def integrate_shell(options, args):
    import json
    config = json.load(open(options.json))
    if options.exp_path:
        config["exp_path"]=options.exp_path
        if config["exp_path"]:
            if config["exp_load"] == 0:
                logger.error("experimental file given but no routine specified to load this file!")
                logger.error("Set the exp_load variable in the json file to the desierd loader and try again.")
                sys.exit(-1)
            exp.LoadExperimet(config)
#   CN: check this!!!
    x_beam=config["Beam_x"]#+0.5#Is this a bad center from scatterbrain or a systematic shift?
    y_beam=config["Beam_y"]#+0.5#Is this a bad center from scatterbrain or a systematic shift?
#    ai = pyFAI.worker.make_ai(config)
#    worker = pyFAI.worker.Worker(azimuthalIntegrator=ai)
#    # TODO this will init again the azimuthal integrator, there is a problem on the architecture
#    worker.setJsonConfig(options.json)

    start_time = time.time()
    
    # Skip not existing files
    image_filenames = []
    for item in args:
        if os.path.exists(item) and os.path.isfile(item):
            image_filenames.append(item)
        else:
            logger.warning("File %s do not exists. Ignored." % item)
    image_filenames = sorted(image_filenames)

    progress_bar = ProgressBar("Integration", len(image_filenames), 20)
        
    alpha=config['alpha_ini']
    gamma=config['gamma_ini']
    tmp_alpha=alpha
    fit_r_range=config['d_radius']*config['pixel1']*1e+3
    fit_tol=config['fit_tol']

    # Integrate files one by one
    for i, item in enumerate(image_filenames):
        logger.debug("Processing %s" % item)
        # if len(item) > 100:
        message = os.path.basename(item)
#        else:
#            message = item
        progress_bar.update(i + 1, message=message)
        img=np.array(Image.open(item))
        if i == 0:
            im_w,im_h=img.shape
            cmap= cm.get_cmap(config['cmap'])
            min=config["c_min_s"]
            max=config["c_max_s"]
            fs=(im_w/150,im_h/150)
#            bs_mask = beam_stop_threshold_mask(img,img.max()-10,0,float(config["Beam_x"]),float(config["Beam_y"]),options.mask_file)
            bs_mask = beam_stop_threshold_mask(img,img.max()-10,0,float(config["Beam_x"]),float(config["Beam_y"]),bsc_size=float(config["Bsc_size"]),bs_alpha=float(config["Bs_alpha"]),bs_w2=float(config["Bs_w2"]),bs_l1=float(config["Bs_l1"]),bs_l2=float(config["Bs_l2"]),bs_x_off=float(config["Bs_x_off"]),bs_y_off=float(config["Bs_y_off"]),mask_file=options.mask_file)

        if options.output:
            if not os.path.isdir(options.output):
                os.makedirs(options.output)
            outpath = os.path.join(options.output, os.path.splitext(os.path.basename(item))[0])
        else:
            outpath = os.path.splitext(item)[0]

        gamma=config['gamma_ini']
        alpha_corr=100
        alpha=tmp_alpha
        max_iter=20
        i=0
        if options.nofit:
            k,qpix=qr_conv(config)
        else:
            while abs(alpha_corr) > fit_tol and i < max_iter:
                alpha_corr,gamma,k,qpix=fit_circ_pyFAI(img,gamma,tmp_alpha,fit_r_range,config,bs_mask)
                tmp_alpha+=alpha_corr*config['damping']
                i+=1
            if (abs(tmp_alpha-alpha) < config['alpha_delta']):
                alpha=tmp_alpha
            else:
                alpha=config['alpha_ini']
                tmp_alpha=alpha
        im_w,im_h=img.shape
        if options.thumb :
            im=np.copy(img)
            r =k*np.tan(np.radians(gamma))
            x0=r*np.cos(np.radians(alpha))+x_beam
            y0=r*np.sin(-np.radians(alpha))+y_beam
            phis=np.arange(0,6.3,0.01)
#            cmap= cm.get_cmap(config['cmap'])
#            min=config["c_min_s"]
#            max=config["c_max_s"]
#            fs=(im_w/150,im_h/150)
            figure = plt.figure(figsize=fs, dpi=150)
            subp=figure.add_subplot(111)
            tmp=subp.imshow(im,cmap=cmap)
            tmp.set_clim([min,max])
            ax=subp.axis()
            circ = subp.plot( *(xy)(r,phis,x0,y0), c='g',ls='-')
            subp.axis(ax)
            figure.savefig(outpath+"_circ.png")
            plt.close('all')
#        extent = self.full_extent(self.subp).transformed(self.figure.dpi_scale_trans.inverted())
#        figure.savefig("test.png", bbox_inches=extent)
#        plt.show()

#        abschi,chi1,I1,sig1 = do_integration1d_x(img,alpha,gamma,config,k,qpix,bs_mask)
        abschi,chi1,I1,sig1 = do_integration1d(img,alpha,gamma,config,k,qpix,bs_mask)
        data = np.array([chi1,I1,sig1]).transpose()
        np.savetxt(outpath+"_sig.xy", data, delimiter="\t",header="q\tI\tsig")
        
        if options.bkg_angle:
            alpha+=float(config["bkg_angle"])
#            abschi,chi2,I2,sig2 = do_integration1d_x(img,alpha,gamma,config,k,qpix,bs_mask)
            abschi,chi2,I2,sig2 = do_integration1d(img,alpha,gamma,config,k,qpix,bs_mask)
            I3=I1-I2
            sig3=sig1*sig1+sig2*sig2
            sig3=np.sqrt(sig3)
            data = np.array([chi2,I2,sig2]).transpose()
            np.savetxt(outpath+"_bkg.xy", data, delimiter="\t",header="q\tI\tsig")
            data = np.array([chi2,I3,sig3]).transpose()
            np.savetxt(outpath+"_sub.xy", data, delimiter="\t",header="q\tI\tsig")
#        if multiframe:
#            writer = HDF5Writer(outpath + "_pyFAI.h5")
#            writer.init(config)
#
#            for i in range(img.nframes):
#                data = img.getframe(i).data
#                if worker.do_2D():
#                    res = worker.process(data)
#                else:
#                    res = worker.process(data)
#                    res = res.T[1]
#                writer.write(res, index=i)
#                writer.close()
#        else:
#        if worker.do_2D():
#            filename = outpath + ".azim"
#        else:
#            filename = outpath + ".dat"
#        data = img.data
#        writer = DefaultAiWriter(filename, worker.ai)
#        if worker.do_2D():
#            worker.process(data, writer=writer)
#        else:
#            worker.process(data, writer=writer)
#        writer.close()
    progress_bar.clear()
    logger.info("Processing done in %.3fs !" % (time.time() - start_time))
    return 0


if __name__ == "__main__":
    usage = "pyITCfit [options] file1.tif file2.tif ..."
#    version = "pyITCfit version %s from %s" % (pyFAI.version, pyFAI.date)
    version = "pyITCfit version %s from %s" % (pyITCfit_version, pyITCfit_date)
    description = """
        pyITCfit is a graphical interface (based on Python/Qt4) to perform a specialized azimuthal
        integration for ion track diffraction pattern."""
    epilog = """pyITCfit saves all parameters in a .azimint.json (hidden) file. This JSON file
        is an ascii file which can be edited and used to configure other plugins like pyFAI.
        
        Nota: there is bug in debian6 making the GUI crash (to be fixed inside pyqt)
        http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348"""
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-v", "--verbose",action="store_true",
                        dest="verbose", default=False,
                        help="switch to verbose/debug mode")
    parser.add_argument("-o", "--output",
                        dest="output", default=None,
                        help="Directory where to store the output data")
    parser.add_argument("-t", "--thumb",
                        dest="thumb", default=False,action="store_true",
                        help="Save tumbnails of the fit as png files.")
#    parser.add_argument("-s", "--save",
#                        dest="save_prog", default=False,action="store_true",
#                        help="Save status of the fit in the config file.")
    parser.add_argument("-b", "--bkg",
                        dest="bkg_angle",default=False,action="store_true",
                        help="Generates a background file with a fixed offset angle relative to the fitted circle. The angle needs to be set in the config file.")
    parser.add_argument("-e", "--exp",
                        dest="exp_path", default=None,
                        help="File to read experimental settings from. The correct subroutie has to be configured in the config file")
    
#    parser.add_argument("-f", "--format",
#                        dest="format", default=None,
#                        help="output data format (can be HDF5)")
#    parser.add_argument("-s", "--slow-motor",
#                        dest="slow", default=None,
#                        help="Dimension of the scan on the slow direction (makes sense only with HDF5)")
#    parser.add_argument("-r", "--fast-motor",
#                        dest="rapid", default=None,
#                        help="Dimension of the scan on the fast direction (makes sense only with HDF5)")
    parser.add_argument("--no-gui",
                        dest="gui", default=True, action="store_false",
                        help="Process the dataset without showing the user interface.")
    parser.add_argument("--no-fit",
                        dest="nofit", default=False, action="store_true",
                        help="run the Process without fitting.")
    parser.add_argument("-j", "--json",
                        dest="json", default=".itcfit.json",
#                        dest="json", default=".azimint.json",
                        help="Configuration file containing the processing to be done")
    parser.add_argument("-m", "--mask",
                        dest="mask_file", default=None,
                        help="Tiff file contaning the mask applied to the image.")
    parser.add_argument("args", metavar='FILE', type=str, nargs='*',
                        help="Files to be integrated")
    parser.add_argument("-c", "--cake",
                        dest="cake", default=False,action="store_true",
                        help="activates the cake view in the gui.")
    options = parser.parse_args()
                        
                        # Analysis arguments and options
    args = pyFAI.utils.expand_args(options.args)
    
    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)

    if options.gui:
        result = integrate_gui(options, args)
    else:
        result = integrate_shell(options, args)
    sys.exit(result)
