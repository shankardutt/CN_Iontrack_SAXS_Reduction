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

import os
from PyQt5.uic import loadUiType
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import *

def get_ui_file_ITC(filename):
    """get the full path of a user-interface file
        
        :return: the full path of the ui
        """
    return os.path.join(".", filename)
#    return _get_data_path_ITC(os.path.join("gui", filename))

UIC = get_ui_file_ITC("Detector_Param.ui")

QDialog, QMainWindow = loadUiType(UIC)

class DetectorParams(QMainWindow, QDialog):
    def __init__(self, config, parent=None):
        super(DetectorParams,self).__init__(parent)
        self.setupUi(self)
        self.config=config
        self.energy=None
        if config is not None:
            self.dsb_Energy.setValue(self.config["energy"])
            self.dsb_pixel1.setValue(self.config["pixel1"]*1e6)
            self.dsb_x_beam.setValue(self.config["Beam_x"])
            self.dsb_y_beam.setValue(self.config["Beam_y"])
            self.dsb_dist.setValue(self.config["dist"]*1e3)
            self.energy=self.config["energy"]
            self.dist=self.config["dist"]*1e3
            self.Beam_x=self.config["Beam_x"]
            self.Beam_y=self.config["Beam_y"]
            self.pixel=config["pixel1"]*1e6
        self.radio_E.setChecked(True)
        self.radio_W.setChecked(False)
        self.energy_state=self.radio_E.text()
        self.e_label.setText("Energy (keV)")
        self.bg = QButtonGroup()
        self.bg.addButton(self.radio_E,1)
        self.bg.addButton(self.radio_W,2)
        self.bg.buttonClicked[QAbstractButton].connect(self.btngroup)
        self.buttonBox.accepted.connect(self.ok)

    def ok(self):
        if self.radio_E.isChecked():
            self.energy=float(self.dsb_Energy.text())
        else:
            self.energy=12.345/float(self.dsb_Energy.text())
        self.config["energy"]=self.energy
        self.config["pixel1"]=float(self.dsb_pixel1.text())*1e-6
        self.config["pixel2"]=float(self.dsb_pixel1.text())*1e-6
        self.config["Beam_x"]=float(self.dsb_x_beam.text())
        self.config["Beam_y"]=float(self.dsb_y_beam.text())
        self.config["dist"]=float(self.dsb_dist.text())*1e-3
    def btngroup(self,btn):
        if self.energy_state != btn.text():
            self.energy_state=btn.text()
            self.e_label.setText(btn.text())
            if btn.text() == "Wavelength (nm)":
                self.dsb_Energy.setValue(12.345/self.energy)
            else:
                self.dsb_Energy.setValue(self.energy)
    # get current date and time from the dialog
    def dateTime(self):
        return 0
    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getParams(config,parent = None):
        dialog = DetectorParams(config,parent)
        result = dialog.exec_()
        return (dialog.config, result == dialog.Accepted)


if __name__ == '__main__':
    import sys
    from PyQt5 import QtGui
    from silx.gui import qt
    
#    app = QtGui.QApplication(sys.argv)
    app = qt.QApplication(sys.argv)

    config={
            "energy":10.0,
            "dist":7.000,
            "Beam_x":500,
            "Beam_y":400,
            "pixel1":0.000172,
            "pixel2":0.000172
            }
    data,ok = DetectorParams.getParams(config)
    if ok:
        config=data
        print (config)
#    date, time,ok = DateDialog.getDateTime()
    sys.exit(0)
    sys.exit(app.exec_())
