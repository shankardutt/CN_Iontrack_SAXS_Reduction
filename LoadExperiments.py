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
__date__ = "23/05/2019"
__status__ = "development"

from xml.etree import cElementTree as ElementTree

import ntpath

from pyFAI.io.ponifile import PoniFile

class scatterBrainRead:
    
    def __init__(self,dictionary):
        self.dictionary=dictionary
        self.filename=dictionary["exp_path"]
        self.tree=ElementTree.parse(self.filename)
        elem=self.tree.getroot()

        iter=elem.getiterator('Parameters')
        
        for parent in iter:
            for child in parent:
                if(child.tag == "CONFIGURATION"):
                    if(child.get('NAME') == 'SAXS'):
                        camera_defs=child.find("CAMERADEFS")

                        self.dictionary["dist"]=float(camera_defs.get('LENGTH'))*1e-3
                        self.dictionary["Beam_x"]=float(camera_defs.get('BEAMX'))
                        self.dictionary["Beam_y"]=float(1679)-float(camera_defs.get('BEAMY'))
#                        self.dictionary['energy_wavelength_box']=0
#                        self.dictionary['energy_wavelength']=12.3984/float(camera_defs.get('WAVELENGTH'))
                        self.dictionary["energy"]=12.3984/float(camera_defs.get('WAVELENGTH'))
    def get_experiment(self):
        return self.dictionary

class pyFAIRead:
     def __init__(self,dictionary):
        self.dictionary=dictionary
        self.filename=dictionary["exp_path"]
        poniFile = PoniFile()
        poniFile.read_from_file(self.filename)
        self.dictionary["dist"]=float(poniFile.dist)
        self.dictionary["Beam_x"]=float(poniFile.poni2)/float(dictionary["pixel2"])
        self.dictionary["Beam_y"]=float(poniFile.poni1)/float(dictionary["pixel1"])
        self.dictionary["energy"]=12.3984/(float(poniFile.wavelength)*1e10)

     def get_experiment(self):
        return self.dictionary

class LoadExperimet:
    def __init__(self,dictionary):
        '''generall class to read experiemt: you can add a call to your own class here to be called on load'''
        self.loader=None
        self.dictionary=dictionary
        self.load_exp()

    def get_experiment(self):
        return self.loader.get_experiment()
    
    def load_exp(self):
        #add your class to load your parameter at the end of the elif statements
        if(self.dictionary['exp_load'] == 0):
            print('set parameters manualy')
        elif(self.dictionary['exp_load'] == 1):
            print('loading scatterBrain xml file')
            self.loader=scatterBrainRead(self.dictionary)
        elif(self.dictionary['exp_load'] == 2):
            print('loading pyFAI poni file')
            self.loader=pyFAIRead(self.dictionary)

#    def load_exp_file(self,filename):
#        if(self.loader != None):
#            print('load or find setting for file:'+ filename)
#            self.loader.load_exp_file(filename)

