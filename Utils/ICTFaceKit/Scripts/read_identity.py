##########################################################################################
#                                                                                        #
# ICT FaceKit                                                                            #
#                                                                                        #
# Copyright (c) 2020 USC Institute for Creative Technologies                             #
#                                                                                        #
# Permission is hereby granted, free of charge, to any person obtaining a copy           #
# of this software and associated documentation files (the "Software"), to deal          #
# in the Software without restriction, including without limitation the rights           #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell              #
# copies of the Software, and to permit persons to whom the Software is                  #
# furnished to do so, subject to the following conditions:                               #
#                                                                                        #
# The above copyright notice and this permission notice shall be included in all         #
# copies or substantial portions of the Software.                                        #
#                                                                                        #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR             #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,               #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE            #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                 #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,          #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE          #
# SOFTWARE.                                                                              #
##########################################################################################

"""Example script that reads an identity and writes its mesh.
"""

import face_model_io
import sys
import os.path as osp

parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(parent_dir)

from utils import save_model_from_removed_vertex


""" def main():
    //Reads an ICT FaceModel .json file and writes its mesh.

    # Create a new FaceModel and load the model
    id_coeffs, ex_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    face_model = face_model_io.load_face_model('../FaceXModel')
    face_model.from_coefficients(id_coeffs, ex_coeffs)

    # Deform the mesh
    face_model.deform_mesh()

    # Write the deformed mesh
    face_model_io.write_deformed_mesh('../sample_data_out/sample_identity.obj', face_model)
 """
 
def main():
    """Reads an ICT FaceModel .json file and writes its mesh.
    """
    # Create a new FaceModel and load the model
    face_model = face_model_io.load_face_model('./Utils/ICTFaceKit/FaceXModel')
    
    # for i in range(10):
    #     for j in range(10):
    #         id_coeffs, ex_coeffs = face_model_io.read_coefficients('../sample_data/sample_coeffs_' +str(i) +'_'+ str(j) + '.json')
    #         face_model.from_coefficients(id_coeffs, ex_coeffs)

    #         # Deform the mesh
    #         face_model.deform_mesh()

    #         # Write the deformed mesh
    #         face_model_io.write_deformed_mesh('../sample_data_out/sample_identity_' +str(i) +'_'+ str(j) + '.obj', face_model)
    
    id_coeffs, ex_coeffs = face_model_io.read_coefficients('./Data/FACS/raw/test_coeffs_40_10_1.json')
    face_model.from_coefficients(id_coeffs, ex_coeffs)

    # Deform the mesh
    face_model.deform_mesh()

    # Write the deformed mesh
    file_dir = './Data/FACS/raw/test_identity_40_10_1.obj'
    face_model_io.write_deformed_mesh(file_dir, face_model)
    save_model_from_removed_vertex(file_dir, file_dir)

 
if __name__ == '__main__':
    main()
