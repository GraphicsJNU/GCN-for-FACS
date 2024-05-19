import torch
import json
import numpy as np
import os.path as osp

from pytorch3d.io import load_obj, save_obj
# from Utils.ICTFaceKit.Scripts import face_model_io
import face_model_io


def load_data(path):
    obj_dict = load_obj(path)
    verts, faces = obj_dict[0].tolist(), obj_dict[1].verts_idx.tolist()

    # convert to torch
    verts = torch.tensor(np.ascontiguousarray(verts)).float()
    faces = torch.tensor(np.ascontiguousarray(faces)).long()

    return {"verts": verts, "faces": faces, "uv": obj_dict[2].verts_uvs}

def load_facs(path):
    with open(path, 'r') as f:
        facs_dict = json.load(f)
    
    exp_coeffs = facs_dict["expression_coefficients"]
    id_coeffs = facs_dict["identity_coefficients"]

    # convert to torch
    exp_coeffs = torch.tensor(np.ascontiguousarray(exp_coeffs)).float()
    id_coeffs = torch.tensor(np.ascontiguousarray(id_coeffs)).float()

    return {"expression_coefficients": exp_coeffs, "identity_coefficients": id_coeffs}

def save_model_from_removed_vertex(file_path, save_path):
    face_indices = [idx for idx in range(9409)]
    head_and_neck_indices = [idx for idx in range(9409, 11248)]
    mouse_socket_indices = [idx for idx in range(11248, 13294)]
    eye_socket_indices = [idx for idx in range(13294, 14062)]
    gums_and_tongue_indices = [idx for idx in range(14062, 17039)]
    teeth_indices = [idx for idx in range(17039, 21451)]
    eyeball_indices = [idx for idx in range(21451, 24591)]
    lacrimal_fluid_indices = [idx for idx in range(24591, 24999)]
    eye_blend_indices = [idx for idx in range(24999, 25047)]
    eye_occlusion_indices = [idx for idx in range(25047, 25351)]
    eyelashes_indices = [idx for idx in range(25351, 26719)]
    delete_candidate = [mouse_socket_indices,
                        eye_socket_indices,
                        gums_and_tongue_indices,
                        teeth_indices,
                        eyeball_indices,
                        lacrimal_fluid_indices,
                        eye_blend_indices,
                        eye_occlusion_indices,
                        eyelashes_indices]

    vert_indices_to_delete = []
    for candidate in delete_candidate:
        vert_indices_to_delete.extend(candidate)

    vert_indices_to_delete.sort(key=lambda x: x, reverse=False)

    obj_dict = load_obj(file_path)
    verts, faces = np.ascontiguousarray(obj_dict[0].tolist()), np.ascontiguousarray(obj_dict[1].verts_idx.tolist())
    if len(vert_indices_to_delete) > 0:
        vert_indices = np.array([True for _ in range(verts.shape[0])], dtype=np.bool_)
        vert_indices[vert_indices_to_delete] = False
        vert_indices = np.arange(verts.shape[0])[vert_indices]
        in_face = np.sum(np.isin(faces, vert_indices).astype(np.int8), axis=1) == 3
        face_indices = np.array([i for i in range(faces.shape[0])], dtype=np.int32)[in_face]
        verts = verts[vert_indices, :]
        faces = faces[face_indices, :]

    verts = torch.tensor(verts).float()
    faces = torch.tensor(faces)
    
    save_obj(save_path, verts, faces)
    
def recon_face_model(model, device, dataset, input_id, input_exp, facs_path, save_path, save_file_name):
    data = dataset.get(input_id*3 + input_exp).to(device)
    facs = load_facs(facs_path)
    
    pred = model(data)

    face_model = face_model_io.load_face_model('./Utils/ICTFaceKit/FaceXModel')
    face_model.from_coefficients(facs["identity_coefficients"], pred.cpu().squeeze(0).numpy())

    # Deform the mesh
    face_model.deform_mesh()

    # Write the deformed mesh
    face_model_io.write_deformed_mesh( save_path + save_file_name, face_model)
    
def get_vertices_edge(faces):
    vertices_edge = [[], []]

    def append(n1, n2):
        vertices_edge[0].append(n1)
        vertices_edge[1].append(n2)

        vertices_edge[0].append(n2)
        vertices_edge[1].append(n1)

    for i in range(faces.shape[0]):
        face = faces[i]
        append(face[0], face[1])
        append(face[1], face[2])
        append(face[2], face[0])

    return torch.tensor(vertices_edge, dtype=torch.int64)
    
if __name__ == '__main__':
    base_path = "./experiments/auto_encoder/data/FACS"
    for i in range(100):
        file_path = osp.join(base_path, "sample_identity_{}_0.obj".format(i))
        save_path = file_path
        save_model_from_removed_vertex(file_path, save_path)
        for j in range(1, 3):
            file_path = osp.join(base_path, "sample_identity_{}_{}.obj".format(i, j))
            save_path = file_path
            save_model_from_removed_vertex(file_path, save_path)