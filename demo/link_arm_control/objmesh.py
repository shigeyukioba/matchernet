# -*- coding: utf-8 -*-
import os
import numpy as np
import pyglet
from pyglet.gl import *


def get_subdir_path(sub_dir):
    # Get the directory this module is located in
    abs_path_module = os.path.realpath(__file__)
    module_dir, _ = os.path.split(abs_path_module)

    dir_path = os.path.join(module_dir, sub_dir)

    return dir_path
        

def get_file_path(sub_dir, file_name):
    """
    Get the absolute path of a resource file, which may be relative to
    the agi_lab module directory, or an absolute path.

    This function is necessary because the simulator may be imported by
    other packages, and we need to be able to load resources no matter
    what the current working directory is.
    """

    # If this is already a real path
    if os.path.exists(file_name):
        return file_name

    subdir_path = get_subdir_path(sub_dir)
    file_path = os.path.join(subdir_path, file_name)

    return file_path


def load_texture(tex_path):
    img = pyglet.image.load(tex_path)
    tex = img.get_texture()

    glEnable(tex.target)
    glBindTexture(tex.target, tex.id)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        img.width,
        img.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img.get_image_data().get_data('RGBA', img.width * 4))

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    return tex


class ObjMesh(object):
    """
    Load and render wavefront .OBJ model files
    """

    # Loaded mesh files, indexed by mesh file path
    cache = {}

    @classmethod
    def get(self, mesh_name):
        """
        Load a mesh or used a cached version
        """

        # Assemble the absolute path to the mesh file
        file_path = get_file_path('data/meshes', mesh_name + '.obj')

        if file_path in self.cache:
            return self.cache[file_path]

        mesh = ObjMesh(file_path)
        self.cache[file_path] = mesh

        return mesh

    def __init__(self, file_path):
        """
        Load an OBJ model file

        Limitations:
        - only one object/group
        - only triangle faces
        """

        # Comments
        # mtllib file_name
        # o object_name
        # v x y z
        # vt u v
        # vn x y z
        # usemtl mtl_name
        # f v0/t0/n0 v1/t1/n1 v2/t2/n2

        # Attempt to load the materials library
        materials = self._load_mtl(file_path)

        mesh_file = open(file_path, 'r')

        verts = []
        texs = []
        normals = []
        faces = []
        face_mtls = []

        cur_mtl = None

        # For each line of the input file
        for line in mesh_file:
            line = line.rstrip(' \r\n')

            # Skip comments
            if line.startswith('#') or line == '':
                continue

            tokens = line.split(' ')
            tokens = map(lambda t: t.strip(' '), tokens)
            tokens = list(filter(lambda t: t != '', tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == 'v':
                vert = list(map(lambda v: float(v), tokens))
                verts.append(vert)

            if prefix == 'vt':
                tc = list(map(lambda v: float(v), tokens))
                texs.append(tc)

            if prefix == 'vn':
                normal = list(map(lambda v: float(v), tokens))
                normals.append(normal)

            if prefix == 'usemtl':
                mtl_name = tokens[0]
                cur_mtl = materials[
                    mtl_name] if mtl_name in materials else None

            if prefix == 'f':
                assert len(tokens) == 3, "only triangle faces are supported"

                face = []
                for token in tokens:
                    indices = filter(lambda t: t != '', token.split('/'))
                    indices = list(map(lambda idx: int(idx), indices))
                    assert len(indices) == 2 or len(indices) == 3
                    face.append(indices)

                faces.append(face)
                face_mtls.append(cur_mtl)

        mesh_file.close()

        self.num_faces = len(faces)

        # Create numpy arrays to store the vertex data
        list_verts = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)
        list_norms = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)
        list_texcs = np.zeros(shape=(3 * self.num_faces, 2), dtype=np.float32)
        list_color = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)

        cur_vert_idx = 0

        # For each triangle
        for f_idx, face in enumerate(faces):
            # Get the color for this face
            f_mtl = face_mtls[f_idx]
            f_color = f_mtl['Kd'] if f_mtl else np.array((1, 1, 1))

            # For each tuple of indices
            for indices in face:
                # Note: OBJ uses 1-based indexing
                # and texture coordinates are optional
                if len(indices) == 3:
                    v_idx, t_idx, n_idx = indices
                    vert = verts[v_idx - 1]
                    texc = texs[t_idx - 1]
                    normal = normals[n_idx - 1]
                else:
                    v_idx, n_idx = indices
                    vert = verts[v_idx - 1]
                    normal = normals[n_idx - 1]
                    texc = [0, 0]

                list_verts[cur_vert_idx, :] = vert
                list_texcs[cur_vert_idx, :] = texc
                list_norms[cur_vert_idx, :] = normal
                list_color[cur_vert_idx, :] = f_color

                # Move to the next vertex
                cur_vert_idx += 1

        # Recompute the object extents after centering
        self.min_coords = list_verts.min(axis=0)
        self.max_coords = list_verts.max(axis=0)

        # Create a vertex list to be used for rendering
        self.vlist = pyglet.graphics.vertex_list(
            3 * self.num_faces, ('v3f', list_verts.reshape(-1)),
            ('t2f', list_texcs.reshape(-1)), ('n3f', list_norms.reshape(-1)),
            ('c3f', list_color.reshape(-1)))

        # Load the texture associated with this mesh
        file_name = os.path.split(file_path)[-1]
        tex_name = file_name.split('.')[0]
        tex_path = get_file_path('data/textures', tex_name + '.png')

        # Try to load the texture, if it exists
        if os.path.exists(tex_path):
            self.texture = load_texture(tex_path)
        else:
            self.texture = None

    def _load_mtl(self, model_path):
        mtl_path = model_path.split('.')[0] + '.mtl'

        if not os.path.exists(mtl_path):
            return {}

        #print('loading materials from "%s"' % mtl_path)

        mtl_file = open(mtl_path, 'r')

        materials = {}
        cur_mtl = None

        # For each line of the input file
        for line in mtl_file:
            line = line.rstrip(' \r\n')

            # Skip comments
            if line.startswith('#') or line == '':
                continue

            tokens = line.split(' ')
            tokens = map(lambda t: t.strip(' '), tokens)
            tokens = list(filter(lambda t: t != '', tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == 'newmtl':
                cur_mtl = {}
                materials[tokens[0]] = cur_mtl

            if prefix == 'Kd':
                vals = list(map(lambda v: float(v), tokens))
                vals = np.array(vals)
                cur_mtl['Kd'] = vals

        mtl_file.close()
        
        return materials

    def render(self):
        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(self.texture.target, self.texture.id)
        else:
            glDisable(GL_TEXTURE_2D)

        self.vlist.draw(GL_TRIANGLES)

        glDisable(GL_TEXTURE_2D)
