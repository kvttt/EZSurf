import vedo
import nibabel as nib
import numpy as np
import torch


def show(vol_fn: str, lh_surf_fn: str, rh_surf_fn: str, show_pts: bool = False, show_mesh: bool = True) -> None:
	"""
	Overlay a surface point cloud and mesh on its corresponding volume.

	:param vol_fn: filename of the volume.
	:param lh_surf_fn: filename of the left hemisphere surface.
	:param rh_surf_fn: filename of the right hemisphere surface.
	:param show_pts: whether to show the point cloud.
	:param show_mesh: whether to show the mesh.
	:return: None

	"""

	vol = nib.load(vol_fn)
	vol_arr = vedo.Volume(vol.get_fdata())
	vol_arr.alpha([0.0, 0.1, 0.2])

	lh_mesh = vedo.load(lh_surf_fn)
	v_tmp = lh_mesh.vertices
	v_tmp = np.hstack([v_tmp, np.ones((v_tmp.shape[0], 1))])
	lh_in = v_tmp.dot(np.linalg.inv(vol.affine).T)[:, :3]

	rh_mesh = vedo.load(rh_surf_fn)
	v_tmp = rh_mesh.vertices
	v_tmp = np.hstack([v_tmp, np.ones((v_tmp.shape[0], 1))])
	rh_in = v_tmp.dot(np.linalg.inv(vol.affine).T)[:, :3]

	if show_pts and not show_mesh:
		indices = np.random.choice(len(lh_in), 10000, replace=False)
		lh_point_cloud = vedo.Points(lh_in[indices], c='b', r=5)
		indices = np.random.choice(len(rh_in), 10000, replace=False)
		rh_point_cloud = vedo.Points(rh_in[indices], c='g', r=5)

		vedo.show(lh_point_cloud, rh_point_cloud, vol_arr, bg='white', axes=True)
	elif show_mesh and not show_pts:
		lh_mesh = vedo.Mesh([lh_in, lh_mesh.cells])
		rh_mesh = vedo.Mesh([rh_in, rh_mesh.cells])

		vedo.show(vol_arr, lh_mesh, rh_mesh, bg='white', axes=True)
	elif show_pts and show_mesh:
		indices = np.random.choice(len(lh_in), 10000, replace=False)
		lh_point_cloud = vedo.Points(lh_in[indices], c='b', r=5)
		indices = np.random.choice(len(rh_in), 10000, replace=False)
		rh_point_cloud = vedo.Points(rh_in[indices], c='g', r=5)

		lh_mesh = vedo.Mesh([lh_in, lh_mesh.cells])
		rh_mesh = vedo.Mesh([rh_in, rh_mesh.cells])

		vedo.show(vol_arr, lh_mesh, rh_mesh, lh_point_cloud, rh_point_cloud, bg='white', axes=True)


def show_arr(vol: torch.Tensor, lh_surf: torch.Tensor, rh_surf: torch.Tensor) -> None:
	"""
	Overlay a surface point cloud on its corresponding volume.

	:param vol: volume tensor of shape 1 x 1 x H x W x D.
	:param lh_surf: left hemisphere surface tensor of shape N x 3.
	:param rh_surf: right hemisphere surface tensor of shape N x 3.
	:return: None

	:note: All input tensors are assumed to be in the voxel coordinate system.

	"""

	vol_arr = vedo.Volume(vol[0, 0].cpu().numpy())
	vol_arr.alpha([0.0, 0.1, 0.2])

	lh_in = lh_surf.cpu().numpy()
	indices = np.random.choice(len(lh_in), 10000, replace=False)
	lh_point_cloud = vedo.Points(lh_in[indices], c='b', r=5)
	rh_in = rh_surf.cpu().numpy()
	indices = np.random.choice(len(rh_in), 10000, replace=False)
	rh_point_cloud = vedo.Points(rh_in[indices], c='g', r=5)

	vedo.show(vol_arr, lh_point_cloud, rh_point_cloud, bg='white', axes=True)


def show_2_vol(vol1_fn: str, vol2_fn: str):
	"""
	Overlay two volumes.

	:param vol1_fn: filename of the first volume.
	:param vol2_fn: filename of the second volume.
	:return: None

	"""

	vol1 = nib.load(vol1_fn)
	vol2 = nib.load(vol2_fn)
	vol1_arr = vedo.Volume(vol1.get_fdata())
	vol2_arr = vedo.Volume(vol2.get_fdata())
	vol1_arr.color('b')
	vol1_arr.alpha([0.0, 0.1, 0.2])
	vol2_arr.color('r')
	vol2_arr.alpha([0.0, 0.1, 0.2])

	vedo.show(vol1_arr, vol2_arr, bg='white', axes=True)


if __name__ == '__main__':
	# show('s01.nii.gz',
	#      's01_lh.vtk',
	#      's01_rh.vtk',
	#      show_pts=True,
	#      show_mesh=True)
	# show('s02.nii.gz',
	#      's02_lh.vtk',
	#      's02_rh.vtk',
	#      show_pts=True,
	#      show_mesh=True)
	show_2_vol('s01.nii.gz', 's02.nii.gz')
