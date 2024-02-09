from typing import Tuple, Callable

import torch
import torch.nn.functional as F


def _apply_affine_vol(volume: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Apply an affine transformation to a 3D volume using PyTorch's affine_grid and grid_sample functions.

	:param volume: Input 3D volume tensor of shape 1 x C x D x H x W.
	:param affine_matrix: Affine matrix of shape (4, 4)
	:return: transformed 3D volume tensor of shape (batch_size, channels, depth, height, width)

	"""

	# Generate grid
	grid = F.affine_grid(
		affine_matrix[:3, :],
		volume.shape,
		align_corners=True
	)

	# Apply transformation
	transformed_volume = F.grid_sample(volume, grid)

	return transformed_volume


def _apply_affine_points(points: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Applies the given `affine_matrix` to the given `points`.

	:param points: the given points of shape N x 3.
	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: transformed points of shape N x 3

	"""

	# Convert points to homogeneous coordinates
	ones_column: torch.Tensor = torch.ones(points.shape[0], 1, dtype=torch.float32, device=points.device)
	homogeneous_coords: torch.Tensor = torch.cat((points, ones_column), dim=1)

	# Apply affine transformation using einsum where k is the number of points
	transformed_points: torch.Tensor = torch.einsum(
		'ij,kj->ki',
		[affine_matrix, homogeneous_coords]
	)[:, :3]

	return transformed_points


def to_function(affine_matrix: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
	"""
	Converts the given `affine_matrix` to the corresponding function.

	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: the function corresponding to the given `affine_matrix`.

	"""

	def f(x: torch.Tensor) -> torch.Tensor:
		"""
		Applies the given `affine_matrix` to the given `x`.

		:param x: the given tensor of shape N x 3.
		:return: the function corresponding to the given `affine_matrix`.

		:note: `x` can be either points of shape N x 3 or a volume of shape 1 x 1 x H x W x D.

		"""

		if x.dim() == 5:
			return _apply_affine_vol(x, affine_matrix)
		elif x.dim() == 2:
			return _apply_affine_points(x, affine_matrix)
		else:
			raise ValueError(f'Unsupported dimension of `x`: {x.dim()}. Expected 2 or 5.')

	return f


def voxel_to_ras(voxel_coords: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Converts voxel coordinates to RAS coordinates using the given `affine_matrix`.

	:param voxel_coords: coordinates in voxel space of shape N x 3.
	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: RAS coordinates of shape N x 3

	"""

	f: Callable[[torch.Tensor], torch.Tensor] = to_function(affine_matrix)
	ras_coords: torch.Tensor = f(voxel_coords)

	return ras_coords


def ras_to_voxel(ras_coords: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Converts RAS coordinates to voxel coordinates using the given `affine_matrix`.

	:param ras_coords: coordinates in RAS space of shape N x 3.
	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: voxel coordinates of shape N x 3

	"""

	f: Callable[[torch.Tensor], torch.Tensor] = to_function(torch.inverse(affine_matrix))
	voxel_coords: torch.Tensor = f(ras_coords)

	return voxel_coords


def crop_vol(vol_arr: torch.Tensor,
             i_min: int,
             i_max: int,
             j_min: int,
             j_max: int,
             k_min: int,
             k_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Crops the given volume tensor `vol_arr` using the given indices.
	Also returns the affine matrix corresponding to the cropping operation.

	:param vol_arr: the given volume tensor of shape 1 x 1 x H x W x D.
	:return: cropped volume and the corresponding affine matrix.

	"""

	new_vol_arr = vol_arr[:, :, i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]
	affine_matrix = torch.tensor(
		[[1.0, 0.0, 0.0, i_min],
		 [0.0, 1.0, 0.0, j_min],
		 [0.0, 0.0, 1.0, k_min],
		 [0.0, 0.0, 0.0, 1.0]],
	)

	return new_vol_arr, affine_matrix


def _unstructured_interpolation_3d(values: torch.Tensor, voxel_coords: torch.Tensor,
                                   mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W x D from which new values are interpolated from.
	:param voxel_coords: voxel coordinates of the given points of shape N_p x 3 at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	:note: Internally, `points` are first reshaped N_p x 3 -> 1 x 1 x 1 x N_p x 3.
		Then, F.grid_sample produces `output` of shape 1 x C x 1 x 1 x N_p.
		We finally obtained the sampled values by reshaping 1 x C x 1 x 1 x N_p -> N_p x C.

		Coordinates of the given `point` should be in the implicit coordinate system implied by `values`,
		i.e., $(x, y, z) \in [0, H] \times [0, W] \times [0, D] \subseteq \mathbb{R}^3$.
		Coordinates of `points` are normalized to [-1, 1].

	"""

	input: torch.Tensor = values
	grid: torch.Tensor = voxel_coords[None, None, None, ...]
	grid[..., 0] = 2 * grid[..., 0] / (values.shape[2] - 1) - 1
	grid[..., 1] = 2 * grid[..., 1] / (values.shape[3] - 1) - 1
	grid[..., 2] = 2 * grid[..., 2] / (values.shape[4] - 1) - 1

	output: torch.Tensor = F.grid_sample(
		input,
		grid,
		mode=mode,
		padding_mode='border',
		align_corners=True,
	)

	return torch.squeeze(output, dim=(0, 2, 3)).permute(1, 0)


def _unstructured_interpolation_2d(values: torch.Tensor, voxel_coords: torch.Tensor,
                                   mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W from which new values are interpolated from.
	:param voxel_coords: voxel coordinates of the given points of shape N_p x 2 at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	:note: Internally, `points` are first reshaped N_p x 2 -> 1 x 1 x N_p x 2.
		Then, F.grid_sample produces `output` of shape 1 x C x 1 x N_p.
		We finally obtained the sampled values by reshaping 1 x C x 1 x N_p -> N_p x C.

		Coordinates of the given `point` should be in the implicit coordinate system implied by `values`,
		i.e., $(x, y) \in [0, H] \times [0, W] \subseteq \mathbb{R}^2$.
		Coordinates of `points` are normalized to [-1, 1].

	"""

	input: torch.Tensor = values
	grid: torch.Tensor = voxel_coords[None, None, ...]
	grid[..., 0] = 2 * grid[..., 0] / (values.shape[2] - 1) - 1
	grid[..., 1] = 2 * grid[..., 1] / (values.shape[3] - 1) - 1

	output: torch.Tensor = F.grid_sample(
		input,
		grid,
		mode=mode,
		padding_mode='border',
		align_corners=True,
	)

	return torch.squeeze(output, dim=(0, 2)).permute(1, 0)


def unstructured_interpolation(values: torch.Tensor, voxel_coords: torch.Tensor,
                               mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W (x D) from which new values are interpolated from.
	:param voxel_coords: voxel coordinates of the given points of shape N_p x dim at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	:note: Internally, `points` are first reshaped N_p x dim -> 1 x 1 (x 1) x N_p x dim.
		Then, F.grid_sample produces `output` of shape 1 x C x 1 (x 1) x N_p.
		We finally obtained the sampled values by reshaping 1 x C x 1 (x 1) x N_p -> N_p x C.

		Coordinates of the given `point` should be in the implicit coordinate system implied by `values`.
		Coordinates of `points` are normalized to [-1, 1].

	"""

	assert values.dim() in [4, 5], f'Unsupported dimension of `values`: {values.dim()}'

	assert voxel_coords.shape[1] in [2, 3], f'Unsupported dimension of `points`: {voxel_coords.shape[1]}'

	assert voxel_coords.shape[1] == values.dim() - 2, \
		f'Incompatible dimension of `values` and `points`: {values.dim()} and {voxel_coords.shape[1]}'

	if values.dim() == 5:
		return _unstructured_interpolation_3d(values, voxel_coords, mode)
	elif values.dim() == 4:
		return _unstructured_interpolation_2d(values, voxel_coords, mode)


def interpolate_disp(disp: torch.Tensor, ras_coords: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Interpolates the displacement field `disp` defined in voxel coordinate system
	at given `points` defined in RAS coordinate system.

	:param disp: the given displacement field of shape 1 x 3 x H x W x D.
	:param ras_coords: RAS coordinates of the given points of shape N x 3 at which new values are sampled.
	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: interpolated deformation field of shape N x 3

	"""

	voxel_coords: torch.Tensor = ras_to_voxel(ras_coords, affine_matrix)
	interpolated_disp: torch.Tensor = unstructured_interpolation(disp, voxel_coords)

	return interpolated_disp


def get_identity(image_shape: torch.Size, device: str = 'cuda') -> torch.Tensor:
	"""
	Returns the identity transformation of the given `image_shape`.

	:param image_shape: the shape of the image tensor (e.g., H x W x D).
	:param device: the device on which the tensor is allocated.
	:return: the identity transformation of shape 1 x 3 x H x W x D.

	"""

	grid: Tuple = torch.meshgrid(
		[torch.arange(size, dtype=torch.float32, device=device) for size in image_shape]
	)
	identity: torch.Tensor = torch.stack(grid)
	identity = identity.unsqueeze(0)
	identity = identity.type(torch.FloatTensor).to(device)

	return identity


def ddf_to_disp(ddf: torch.Tensor) -> torch.Tensor:
	"""
	Converts dense displacement field (DDF) to displacement field (disp)
	by subtracting the identity transformation.

	:param ddf: the dense displacement field of shape 1 x 3 x H x W x D.
	:return: displacement field of shape 1 x 3 x H x W x D

	"""

	image_shape: torch.Size = ddf.shape[2:]
	identity: torch.Tensor = get_identity(image_shape, device=ddf.device)
	disp: torch.Tensor = ddf - identity

	return disp


def surf_stn(ras_coords: torch.Tensor, ddf: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Given the dense displacement field `ddf` defined in voxel coordinate system,
	and the affine matrix `affine_matrix` that maps voxel to RAS coordinates,
	interpolates the displacement field at the voxel locations corresponding to each location in `ras_coords`.

	:param ras_coords: RAS coordinates of the given points of shape N x 3.
	:param ddf: the dense displacement field of shape 1 x 3 x H x W x D.
	:param affine_matrix: the affine matrix of shape 4 x 4 (read from NIfTI file).
	:return: deformed RAS coordinates of shape N x 3

	"""

	disp: torch.Tensor = ddf_to_disp(ddf)
	interpolated_disp = interpolate_disp(disp, ras_coords, affine_matrix)
	deformed_ras_coords: torch.Tensor = ras_coords + interpolated_disp

	return deformed_ras_coords


if __name__ == '__main__':
	values: torch.Tensor = torch.tensor(
		[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
	).unsqueeze(0).unsqueeze(0).cuda()
	points: torch.Tensor = torch.tensor(
		[[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
	).cuda()

	results: torch.Tensor = unstructured_interpolation(values, points)
	print(results)

	values: torch.Tensor = torch.tensor(
		[[1.0, 2.0], [3.0, 4.0]]
	).unsqueeze(0).unsqueeze(0).cuda()
	points: torch.Tensor = torch.tensor(
		[[0.5, 0.5], [0.0, 0.5], [1.0, 0.5]]
	).cuda()

	results: torch.Tensor = unstructured_interpolation(values, points)
	print(results)
