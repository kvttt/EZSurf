.. EZSurf documentation master file, created by
   sphinx-quickstart on Fri Feb  9 08:32:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EZSurf
======

I developed this package to

#. Make it easy to check whether a **reconstructed surface** is in the same
   space as the **corresponding volume**.

#. Provide a number of mathematically intuitive tools that deal with

   * Conversion between **RAS+** and **voxel** coordinates.

   * Conversion between displacement fields and deformation fields.

#. Make it easy to warp a surface in **RAS+ space** using a dense displacement field
   predicted in the **voxel space**.

#. Most importantly, help noobs, i.e., myself, understand change of
   coordinates between RAS+ and voxel spaces.

In addition, except for visualization tools, all functions are implemented in PyTorch,
making it easy to embed them in a deep learning pipeline.

.. note:: This package is still in its infancy. I am still learning and
   developing it. Feel free to use it, but I am sure there are still bugs to be fixed.
   If you have any suggestions, please let me know.

.. toctree::
   :maxdepth: 2

   dependencies
   api
   tools
   visualization


