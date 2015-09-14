UNSUPGPU
========

A package for convPSD with CUDA in Torch.

Requirements
------------

Basic dependencies:

  * Torch7 (github.com/andresy/torch)
  * optim  (github.cim/koraykv/optim)
  * cutorch (https://github.com/torch/cutorch)
To run the demo scripts, you also need the following:

  * image (github.com/clementfarabet/lua---image)
  * sys   (github.com/clementfarabet/lua---sys)
  * xlua  (github.com/clementfarabet/lua---xlua)

Installation
------------

Build/Install:

  * Install Torch7 (refer to its own documentation).
  * clone all other repos (including this one) into dev directory of Torch7.
  * Rebuild torch, it will include all these projects too.

Alternatively, you can use torch's package manager. Once
Torch is installed, you can install `unsupgpu`: `$ torch-pkg install unsupgpu`.
