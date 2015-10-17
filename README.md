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
I was able to install the package using the following procedure:

 * git clone https://github.com/viorik/unsupgpu
 * cd unsupgpu/
 * luarocks install unsupgpu-0.1-0.rockspec

The project is not yet uploaded to the global LuaRocks server so using "luarocks install unsupgpu" does not work yet

