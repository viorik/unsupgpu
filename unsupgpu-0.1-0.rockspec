package = "unsupgpu"
version = "0.1-0"

source = {
   url = "git://github.com/viorik/unsupgpu",
   tag = "master"
}

description = {
   summary = "Extending unsup for unsupervised learning in Torch to run conv_psd with CUDA",
   detailed = [[
   Made few changes to support cuda for conv_psd.
   ]],
   homepage = "https://github.com/viorik/unsupgpu",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "xlua >= 1.0",
   "optim >= 1.0",
   "cutorch"
}

build = {
   type = "command",
   build_command = [[
   		 cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
