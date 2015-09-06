#include "THCApply.cuh"
#include "util.h"


struct shrinkage_ {
  const double threshold_;

  shrinkage_(double threshold): threshold_(threshold) {}

  __device__ __forceinline__ void operator()(double* x) {
    if (threshold_ == 0) *x = 1;
    if (*x > threshold_) *x -= threshold_;
    else if (*x < -threshold_) *x += threshold_;
    else *x = 0;
  }
};

static int unsupgpu_(shrinkage)(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  THCudaTensor_pointwiseApply1(state, input,
                               shrinkage_(threshold));
  THCudaTensor_set(state, output, input);

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg unsupgpu_(shrinkage__) [] = {
  {"shrinkage", unsupgpu_(shrinkage)},
  {NULL, NULL}
};

void unsupgpu_shrinkage_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, unsupgpu_shrinkage__);
  lua_pop(L,1);
}

