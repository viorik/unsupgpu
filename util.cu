#include "THCApply.cuh"
#include "util.h"

THCState* getCutorchState(lua_State* L)
{
    lua_getglobal(L, "cutorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THCState *state = (THCState*) lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}

struct shrinkage {
  const float threshold_;

  shrinkage(float threshold): threshold_(threshold) {}

  __device__ __forceinline__ void operator()(float* x) {
    if (threshold_ == 0) *x = 1;
    if (*x > threshold_) *x -= threshold_;
    else if (*x < -threshold_) *x += threshold_;
    else *x = 0;
  }
};

static int unsupgpu_shrinkage(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  double threshold = luaT_getfieldchecknumber(L, 1, "lambda");

  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  THCudaTensor_pointwiseApply1(state, input,
                               shrinkage(threshold));
  THCudaTensor_set(state, output, input);

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg unsupgpu_util__ [] = {
  {"shrinkage", unsupgpu_shrinkage},
  {NULL, NULL}
};

void unsupgpu_util_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, unsupgpu_util__, "nn");
  lua_pop(L,1);
}

