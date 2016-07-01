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

struct shrinkagegpu {
  const float lambda_;

  shrinkagegpu(float lambda): lambda_(lambda) {}

  __device__ __forceinline__ void operator()(float* x) {
    if (lambda_ == 0) *x = 1;
    if (*x > lambda_) *x -= lambda_;
    else if (*x < -lambda_) *x += lambda_;
    else *x = 0;
  }
};

static int unsupgpu_shrinkagegpu(lua_State *L)
{
  //printf("ok\n");
  THCState *state = getCutorchState(L);
  double lambda = luaL_checknumber(L,2);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  //THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  //double lambda = luaT_getfieldchecknumber(L, 1, "lambda");

  //THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  THC_pointwiseApply1(state, input,
                               shrinkagegpu(lambda));
  //THCudaTensor_set(state, output, input);
  
  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg unsupgpu_util [] = {
  {"shrinkagegpu", unsupgpu_shrinkagegpu},
  {NULL, NULL}
};

void unsupgpu_util_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  //luaT_registeratname(L, unsupgpu_util, "nn");
  luaL_register(L,NULL, unsupgpu_util);
  lua_pop(L,1);
}

