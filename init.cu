#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define unsupgpu_(NAME) TH_CONCAT_3(unsupgpu_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#include "generic/util.h"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libunsupgpu(lua_State *L);

int luaopen_libundupgpu(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  unsupgpu_shrinkage_init(L);
  lua_setfield(L, LUA_GLOBALSINDEX, "unsupgpu");
  return 1;
}


