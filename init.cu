#include "TH.h"
#include "luaT.h"

#include "util.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libunsupgpu(lua_State *L);

int luaopen_libundupgpu(lua_State *L)
{
  lua_newtable(L);
  unsupgpu_util_init(L);
  lua_setfield(L, LUA_GLOBALSINDEX, "unsupgpu");
  return 1;
}


