#include "TH.h"
#include "luaT.h"

#include "util.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libunsupgpu(lua_State *L);

int luaopen_libunsupgpu(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "unsupgpu");
  unsupgpu_util_init(L);
  return 1;
}


