#ifndef UTIL_H
#define UTIL_H

extern "C"
{
#include <lua.h>
}
#include <luaT.h>
#include <THC/THC.h>

THCState* getCutorchState(lua_State* L);

void unsupgpu_shrinkage_init(lua_State *L);
