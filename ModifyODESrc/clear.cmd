@echo off
del *.vcxproj
del *.a
del *.so
del Makefile
del ModifyODE.cpp
del DrawStuffWorld.cpp
del FeatureInfoDbg.cpp
del MathHelperCython.cpp
del MotionUtils.cpp
del EnumODE.cpp
del *.sln
del *.filters
del *.cmake
del CMakeCache.txt
rmdir Release /s /q
rmdir ode.egg-info /s /q
rmdir build /s /q
rmdir CMakeFiles /s /q
rmdir x64 /s /q
rmdir ModifyODE.dir /s /q
rmdir MotionUtils.dir /s /q
rmdir Utils.dir /s /q
rmdir testbuild /s /q
del *.lib
del EigenWrapper.cpp
del *.pyd
