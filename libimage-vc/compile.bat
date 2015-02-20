@echo off
cl -c /O2 /I ..\libimage\include ..\libimage\src\*.c 
lib *.obj /out:image.lib
del *.obj