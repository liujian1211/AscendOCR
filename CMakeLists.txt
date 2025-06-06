# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(sampleResnetQuickStart)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(CMAKE_C_COMPILER "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-g++")
set(CMAKE_AR "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-ar" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB  "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-gcc-ranlib" CACHE FILEPATH "Ranlib")
set(CMAKE_LINKER "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-ld" CACHE FILEPATH "Linker")
set(CMAKE_STRIP "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/bin/aarch64-target-linux-gnu-strip" CACHE FILEPATH "Strip")

set(CMAKE_SYSROOT "/home/chenqiang/project/liujian/orangePi/tool/cross_compile/toolchain/sysroot")

set(HOME "/home/chenqiang")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "${HOME}/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "${HOME}/Ascend/ascend-toolkit/latest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "${HOME}/Ascend/ascend-toolkit/latest/thirdpart")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()

include_directories(
        /home/chenqiang/project/liujian/orangePi/envs/opencv_install/include/opencv4
   ${INC_PATH}/runtime/include/
)

link_directories(
        /home/chenqiang/project/liujian/orangePi/envs/opencv_install/lib
    ${LIB_PATH}
)

add_executable(main_ocr
#        sampleResnetQuickStart.cpp
        AscendOCR.cpp
        Ascend_rec.cpp
        AscendOCR.h
        #AscendOCR_det.cpp
)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main_ocr funcsim)
else()
    target_link_libraries(main_ocr ascendcl stdc++ opencv_world dl rt
            -Wl,-rpath-link,/home/chenqiang/Ascend/ascend-toolkit/latest/thirdpart/lib
            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/zlib_install/lib
            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/jpeg_install/lib
            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/png_install/lib
            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/libx11_install/lib

            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/mysql_install/lib
            -Wl,-rpath-link,/home/chenqiang/project/liujian/orangePi/envs/openssl_install/lib)
endif()

install(TARGETS main_ocr DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
