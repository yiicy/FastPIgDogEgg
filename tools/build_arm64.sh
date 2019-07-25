#!/bin/bash
# This script shows how one can build a fpde for the Android platform using android-tool-chain.
# IMPORTANT!!!!!!!!!!!!!!
# remove "-g" compile flags in  "$ANDROID_NDK/build/cmake/android.toolchain.cmake"
# to remove debug info
#export ANDROID_NDK=/home/public/android-ndk-r16b/
export ANDROID_NDK=/your/andorid-ndk/path

FAST_PIG_DOG_EGG_ROOT="$( cd "$(dirname "$0")" ;cd ..; pwd -P)"
echo "-- Fast Pig Dog Egg root dir is: $FAST_PIG_DOG_EGG_ROOT"

if [ -z "$ANDROID_NDK" ]; then
    echo "-- Did you set ANDROID_NDK variable?"
    exit 1
fi

if [ -d "$ANDROID_NDK" ]; then
    echo "-- Using Android ndk at $ANDROID_NDK"
else
    echo "-- Cannot find ndk: did you install it under $ANDROID_NDK ?"
    exit 1
fi

# build the target into build_android.
BUILD_ROOT=$FAST_PIG_DOG_EGG_ROOT/build-arm64

#if [ -d $BUILD_ROOT ];then
#   rm -rf $BUILD_ROOT
#fi

mkdir -p $BUILD_ROOT
echo "-- Build Fast Pig Dog Egg into: $BUILD_ROOT"

if [ ! -d $FAST_PIG_DOG_EGG_ROOT/output ];then
    mkdir -p $FAST_PIG_DOG_EGG_ROOT/output
fi

echo "-- set Fast Pig Dog Egg Output into: $FAST_PIG_DOG_EGG_ROOT/output"

# Now, actually build.
echo "-- Building Fast Pig Dog Egg ..."
cd $BUILD_ROOT
#-DCMAKE_TOOLCHAIN_FILE=../../../cmake/android/android.toolchain.cmake \ # set toolchain file to file in this project
#-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \ # set toolchain file to NDK default
#-DANDROID_STL=gnustl_static \ # set stl lib
#-DANDROID_TOOLCHAIN=clang \ # set compile to gcc or clang
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK                  \
    -DANDROID_ABI="arm64-v8a"                   \
    -DANDROID_NATIVE_API_LEVEL=21               \
    -DENABLE_DEBUG=NO                           \
    -DUSE_ARM64=YES                             \
    -DUSE_CL=NO                                 \
    -DUSE_OPENMP=NO                             \
    -DBUILD_SHARED=NO                           \
    -DBUILD_WITH_TEST=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" #&& make install
else
    make "-j$(nproc)" #&& make install
fi
