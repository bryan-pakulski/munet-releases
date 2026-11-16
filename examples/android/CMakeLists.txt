# Builds libmunet_jni.so, the JNI bridge of MuNet.kt. See docs/android.md.
# MUNET_ANDROID: the extracted munet-android-<version> archive (pass it from build.gradle: -DMUNET_ANDROID=...).
cmake_minimum_required(VERSION 3.22)
project(munet_jni CXX)

add_library(munet_c SHARED IMPORTED)
set_target_properties(munet_c PROPERTIES
  IMPORTED_LOCATION ${MUNET_ANDROID}/jniLibs/${ANDROID_ABI}/libmunet_c.so
  INTERFACE_INCLUDE_DIRECTORIES ${MUNET_ANDROID}/include)

add_library(munet_jni SHARED munet_jni.cpp)
target_compile_features(munet_jni PRIVATE cxx_std_17)
target_link_libraries(munet_jni PRIVATE munet_c log)
target_link_options(munet_jni PRIVATE -Wl,-z,max-page-size=16384)  # devices with 16 KB pages
