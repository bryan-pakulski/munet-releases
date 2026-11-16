// Kotlin wrapper of the MuNet C API through munet_jni.cpp. See docs/android.md.
package com.example.munet

import android.content.Context

class MuNet private constructor(private var handle: Long) : AutoCloseable {
    /** Runs the model on one float array per input, in declaration order. Call off the UI thread. */
    fun run(vararg inputs: FloatArray) = nativeRun(handle, inputs)

    /** Output `index` of the last run. */
    fun output(index: Int = 0): FloatArray = nativeOutput(handle, index)

    fun outputShape(index: Int = 0): LongArray = nativeOutputShape(handle, index)

    override fun close() {
        if (handle != 0L) nativeFree(handle)
        handle = 0
    }

    companion object {
        init {
            System.loadLibrary("munet_jni")  // loads libmunet_c.so with it
        }

        /** Call once, first. Returns the license status as JSON. */
        fun init(context: Context, licenseKey: String? = null): String =
            nativeInit(context.cacheDir.absolutePath, licenseKey)

        /** Checks and times the kernel choices for this GPU and stores them. Seconds; call off the UI thread. */
        fun probe(device: Int = 0): String = nativeProbe(device)

        /** Loads model.onnx or model.mnet from a file path. device -1 is the CPU reference executor. */
        fun load(path: String, device: Int = 0, fp16: Boolean = true) = MuNet(nativeLoad(path, device, fp16))

        @JvmStatic private external fun nativeInit(cacheDir: String, licenseKey: String?): String
        @JvmStatic private external fun nativeProbe(device: Int): String
        @JvmStatic private external fun nativeLoad(path: String, device: Int, fp16: Boolean): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeRun(handle: Long, inputs: Array<out FloatArray>)
        @JvmStatic private external fun nativeOutputShape(handle: Long, index: Int): LongArray
        @JvmStatic private external fun nativeOutput(handle: Long, index: Int): FloatArray
    }
}
