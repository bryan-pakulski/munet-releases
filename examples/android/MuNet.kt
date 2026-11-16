// Kotlin wrapper of the MuNet C API through munet_jni.cpp. See docs/android.md.
package com.example.munet

import android.content.Context

class MuNet private constructor(private var handle: Long) : AutoCloseable {
    val inputCount: Int get() = nativeCount(handle, true)
    val outputCount: Int get() = nativeCount(handle, false)

    fun inputName(index: Int): String = nativeName(handle, index, true)

    fun outputName(index: Int): String = nativeName(handle, index, false)

    fun inputShape(index: Int): LongArray = nativeShape(handle, index, true)

    fun outputShape(index: Int = 0): LongArray = nativeShape(handle, index, false)

    /** Runs the model on one float array per input, in declaration order. Call off the UI thread. */
    fun run(vararg inputs: FloatArray) = nativeRun(handle, inputs)

    /** Output `index` of the last run. */
    fun output(index: Int = 0): FloatArray = nativeOutput(handle, index)

    /** GPU time of the last run in milliseconds, after its outputs were read. Negative when unknown. */
    fun gpuMs(): Double = nativeGpuMs(handle)

    /** Device name, load and compile times, kernel and memory counters as JSON. */
    fun stats(): String = nativeStats(handle)

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

        /**
         * Loads model.onnx or model.mnet from a file path. device -1 is the CPU reference executor.
         * ONNX only: inputShapes gives a static shape to every input with a symbolic dimension, and cache names a
         * .mnet file that later loads reuse while the ONNX file and these options are unchanged.
         */
        fun load(
            path: String, device: Int = 0, fp16: Boolean = true,
            inputShapes: Map<String, LongArray> = emptyMap(), cache: String? = null,
        ): MuNet {
            val names = inputShapes.keys.toTypedArray()
            return MuNet(nativeLoad(path, device, fp16, names, Array(names.size) { inputShapes.getValue(names[it]) }, cache))
        }

        @JvmStatic private external fun nativeInit(cacheDir: String, licenseKey: String?): String
        @JvmStatic private external fun nativeProbe(device: Int): String
        @JvmStatic private external fun nativeLoad(
            path: String, device: Int, fp16: Boolean, shapeNames: Array<String>, shapes: Array<LongArray>, cachePath: String?,
        ): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeRun(handle: Long, inputs: Array<out FloatArray>)
        @JvmStatic private external fun nativeCount(handle: Long, input: Boolean): Int
        @JvmStatic private external fun nativeName(handle: Long, index: Int, input: Boolean): String
        @JvmStatic private external fun nativeShape(handle: Long, index: Int, input: Boolean): LongArray
        @JvmStatic private external fun nativeOutput(handle: Long, index: Int): FloatArray
        @JvmStatic private external fun nativeGpuMs(handle: Long): Double
        @JvmStatic private external fun nativeStats(handle: Long): String
    }
}
