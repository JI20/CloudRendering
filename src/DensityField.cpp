/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Utils/File/Logfile.hpp>
#include <Utils/Parallel/Reduction.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

//#include "Loaders/half/half.hpp"
#include "DensityField.hpp"

DensityField::~DensityField() {
    if (dataFloat) {
        delete[] dataFloat;
        dataFloat = nullptr;
    }
    if (dataByte) {
        delete[] dataByte;
        dataByte = nullptr;
    }
    if (dataShort) {
        delete[] dataShort;
        dataShort = nullptr;
    }
    //if (dataFloat16) {
    //    delete[] dataFloat16;
    //    dataFloat16 = nullptr;
    //}
}

const void* DensityField::getDataNative() {
    if (scalarDataFormatNative == ScalarDataFormat::FLOAT) {
        return dataFloat;
    } else if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        return dataByte;
    } else if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        return dataShort;
    //} else if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return dataFloat16;
    } else {
        return nullptr;
    }
}

size_t DensityField::getEntrySizeInBytes() {
    if (scalarDataFormatNative == ScalarDataFormat::FLOAT) {
        return 4;
    } else if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        return 1;
    } else if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        return 2;
    //} else if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return 2;
    } else {
        return 0;
    }
}

uint32_t DensityField::getEntrySizeInBytesUint32() {
    if (scalarDataFormatNative == ScalarDataFormat::FLOAT) {
        return 4;
    } else if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        return 1;
    } else if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        return 2;
    //} else if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return 2;
    } else {
        return 0;
    }
}

VkFormat DensityField::getEntryVulkanFormat() {
    if (scalarDataFormatNative == ScalarDataFormat::FLOAT) {
        return VK_FORMAT_R32_SFLOAT;
    } else if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        return VK_FORMAT_R8_UNORM;
    } else if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        return VK_FORMAT_R16_UNORM;
    //} else if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return VK_FORMAT_R16_SFLOAT;
    } else {
        return VK_FORMAT_UNDEFINED;
    }
}

float DensityField::getMinValue() {
    if (!minMaxComputed) {
        computeMinMax();
    }
    return minValue;
}

float DensityField::getMaxValue() {
    if (!minMaxComputed) {
        computeMinMax();
    }
    return maxValue;
}

void DensityField::computeMinMax() {
    std::pair<float, float> minMaxVal = {0.0f, 0.0f};
    if (scalarDataFormatNative == ScalarDataFormat::FLOAT) {
        minMaxVal = sgl::reduceFloatArrayMinMax(dataFloat, numEntries);
    } else if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        minMaxVal = sgl::reduceUnormByteArrayMinMax(dataByte, numEntries);
    } else if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        minMaxVal = sgl::reduceUnormShortArrayMinMax(dataShort, numEntries);
    //} else if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return VK_FORMAT_R16_SFLOAT;
    }
    minValue = minMaxVal.first;
    maxValue = minMaxVal.second;
    minMaxComputed = true;
}

/*void DensityField::switchNativeFormat(ScalarDataFormat newNativeFormat) {
    if (scalarDataFormatNative != ScalarDataFormat::FLOAT
            || newNativeFormat != ScalarDataFormat::FLOAT16) {
        sgl::Logfile::get()->throwError(
                "Error in DensityField::switchNativeFormat: "
                "Currently, only switching from float to float16 is supported.");
    }
    scalarDataFormatNative = newNativeFormat;
    dataFloat16 = new HalfFloat[numEntries];
    for (size_t i = 0; i < numEntries; i++) {
        dataFloat16[i] = HalfFloat(dataFloat[i]);
    }
}*/

const uint8_t* DensityField::getDataByte() {
    if (scalarDataFormatNative != ScalarDataFormat::BYTE) {
        sgl::Logfile::get()->throwError("Error in DensityField::getDataByte: Native format is not byte.");
    }
    return dataByte;
}

const uint16_t* DensityField::getDataShort() {
    if (scalarDataFormatNative != ScalarDataFormat::SHORT) {
        sgl::Logfile::get()->throwError("Error in DensityField::getDataByte: Native format is not short.");
    }
    return dataShort;
}

/*const HalfFloat* DensityField::getDataFloat16() {
    if (scalarDataFormatNative != ScalarDataFormat::FLOAT16) {
        sgl::Logfile::get()->throwError("Error in DensityField::getDataFloat16: Native format is not float16.");
    }
    return dataFloat16;
}*/

const float* DensityField::getDataFloat() {
    if (!dataFloat && scalarDataFormatNative == ScalarDataFormat::BYTE) {
        dataFloat = new float[numEntries];
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numEntries), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for default(none)
#endif
        for (size_t i = 0; i < numEntries; i++) {
#endif
            dataFloat[i] = float(dataByte[i]) / 255.0f;
        }
#ifdef USE_TBB
        });
#endif
    }
    if (!dataFloat && scalarDataFormatNative == ScalarDataFormat::SHORT) {
        dataFloat = new float[numEntries];
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numEntries), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for default(none) shared(numEntries)
#endif
        for (size_t i = 0; i < numEntries; i++) {
#endif
            dataFloat[i] = float(dataShort[i]) / 65535.0f;
        }
#ifdef USE_TBB
        });
#endif
    }
    /*if (!dataFloat && scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
        dataFloat = new float[numEntries];
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numEntries), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for default(none) shared(numEntries)
#endif
        for (size_t i = 0; i < numEntries; i++) {
#endif
            dataFloat[i] = float(dataFloat16[i]);
        }
#ifdef USE_TBB
        });
#endif
    }*/
    return dataFloat;
}

float DensityField::getDataFloatAt(size_t idx) {
    if (dataFloat) {
        return dataFloat[idx];
    }
    if (scalarDataFormatNative == ScalarDataFormat::BYTE) {
        return float(dataByte[idx]) / 255.0f;
    }
    if (scalarDataFormatNative == ScalarDataFormat::SHORT) {
        return float(dataShort[idx]) / 65535.0f;
    }
    //if (scalarDataFormatNative == ScalarDataFormat::FLOAT16) {
    //    return float(dataFloat16[idx]);
    //}
    return 0.0f;
}

float DensityField::getDataFloatAtNorm(size_t idx) {
    if (!minMaxComputed) {
        computeMinMax();
    }
    return (getDataFloatAt(idx) - minValue) / (maxValue - minValue);
}
