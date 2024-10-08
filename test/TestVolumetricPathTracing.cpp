/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <gtest/gtest.h>
#include <json/json.h>

#include <Utils/File/FileUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Graphics/Texture/Bitmap.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>

#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/Primitives.h"
#include "CloudData.hpp"
#include "PathTracer/LightEditorWidget.hpp"
#include "VolumetricPathTracingTestData.hpp"
#include "VolumetricPathTracingTestRenderer.hpp"

class VolumetricPathTracingTest : public ::testing::Test {
protected:
    void SetUp() override {
        renderer = new sgl::vk::Renderer(sgl::AppSettings::get()->getPrimaryDevice());
        transferFunctionWindow = new sgl::MultiVarTransferFunctionWindow;
        transferFunctionWindow->setShowWindow(false);
        transferFunctionWindow->setAttributeNames({"Volume", "Isosurface"});
        lightEditorWidget = new LightEditorWidget(renderer);
        lightEditorWidget->setShowWindow(false);
        vptRenderer0 = std::make_shared<VolumetricPathTracingTestRenderer>(renderer);
        vptRenderer1 = std::make_shared<VolumetricPathTracingTestRenderer>(renderer);
    }

    void TearDown() override {
        vptRenderer0 = {};
        vptRenderer1 = {};
        if (transferFunctionWindow) {
            delete transferFunctionWindow;
            transferFunctionWindow = nullptr;
        }
        if (lightEditorWidget) {
            delete lightEditorWidget;
            lightEditorWidget = nullptr;
        }
        if (renderer) {
            delete renderer;
            renderer = nullptr;
        }
    }

    void testEqualMean() {
        vptRenderer0->setRenderingResolution(renderingResolution, renderingResolution);
        vptRenderer1->setRenderingResolution(renderingResolution, renderingResolution);

        uint32_t width = vptRenderer0->getFrameWidth();
        uint32_t height = vptRenderer0->getFrameHeight();
        float* frameData0 = vptRenderer0->renderFrame(numSamples);
        float* frameData1 = vptRenderer1->renderFrame(numSamples);

        auto numPixelsFlt = double(width * height);
        double mean0[] = { 0.0, 0.0, 0.0 };
        double mean1[] = { 0.0, 0.0, 0.0 };
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < 3; c++) {
                    mean0[c] += double(frameData0[(x + y * width) * 3 + c]) / numPixelsFlt;
                    mean1[c] += double(frameData1[(x + y * width) * 3 + c]) / numPixelsFlt;
                }
            }
        }

        for (uint32_t c = 0; c < 3; c++) {
            if (std::abs(mean0[c] - mean1[c]) > epsilon || outputImagesAlways) {
                debugOutputImage(
                        std::string() + "out_" + ::testing::UnitTest::GetInstance()->current_test_info()->name()
                        + "_0.png",
                        frameData0, width, height);
                debugOutputImage(
                        std::string() + "out_" + ::testing::UnitTest::GetInstance()->current_test_info()->name()
                        + "_1.png",
                        frameData1, width, height);
            }
            ASSERT_NEAR(mean0[c], mean1[c], epsilon);
        }
    }

    static void debugOutputImage(const std::string& filename, const float* frameData, uint32_t width, uint32_t height) {
        sgl::BitmapPtr bitmap(new sgl::Bitmap(int(width), int(height), 32));
        uint8_t* bitmapData = bitmap->getPixels();
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < 3; c++) {
                    float value = frameData[(x + y * width) * 3 + c];
                    bitmapData[(x + y * width) * 4 + c] = uint8_t(std::clamp(value * 255.0f, 0.0f, 255.0f));
                }
                bitmapData[(x + y * width) * 4 + 3] = 255;
            }
        }
        bitmap->savePNG(filename.c_str(), false);
    }

    void loadEmptyTf() {
        std::string emptyTfString =
                "<TransferFunction>\n"
                "<OpacityPoints>\n"
                "<OpacityPoint position=\"0\" opacity=\"0\"/><OpacityPoint position=\"1\" opacity=\"0\"/>\n"
                "</OpacityPoints>\n"
                "<ColorPoints color_data=\"ushort\">\n"
                "<ColorPoint position=\"0\" r=\"0\" g=\"0\" b=\"0\"/><ColorPoint position=\"1\" r=\"0\" g=\"0\" b=\"0\"/>\n"
                "</ColorPoints>\n"
                "</TransferFunction>";
        transferFunctionWindow->loadFunctionFromXmlString(0, emptyTfString);
        transferFunctionWindow->loadFunctionFromXmlString(1, emptyTfString);
    }

    sgl::vk::Renderer* renderer = nullptr;
    sgl::MultiVarTransferFunctionWindow* transferFunctionWindow = nullptr;
    LightEditorWidget* lightEditorWidget = nullptr;
    int numSamples = 64;
    float epsilon = 1e-3;
    int renderingResolution = 128;
    std::shared_ptr<VolumetricPathTracingTestRenderer> vptRenderer0;
    std::shared_ptr<VolumetricPathTracingTestRenderer> vptRenderer1;
    bool outputImagesAlways = false; // For testing output.
};

/**
 * Test whether different volumetric path tracing renderers produce the same image mean when rendering a cube with
 * constant density across the whole volume domain.
 */
TEST_F(VolumetricPathTracingTest, DeltaTrackingRatioTrackingEqualMeanTest) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 1, 1, 1, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setVptMode(VptMode::RATIO_TRACKING);
    testEqualMean();
}
TEST_F(VolumetricPathTracingTest, DeltaTrackingSeedIndependentEqualMeanTest) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 1, 1, 1, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setCustomSeedOffset(268435456u);
    testEqualMean();
}
// TODO: Fix this test case.
TEST_F(VolumetricPathTracingTest, DeltaTrackingGridTypesGrid1Test) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 1, 1, 1, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer0->setUseSparseGrid(false);
    vptRenderer1->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setUseSparseGrid(true);
    testEqualMean();
}
TEST_F(VolumetricPathTracingTest, DeltaTrackingGridTypesGrid8Test) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer0->setUseSparseGrid(false);
    vptRenderer1->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setUseSparseGrid(true);
    testEqualMean();
}
// The following two tests fail due to recent changes. Empty layers are now suppressed, and filtering can be different.
/*TEST_F(VolumetricPathTracingTest, DeltaTrackingGridTypesGrid8BoundaryLayerTest) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f, true);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer0->setUseSparseGrid(false);
    vptRenderer1->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setUseSparseGrid(true);
    testEqualMean();
}
TEST_F(VolumetricPathTracingTest, DeltaTrackingGridTypesGrid8BoundaryLayerTest2) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f, true);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer0->setUseSparseGrid(false);
    vptRenderer1->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setUseSparseGrid(true);
    vptRenderer1->setGridInterpolationType(GridInterpolationType::TRILINEAR);
    testEqualMean();
}*/

// TODO: Fix this test case. Decomposition tracking plus sparse volumes can lead to artifacts.
/*TEST_F(VolumetricPathTracingTest, DecompositionTrackingGridTypesSphereTest) {
    CloudDataPtr cloudData = std::make_shared<CloudData>();
    cloudData->setNanoVdbGridHandle(nanovdb::createFogVolumeSphere<float>(
            0.25f, nanovdb::Vec3<float>(0), 0.01f));
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    numSamples = 256ad;
    vptRenderer0->setVptMode(VptMode::DECOMPOSITION_TRACKING);
    vptRenderer0->setUseSparseGrid(false);
    vptRenderer1->setVptMode(VptMode::DECOMPOSITION_TRACKING);
    vptRenderer1->setUseSparseGrid(true);
    testEqualMean();
}*/

TEST_F(VolumetricPathTracingTest, DeltaTrackingDecompositionTrackingEqualMeanTest1) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setGridInterpolationType(GridInterpolationType::NEAREST);
    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setGridInterpolationType(GridInterpolationType::NEAREST);
    vptRenderer1->setVptMode(VptMode::DECOMPOSITION_TRACKING);
    testEqualMean();
}

TEST_F(VolumetricPathTracingTest, DeltaTrackingDecompositionTrackingEqualMeanTest2) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setGridInterpolationType(GridInterpolationType::STOCHASTIC);
    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer0->setGridInterpolationType(GridInterpolationType::STOCHASTIC);
    vptRenderer1->setVptMode(VptMode::DECOMPOSITION_TRACKING);
    testEqualMean();
}

TEST_F(VolumetricPathTracingTest, DeltaTrackingDecompositionTrackingEqualMeanTest3) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f, true);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setVptMode(VptMode::DECOMPOSITION_TRACKING);
    testEqualMean();
}

TEST_F(VolumetricPathTracingTest, DeltaTrackingNextEventTrackingEqualMeanTest) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f, true);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setVptMode(VptMode::NEXT_EVENT_TRACKING);
    testEqualMean();
}

TEST_F(VolumetricPathTracingTest, DeltaTrackingNextEventTrackingEqualMeanTestSurfaces) {
    CloudDataPtr cloudData = createCloudBlock(transferFunctionWindow, lightEditorWidget, 8, 8, 8, 1.0f, true);
    vptRenderer0->setCloudData(cloudData);
    vptRenderer1->setCloudData(cloudData);

    epsilon = 5e-3; //< TODO: Examine reasons for small error.
    numSamples = 256;
    outputImagesAlways = true;
    transferFunctionWindow->setShowWindow(true);
    loadEmptyTf();
    vptRenderer0->setVptMode(VptMode::DELTA_TRACKING);
    vptRenderer1->setVptMode(VptMode::NEXT_EVENT_TRACKING);
    for (auto vptRenderer : { vptRenderer0, vptRenderer1 }) {
        vptRenderer->getVptPass()->setUseIsosurfaces(true);
        vptRenderer->getVptPass()->setUseEnvironmentMapFlag(true);
        vptRenderer->getVptPass()->loadEnvironmentMapImage(
                sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/env_maps/small_empty_room_1_1k.exr");
    }
    testEqualMean();
}

void vulkanErrorCallback() {
    std::cerr << "Application callback" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize the filesystem utilities.
    sgl::FileUtils::get()->initialize("CloudRendering", argc, argv);

    // Load the file containing the app settings.
    std::string settingsFile = sgl::FileUtils::get()->getConfigDirectory() + "settings.txt";
    sgl::AppSettings::get()->setSaveSettings(false);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);

#ifdef DATA_PATH
    if (!sgl::FileUtils::get()->directoryExists("Data") && !sgl::FileUtils::get()->directoryExists("../Data")) {
        sgl::AppSettings::get()->setDataDirectory(DATA_PATH);
    }
#endif

    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);
    sgl::AppSettings::get()->createHeadless();

    std::vector<const char*> optionalDeviceExtensions;
#ifdef SUPPORT_OPTIX
    optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    instance->setDebugCallback(&vulkanErrorCallback);
    sgl::vk::Device* device = new sgl::vk::Device;
    device->createDeviceHeadless(
            instance, {
                    VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
                    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
            },
            optionalDeviceExtensions);
    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->initializeSubsystems();

    int returnValue = RUN_ALL_TESTS();

    sgl::AppSettings::get()->release();
    return returnValue;
}
