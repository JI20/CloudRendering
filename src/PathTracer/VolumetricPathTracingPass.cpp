/**
 * MIT License
 *
 * Copyright (c) 2021, Christoph Neuhauser, Ludwig Leonard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <memory>
#include <utility>
#include <glm/vec3.hpp>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Texture/Bitmap.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/RayTracingPipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/imgui_stdlib.h>

#include "Denoiser/EAWDenoiser.hpp"
#ifdef SUPPORT_OPTIX
#include "Denoiser/OptixVptDenoiser.hpp"
#endif

#include "CloudData.hpp"
#include "MomentUtils.hpp"
#include "SuperVoxelGrid.hpp"
#include "VolumetricPathTracingPass.hpp"

VolumetricPathTracingPass::VolumetricPathTracingPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    frameInfoBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(FrameInfo),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    computeWrappingZoneParameters(momentUniformData.wrapping_zone_parameters);
    momentUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(MomentUniformData), &momentUniformData,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    if (sgl::AppSettings::get()->getSettings().getValueOpt(
            "vptEnvironmentMapImage", environmentMapFilenameGui)) {
        useEnvironmentMapImage = true;
        sgl::AppSettings::get()->getSettings().getValueOpt("vptUseEnvironmentMap", useEnvironmentMapImage);
        loadEnvironmentMapImage();
    }

    blitPrimaryRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Primary");
    blitScatterRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Scatter");

    fileDialogInstance = IGFD_Create();

    createDenoiser();
    updateVptMode();
}

VolumetricPathTracingPass::~VolumetricPathTracingPass() {
    if (isEnvironmentMapLoaded) {
        sgl::AppSettings::get()->getSettings().addKeyValue(
                "vptEnvironmentMapImage", loadedEnvironmentMapFilename);
        sgl::AppSettings::get()->getSettings().addKeyValue(
                "vptUseEnvironmentMap", useEnvironmentMapImage);
    }

    IGFD_Destroy(fileDialogInstance);
}

void VolumetricPathTracingPass::createDenoiser() {
    denoiser = createDenoiserObject(denoiserType, renderer);

    if (resultImageTexture) {
        setDenoiserFeatureMaps();
        if (denoiser) {
            denoiser->recreateSwapchain(lastViewportWidth, lastViewportHeight);
        }
    }
}

void VolumetricPathTracingPass::setOutputImage(sgl::vk::ImageViewPtr& imageView) {
    sceneImageView = imageView;

    sgl::vk::ImageSamplerSettings samplerSettings;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    sgl::vk::ImageSettings imageSettings = imageView->getImage()->getImageSettings();
    imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    resultImageView = std::make_shared<sgl::vk::ImageView>(
            std::make_shared<sgl::vk::Image>(device, imageSettings));
    imageSettings.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    denoisedImageView = std::make_shared<sgl::vk::ImageView>(
            std::make_shared<sgl::vk::Image>(device, imageSettings));

    resultImageTexture = std::make_shared<sgl::vk::Texture>(resultImageView, samplerSettings);

    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    accImageTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    firstXTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    firstWTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);

    blitPrimaryRayMomentTexturePass->setOutputImage(imageView);
    blitScatterRayMomentTexturePass->setOutputImage(imageView);

    setDenoiserFeatureMaps();

    frameInfo.frameCount = 0;
    setDataDirty();
}

void VolumetricPathTracingPass::setDenoiserFeatureMaps() {
    if (denoiser) {
        denoiser->setFeatureMap("color", resultImageTexture);
        denoiser->setFeatureMap("position", firstXTexture);
        denoiser->setFeatureMap("normal", firstWTexture);
        denoiser->setOutputImage(denoisedImageView);
    }
}

void VolumetricPathTracingPass::recreateSwapchain(uint32_t width, uint32_t height) {
    lastViewportWidth = width;
    lastViewportHeight = height;

    blitScatterRayMomentTexturePass->recreateSwapchain(width, height);
    blitPrimaryRayMomentTexturePass->recreateSwapchain(width, height);

    if (useDenoiser && denoiser) {
        denoiser->recreateSwapchain(width, height);
    }
}

void VolumetricPathTracingPass::setCloudDataSet(CloudDataPtr& data) {
    cloudData = data;

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = data->getGridSizeX();
    imageSettings.height = data->getGridSizeY();
    imageSettings.depth = data->getGridSizeZ();
    imageSettings.imageType = VK_IMAGE_TYPE_3D;
    imageSettings.format = VK_FORMAT_R32_SFLOAT;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    sgl::vk::ImageSamplerSettings samplerSettings;
    samplerSettings.addressModeU = samplerSettings.addressModeV = samplerSettings.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerSettings.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    densityFieldTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    densityFieldTexture->getImage()->uploadData(
            data->getGridSizeX() * data->getGridSizeY() * data->getGridSizeZ() * sizeof(float),
            data->getDensityField());

    frameInfo.frameCount = 0;
    setDataDirty();
    updateVptMode();
}

void VolumetricPathTracingPass::onHasMoved() {
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::updateVptMode() {
    if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING && cloudData) {
        superVoxelGridDecompositionTracking = {};
        superVoxelGridResidualRatioTracking = std::make_shared<SuperVoxelGridResidualRatioTracking>(
                device, cloudData->getGridSizeX(), cloudData->getGridSizeY(),
                cloudData->getGridSizeZ(), cloudData->getDensityField(),
                superVoxelSize);
        superVoxelGridResidualRatioTracking->setExtinction((cloudExtinctionBase * cloudExtinctionScale).x);
    } else if (vptMode == VptMode::DECOMPOSITION_TRACKING && cloudData) {
        superVoxelGridResidualRatioTracking = {};
        superVoxelGridDecompositionTracking = std::make_shared<SuperVoxelGridDecompositionTracking>(
                device, cloudData->getGridSizeX(), cloudData->getGridSizeY(),
                cloudData->getGridSizeZ(), cloudData->getDensityField(),
                superVoxelSize);
    } else {
        superVoxelGridResidualRatioTracking = {};
        superVoxelGridDecompositionTracking = {};
    }
}

void VolumetricPathTracingPass::loadEnvironmentMapImage() {
    if (!sgl::FileUtils::get()->exists(environmentMapFilenameGui)) {
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::loadEnvironmentMapImage: The file \""
                + environmentMapFilenameGui + "\" does not exist.");
    }

    sgl::BitmapPtr bitmap(new sgl::Bitmap);
    if (sgl::FileUtils::get()->hasExtension(environmentMapFilenameGui.c_str(), ".png")) {
        bitmap = std::make_shared<sgl::Bitmap>();
        bitmap->fromFile(environmentMapFilenameGui.c_str());
    } else {
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::loadEnvironmentMapImage: The file \""
                + environmentMapFilenameGui + "\" has an unknown file extension.");
        return;
    }

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = uint32_t(bitmap->getWidth());
    imageSettings.height = uint32_t(bitmap->getHeight());
    imageSettings.imageType = VK_IMAGE_TYPE_2D;
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    sgl::vk::ImageSamplerSettings samplerSettings;
    samplerSettings.addressModeU = samplerSettings.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    environmentMapTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    environmentMapTexture->getImage()->uploadData(
            bitmap->getWidth() * bitmap->getHeight() * (bitmap->getBPP() / 8),
            bitmap->getPixels());
    loadedEnvironmentMapFilename = environmentMapFilenameGui;
    isEnvironmentMapLoaded = true;
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> customPreprocessorDefines = {
            { "COMPUTE_PRIMARY_RAY_ABSORPTION_MOMENTS", "" },
            { "COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS", "" },
            { "NUM_PRIMARY_RAY_ABSORPTION_MOMENTS",
              std::to_string(blitPrimaryRayMomentTexturePass->getNumMoments()) },
            { "NUM_SCATTER_RAY_ABSORPTION_MOMENTS",
              std::to_string(blitScatterRayMomentTexturePass->getNumMoments()) },
    };
    if (vptMode == VptMode::DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_DELTA_TRACKING", "" });
    } else if (vptMode == VptMode::SPECTRAL_DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_SPECTRAL_DELTA_TRACKING", "" });
    } else if (vptMode == VptMode::RATIO_TRACKING) {
        customPreprocessorDefines.insert({ "USE_RATIO_TRACKING", "" });
    } else if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
        customPreprocessorDefines.insert({ "USE_RESIDUAL_RATIO_TRACKING", "" });
    } else if (vptMode == VptMode::DECOMPOSITION_TRACKING) {
        customPreprocessorDefines.insert({ "USE_DECOMPOSITION_TRACKING", "" });
    }
    if (blitPrimaryRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
        customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_PRIMARY_RAY", "" });
    }
    if (blitScatterRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
        customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_SCATTER_RAY", "" });
    }
    if (useEnvironmentMapImage) {
        customPreprocessorDefines.insert({ "USE_ENVIRONMENT_MAP_IMAGE", "" });
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Clouds.Compute"}, customPreprocessorDefines);
}

void VolumetricPathTracingPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticImageView(resultImageView, "resultImage");
    computeData->setStaticTexture(densityFieldTexture, "gridImage");
    computeData->setStaticBuffer(uniformBuffer, "Parameters");
    computeData->setStaticBuffer(frameInfoBuffer, "FrameInfo");
    computeData->setStaticImageView(accImageTexture->getImageView(), "accImage");
    computeData->setStaticImageView(firstXTexture->getImageView(), "firstX");
    computeData->setStaticImageView(firstWTexture->getImageView(), "firstW");
    computeData->setStaticImageView(
            blitPrimaryRayMomentTexturePass->getMomentTexture()->getImageView(),
            "primaryRayAbsorptionMomentsImage");
    computeData->setStaticImageView(
            blitScatterRayMomentTexturePass->getMomentTexture()->getImageView(),
            "scatterRayAbsorptionMomentsImage");
    computeData->setStaticBuffer(momentUniformDataBuffer, "MomentUniformData");
    if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
        computeData->setStaticTexture(
                superVoxelGridResidualRatioTracking->getSuperVoxelGridTexture(),
                "superVoxelGridImage");
        computeData->setStaticTexture(
                superVoxelGridResidualRatioTracking->getSuperVoxelGridOccupancyTexture(),
                "superVoxelGridOccupancyImage");
    } else if (vptMode == VptMode::DECOMPOSITION_TRACKING) {
        computeData->setStaticTexture(
                superVoxelGridDecompositionTracking->getSuperVoxelGridTexture(),
                "superVoxelGridImage");
        computeData->setStaticTexture(
                superVoxelGridDecompositionTracking->getSuperVoxelGridOccupancyTexture(),
                "superVoxelGridOccupancyImage");
    }
    if (useEnvironmentMapImage) {
        computeData->setStaticTexture(environmentMapTexture, "environmentMapTexture");
    }
}

std::string VolumetricPathTracingPass::getCurrentEventName() {
    return std::string() + VPT_MODE_NAMES[int(vptMode)] + " " + std::to_string(targetNumSamples) + "spp";
}

void VolumetricPathTracingPass::_render() {
    if (denoiserChanged) {
        createDenoiser();
        denoiserChanged = false;
    }

    std::string eventName = getCurrentEventName();
    if (!reachedTarget) {
        accumulationTimer->startGPU(eventName);
    }

    if (!changedDenoiserSettings) {
        uniformData.inverseViewProjMatrix = glm::inverse(renderer->getProjectionMatrix() * renderer->getViewMatrix());
        VkExtent3D gridExtent = {
                cloudData->getGridSizeX(),
                cloudData->getGridSizeY(),
                cloudData->getGridSizeZ()
        };
        uint32_t maxDim = std::max(gridExtent.width, std::max(gridExtent.height, gridExtent.depth));
        uniformData.boxMax = glm::vec3(gridExtent.width, gridExtent.height, gridExtent.depth) * 0.25f / float(maxDim);
        uniformData.boxMin = -uniformData.boxMax;
        uniformData.extinction = cloudExtinctionBase * cloudExtinctionScale;
        uniformData.scatteringAlbedo = cloudScatteringAlbedo;
        uniformData.sunDirection = sunlightDirection;
        uniformData.sunIntensity = sunlightIntensity * sunlightColor;
        uniformData.environmentMapIntensityFactor = environmentMapIntensityFactor;
        if (superVoxelGridResidualRatioTracking) {
            uniformData.superVoxelSize = superVoxelGridResidualRatioTracking->getSuperVoxelSize();
            uniformData.superVoxelGridSize = superVoxelGridResidualRatioTracking->getSuperVoxelGridSize();
        } else if (superVoxelGridDecompositionTracking) {
            uniformData.superVoxelSize = superVoxelGridDecompositionTracking->getSuperVoxelSize();
            uniformData.superVoxelGridSize = superVoxelGridDecompositionTracking->getSuperVoxelGridSize();
        }
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());

        frameInfoBuffer->updateData(
                sizeof(FrameInfo), &frameInfo, renderer->getVkCommandBuffer());
        frameInfo.frameCount++;

        renderer->transitionImageLayout(resultImageView->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        renderer->transitionImageLayout(
                densityFieldTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        renderer->transitionImageLayout(accImageTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        renderer->transitionImageLayout(firstXTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        renderer->transitionImageLayout(firstWTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        renderer->transitionImageLayout(
                blitPrimaryRayMomentTexturePass->getMomentTexture()->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        renderer->transitionImageLayout(
                blitScatterRayMomentTexturePass->getMomentTexture()->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        auto& imageSettings = resultImageView->getImage()->getImageSettings();
        renderer->dispatch(
                computeData,
                sgl::iceil(int(imageSettings.width), blockSize2D.x),
                sgl::iceil(int(imageSettings.height), blockSize2D.y),
                1);
    }
    changedDenoiserSettings = false;

    if (featureMapType == FeatureMapType::RESULT) {
        if (useDenoiser && denoiser && denoiser->getIsEnabled()) {
            denoiser->denoise();
            renderer->transitionImageLayout(
                    denoisedImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(
                    sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            denoisedImageView->getImage()->blit(
                    sceneImageView->getImage(), renderer->getVkCommandBuffer());
        } else {
            renderer->transitionImageLayout(
                    resultImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(
                    sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            resultImageView->getImage()->blit(
                    sceneImageView->getImage(), renderer->getVkCommandBuffer());
        }
    } else if (featureMapType == FeatureMapType::FIRST_X) {
        renderer->transitionImageLayout(firstXTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        firstXTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapType::FIRST_W) {
        renderer->transitionImageLayout(firstWTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        firstWTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapType::PRIMARY_RAY_ABSORPTION_MOMENTS) {
        blitPrimaryRayMomentTexturePass->render();
    } else if (featureMapType == FeatureMapType::SCATTER_RAY_ABSORPTION_MOMENTS) {
        blitScatterRayMomentTexturePass->render();
    }

    if (!reachedTarget) {
        accumulationTimer->endGPU(eventName);
    }
}

void VolumetricPathTracingPass::renderGui() {
    sgl::ImGuiWrapper::get()->setNextWindowStandardPosSize(2, 14, 735, 564);

    bool optionChanged = false;

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseEnvironmentMapImage", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathName(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilter(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            // Is this line data set or a volume data file for the scattering line tracer?
            const char* currentPath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            filename += selection.table[0].fileName;
            IGFD_Selection_DestroyContent(&selection);
            if (currentPath) {
                free((void*)currentPath);
                currentPath = nullptr;
            }

            environmentMapFilenameGui = filename;
            loadEnvironmentMapImage();
            setShaderDirty();
            reRender = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (ImGui::Begin("VPT Renderer", &showWindow)) {
        std::string numSamplesText = "#Samples: " + std::to_string(frameInfo.frameCount);
        // + "##numSamplesText";
        ImGui::Text("%s", numSamplesText.c_str());
        ImGui::SameLine();
        if (ImGui::Button(" = ")) {
            accumulationTimer = std::make_shared<sgl::vk::Timer>(renderer);
            reachedTarget = false;
            reRender = true;

            if (int(frameInfo.frameCount) >= targetNumSamples) {
                frameInfo.frameCount = 0;
            }
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(220.0f);
        ImGui::InputInt("##targetNumSamples", &targetNumSamples);
        if (ImGui::ColorEdit3("Sunlight Color", &sunlightColor.x)) {
            optionChanged = true;
        }
        if (ImGui::SliderFloat("Sunlight Intensity", &sunlightIntensity, 0.0f, 10.0f)) {
            optionChanged = true;
        }
        if (ImGui::SliderFloat3("Sunlight Direction", &sunlightDirection.x, 0.0f, 1.0f)) {
            optionChanged = true;
        }
        if (ImGui::SliderFloat("Extinction Scale", &cloudExtinctionScale, 1.0f, 2048.0f)) {
            optionChanged = true;
        }
        if (ImGui::SliderFloat3("Extinction Base", &cloudExtinctionBase.x, 0.01f, 1.0f)) {
            optionChanged = true;
        }
        if (ImGui::ColorEdit3("Scattering Albedo", &cloudScatteringAlbedo.x)) {
            optionChanged = true;
        }
        if (ImGui::SliderFloat("G", &uniformData.G, 0.0f, 1.0f)) {
            optionChanged = true;
        }
        if (ImGui::Combo(
                "Feature Map", (int*)&featureMapType, FEATURE_MAP_NAMES,
                IM_ARRAYSIZE(FEATURE_MAP_NAMES))) {
            optionChanged = true;
            blitPrimaryRayMomentTexturePass->setVisualizeMomentTexture(
                    featureMapType == FeatureMapType::PRIMARY_RAY_ABSORPTION_MOMENTS);
            blitScatterRayMomentTexturePass->setVisualizeMomentTexture(
                    featureMapType == FeatureMapType::SCATTER_RAY_ABSORPTION_MOMENTS);
        }
        if (ImGui::Combo(
                "VPT Mode", (int*)&vptMode, VPT_MODE_NAMES,
                IM_ARRAYSIZE(VPT_MODE_NAMES))) {
            optionChanged = true;
            updateVptMode();
            setShaderDirty();
            setDataDirty();
        }
        if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
            if (ImGui::SliderInt("Super Voxel Size", &superVoxelSize, 1, 64)) {
                optionChanged = true;
                updateVptMode();
                setShaderDirty();
                setDataDirty();
            }
        }

        int numDenoisersSupported = IM_ARRAYSIZE(DENOISER_NAMES);
#ifdef SUPPORT_OPTIX
        if (!OptixVptDenoiser::isOpitEnabled()) {
            numDenoisersSupported--;
        }
#endif
        if (ImGui::Combo(
                "Denoiser", (int*)&denoiserType,
                DENOISER_NAMES, numDenoisersSupported)) {
            denoiserChanged = true;
            reRender = true;
            changedDenoiserSettings = true;
        }

        ImGui::InputText("Environment Map", &environmentMapFilenameGui);
        if (ImGui::Button("Open from Disk...")) {
            IGFD_OpenModal(
                    fileDialogInstance,
                    "ChooseEnvironmentMapImage", "Choose an Environment Map Image",
                    ".*,.png",
                    sgl::AppSettings::get()->getDataDirectory().c_str(),
                    "", 1, nullptr,
                    ImGuiFileDialogFlags_ConfirmOverwrite);
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            loadEnvironmentMapImage();
            setShaderDirty();
            reRender = true;
        }
        if (isEnvironmentMapLoaded && ImGui::Checkbox("Use Environment Map Image", &useEnvironmentMapImage)) {
            setShaderDirty();
            reRender = true;
        }
        if (useEnvironmentMapImage && ImGui::SliderFloat(
                "environmentMapIntensityFactor", &environmentMapIntensityFactor, 0.0f, 5.0f)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }

        bool shallRecreateMomentTextureA = false;
        bool momentTypeChangedA = false;
        bool shallRecreateMomentTextureB = false;
        bool momentTypeChangedB = false;
        optionChanged = blitPrimaryRayMomentTexturePass->renderGui(shallRecreateMomentTextureA, momentTypeChangedA)
                || optionChanged;
        optionChanged = blitScatterRayMomentTexturePass->renderGui(shallRecreateMomentTextureB, momentTypeChangedB)
                || optionChanged;
        if (shallRecreateMomentTextureA || shallRecreateMomentTextureB) {
            setShaderDirty();
            setDataDirty();
        }
        if (momentTypeChangedA || momentTypeChangedB) {
            setShaderDirty();
        }
    }
    ImGui::End();

    if (useDenoiser && denoiser) {
        bool denoiserReRender = denoiser->renderGui();
        reRender = denoiserReRender || reRender;
        changedDenoiserSettings = denoiserReRender || changedDenoiserSettings;
    }

    if (optionChanged) {
        frameInfo.frameCount = 0;
        reRender = true;
    }

    if (!reachedTarget) {
        if (int(frameInfo.frameCount) > targetNumSamples) {
            frameInfo.frameCount = 0;
        }
        if (int(frameInfo.frameCount) < targetNumSamples) {
            reRender = true;
        }
        if (int(frameInfo.frameCount) == targetNumSamples) {
            reachedTarget = true;
            std::string eventName = getCurrentEventName();
            accumulationTimer->finishGPU();
            accumulationTimer->printTimeMS(eventName);
        }
    }
}



BlitMomentTexturePass::BlitMomentTexturePass(sgl::vk::Renderer* renderer, std::string prefix)
        : BlitRenderPass(renderer, {"BlitMomentTexture.Vertex", "BlitMomentTexture.Fragment"}),
        prefix(std::move(prefix)) {
}

// Public interface.
void BlitMomentTexturePass::setOutputImage(sgl::vk::ImageViewPtr& colorImage) {
    BlitRenderPass::setOutputImage(colorImage);
    recreateMomentTexture();
}

void BlitMomentTexturePass::setVisualizeMomentTexture(bool visualizeMomentTexture) {
    this->visualizeMomentTexture = visualizeMomentTexture;
}

void BlitMomentTexturePass::recreateMomentTexture() {
    sgl::vk::ImageSamplerSettings samplerSettings;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    sgl::vk::ImageSettings imageSettings = outputImageViews.front()->getImage()->getImageSettings();
    imageSettings.format = VK_FORMAT_R32_SFLOAT;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageSettings.arrayLayers = numMoments + 1;
    momentTexture = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, VK_IMAGE_VIEW_TYPE_2D_ARRAY, samplerSettings);
    BlitRenderPass::setInputTexture(momentTexture);
}

bool BlitMomentTexturePass::renderGui(bool& shallRecreateMomentTexture, bool& momentTypeChanged) {
    bool reRender = false;
    shallRecreateMomentTexture = false;
    momentTypeChanged = false;

    std::string headerName = prefix;
    if (ImGui::CollapsingHeader(headerName.c_str(), nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
        std::string momentTypeName = "Moment Type## (" + prefix + ")";
        if (ImGui::Combo(
                momentTypeName.c_str(), (int*)&momentType, MOMENT_TYPE_NAMES,
                IM_ARRAYSIZE(MOMENT_TYPE_NAMES))) {
            reRender = true;
            momentTypeChanged = true;
        }
        std::string numMomentName = "#Moments## (" + prefix + ")";
        if (ImGui::Combo(
                numMomentName.c_str(), &numMomentsIdx, NUM_MOMENTS_NAMES,
                IM_ARRAYSIZE(NUM_MOMENTS_NAMES))) {
            numMoments = NUM_MOMENTS_SUPPORTED[numMomentsIdx];
            reRender = true;
            shallRecreateMomentTexture = true;
        }

        if (visualizeMomentTexture) {
            if (ImGui::SliderInt("Visualized Moment", &selectedMomentBlitIdx, 0, numMoments)) {
                reRender = true;
            }
        }
    }

    if (shallRecreateMomentTexture) {
        selectedMomentBlitIdx = std::min(selectedMomentBlitIdx, numMoments);
        recreateMomentTexture();
    }

    return reRender;
}

void BlitMomentTexturePass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    BlitRenderPass::createRasterData(renderer, graphicsPipeline);
}

void BlitMomentTexturePass::_render() {
    renderer->transitionImageLayout(momentTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    renderer->pushConstants(
            rasterData->getGraphicsPipeline(), VK_SHADER_STAGE_FRAGMENT_BIT,
            0, selectedMomentBlitIdx);
    BlitRenderPass::_render();
    renderer->transitionImageLayout(momentTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
}
