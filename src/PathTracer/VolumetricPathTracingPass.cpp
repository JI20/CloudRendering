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
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/RayTracingPipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>

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

    blitPrimaryRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Primary");
    blitScatterRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Scatter");

    createDenoiser();
    updateVptMode();
}

void VolumetricPathTracingPass::createDenoiser() {
    if (denoiserType == DenoiserType::NONE) {
        denoiser = {};
    } else if (denoiserType == DenoiserType::EAW) {
        denoiser = std::shared_ptr<Denoiser>(new EAWDenoiser(renderer));
    }
#ifdef SUPPORT_OPTIX
    else if (denoiserType == DenoiserType::OPTIX) {
        denoiser = std::shared_ptr<Denoiser>(new OptixVptDenoiser(renderer));
    }
#endif
    else {
        denoiser = {};
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::createDenoiser: Invalid denoiser type selected.");
    }

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
        superVoxelGrid = std::make_shared<SuperVoxelGrid>(
                device, cloudData->getGridSizeX(), cloudData->getGridSizeY(),
                cloudData->getGridSizeZ(), cloudData->getDensityField(),
                superVoxelSize);
        superVoxelGrid->setExtinction((cloudExtinctionBase * cloudExtinctionScale).x);
    } else {
        superVoxelGrid = std::shared_ptr<SuperVoxelGrid>();
    }
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
    if (blitPrimaryRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
        customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_PRIMARY_RAY", "" });
    }
    if (blitScatterRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
        customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_SCATTER_RAY", "" });
    }
    if (vptMode == VptMode::DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_DELTA_TRACKING", "" });
    } else if (vptMode == VptMode::SPECTRAL_DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_SPECTRAL_DELTA_TRACKING", "" });
    } else if (vptMode == VptMode::RATIO_TRACKING) {
    customPreprocessorDefines.insert({ "USE_RATIO_TRACKING", "" });
} else if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
        customPreprocessorDefines.insert({ "USE_RESIDUAL_RATIO_TRACKING", "" });
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
                superVoxelGrid->getSuperVoxelGridTexture(), "superVoxelGridImage");
        computeData->setStaticTexture(
                superVoxelGrid->getSuperVoxelGridEmptyTexture(), "superVoxelGridEmptyImage");
    }
}

void VolumetricPathTracingPass::_render() {
    if (denoiserChanged) {
        createDenoiser();
        denoiserChanged = false;
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
        if (superVoxelGrid) {
            uniformData.superVoxelSize = superVoxelGrid->getSuperVoxelSize();
            uniformData.superVoxelGridSize = superVoxelGrid->getSuperVoxelGridSize();
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
        if (useDenoiser && denoiser) {
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
}

void VolumetricPathTracingPass::renderGui() {
    sgl::ImGuiWrapper::get()->setNextWindowStandardPosSize(2, 14, 735, 564);

    bool optionChanged = false;

    if (ImGui::Begin("VPT Renderer", &showWindow)) {
        std::string numSamplesText = "Number of Samples: " + std::to_string(frameInfo.frameCount);
        // + "##numSamplesText";
        ImGui::Text("%s", numSamplesText.c_str());
        ImGui::SameLine();
        if (ImGui::Button(" = ")) {
            reachedTarget = false;
            reRender = true;
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

        if (ImGui::Combo(
                "Denoiser", (int*)&denoiserType,
                DENOISER_NAMES, IM_ARRAYSIZE(DENOISER_NAMES))) {
            denoiserChanged = true;
            reRender = true;
            changedDenoiserSettings = true;
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
