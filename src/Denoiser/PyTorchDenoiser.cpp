/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022-2023, Christoph Neuhauser, Timm Knörle
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

#include <memory>

#include <json/json.h>
#include <c10/core/MemoryFormat.h>
#ifdef SUPPORT_CUDA_INTEROP
#include <c10/cuda/CUDAStream.h>
#endif

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Texture/Bitmap.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>

#include "PyTorchDenoiser.hpp"

#ifdef USE_KPN_MODULE
#include "../../modules/kpn_module/src/perPixelKernel.hpp"
#endif

#if CUDA_VERSION >= 11020
#define USE_TIMELINE_SEMAPHORES
#elif defined(_WIN32)
#error Binary semaphore sharing is broken on Windows. Please install CUDA >= 11.2 for timeline semaphore support.
#endif

struct ModuleWrapper {
    torch::jit::Module module;
};

static torch::DeviceType getTorchDeviceType(PyTorchDevice pyTorchDevice) {
    if (pyTorchDevice == PyTorchDevice::CPU) {
        return torch::kCPU;
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (pyTorchDevice == PyTorchDevice::CUDA) {
        return torch::kCUDA;
    }
#endif
    else {
        sgl::Logfile::get()->writeError("Error in getTorchDeviceType: Unsupported device type.");
        return torch::kCPU;
    }
}

PyTorchDenoiser::PyTorchDenoiser(sgl::vk::Renderer* renderer) : renderer(renderer) {
    sgl::vk::Device* device = renderer->getDevice();

    sgl::AppSettings::get()->getSettings().getValueOpt("pyTorchDenoiserModelFilePath", modelFilePath);

#ifdef SUPPORT_CUDA_INTEROP
    // Support CUDA on NVIDIA GPUs using the proprietary driver.
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY && torch::cuda::is_available()) {
        pyTorchDevice = PyTorchDevice::CUDA;
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: "
                    "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
        }
    }
#endif

    renderFinishedFence = std::make_shared<sgl::vk::Fence>(device);
    featureCombinePass = std::make_shared<FeatureCombinePass>(renderer);
    backgroundAddPass = std::make_shared<BackgroundAddPass>(renderer);

    // When loading a model, the metadata in the TorchScript file is read to get the used feature map names.
    inputFeatureMapsUsed = { FeatureMapType::COLOR };
    for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
        inputFeatureMapsIndexMap.insert(std::make_pair(inputFeatureMapsUsed.at(i), i));
    }
    inputFeatureMaps.resize(inputFeatureMapsUsed.size());
    computeNumChannels();
}

PyTorchDenoiser::~PyTorchDenoiser() {
    sgl::AppSettings::get()->getSettings().addKeyValue("pyTorchDenoiserModelFilePath", modelFilePath);

    if (renderedImageData) {
        delete[] renderedImageData;
        renderedImageData = nullptr;
    }
    if (denoisedImageData) {
        delete[] denoisedImageData;
        denoisedImageData = nullptr;
    }
}

void PyTorchDenoiser::setOutputImage(sgl::vk::ImageViewPtr& outputImage) {
    outputImageVulkan = outputImage;
    backgroundAddPass->setTargetImage(outputImageVulkan);
}

void PyTorchDenoiser::setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) {
    // COLOR and CLOUDONLY seem to be mutually exclusive at the moment.
    if (featureMapType == FeatureMapType::COLOR || featureMapType == FeatureMapType::CLOUDONLY) {
        inputImageVulkan = featureTexture->getImageView();
    }
    if (featureMapType == FeatureMapType::BACKGROUND) {
        backgroundImage = featureTexture->getImageView();
        backgroundAddPass->setBackgroundImage(backgroundImage);
    }
    if (inputFeatureMapsIndexMap.find(featureMapType) != inputFeatureMapsIndexMap.end()) {
        inputFeatureMaps.at(inputFeatureMapsIndexMap.find(featureMapType)->second) = featureTexture;
        featureCombinePass->setFeatureMap(featureMapType, featureTexture);
    }
}

bool PyTorchDenoiser::getUseFeatureMap(FeatureMapType featureMapType) const {
    return inputFeatureMapsIndexMap.find(featureMapType) != inputFeatureMapsIndexMap.end()
            || featureMapType == FeatureMapType::BACKGROUND; // TODO: Find nicer way to do this
}

void PyTorchDenoiser::computeNumChannels() {
    inputFeatureMapsChannelOffset.clear();
    inputFeatureMapsChannelOffset.resize(inputFeatureMapsUsed.size());
    numChannels = 0;
    for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
        inputFeatureMapsChannelOffset.at(i) = numChannels;
        auto featureMap = inputFeatureMapsUsed.at(i);
        numChannels += FEATURE_MAP_NUM_CHANNELS[int(featureMap)];
    }
    featureCombinePass->setUsedInputFeatureMaps(
            inputFeatureMapsUsed, inputFeatureMapsIndexMap, inputFeatureMapsChannelOffset, numChannels);
}

void PyTorchDenoiser::setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) {
    // Ignore, as we can't easily turn on or off individual feature maps of the loaded model.
}

#ifdef USE_KPN_MODULE
torch::Tensor inv_iter_kernel(torch::Tensor x, torch::Tensor low_res_pred, torch::Tensor low_res_x, torch::Tensor prev, torch::Tensor fused_weights, int kernelSize, bool usePrevious) {
    //torch::Tensor base = x;
    torch::Tensor base = getInvIterBaseCuda(x, low_res_pred, low_res_x, prev, fused_weights, usePrevious, kernelSize);
    torch::Tensor weights = torch::softmax(fused_weights.index({torch::indexing::Slice(),torch::indexing::Slice(0,kernelSize*kernelSize)}), 1);
    return perPixelKernelCuda(base, weights, kernelSize);
}
#endif

void PyTorchDenoiser::denoise() {
    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t frameIndex = swapchain ? swapchain->getImageIndex() : 0;

    torch::DeviceType deviceType = getTorchDeviceType(pyTorchDevice);
    torch::Tensor inputTensor;

    uint32_t width = inputImageVulkan->getImage()->getImageSettings().width;
    uint32_t height = inputImageVulkan->getImage()->getImageSettings().height;

    std::vector<int64_t> inputSizes;
    if (useBatchDimension) {
        inputSizes = { 1, int64_t(height), int64_t(width), int64_t(numChannels) };
    } else {
        inputSizes = { int64_t(height), int64_t(width), int64_t(numChannels) };
    }

    // Copy color and auxiliary images from Vulkan to PyTorch.
    if (pyTorchDevice == PyTorchDevice::CPU) {
        for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
            const sgl::vk::TexturePtr& featureMap = inputFeatureMaps.at(i);
            renderer->transitionImageLayout(featureMap->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            featureMap->getImage()->copyToBuffer(
                    renderImageStagingBuffers.at(i), renderer->getVkCommandBuffer());
        }

        // Submit the rendering operations in Vulkan.
        renderer->endCommandBuffer();
        renderer->submitToQueue(
                {}, {}, renderFinishedFence,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        renderFinishedFence->wait();
        renderFinishedFence->reset();

        for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
            const sgl::vk::BufferPtr& renderImageStagingBuffer = renderImageStagingBuffers.at(i);
            auto* mappedData = (float*)renderImageStagingBuffer->mapMemory();
            uint32_t numChannelsFeature = FEATURE_MAP_NUM_CHANNELS[int(inputFeatureMapsUsed.at(i))];
            uint32_t numChannelsFeaturePadded = FEATURE_MAP_NUM_CHANNELS_PADDED[int(inputFeatureMapsUsed.at(i))];
            uint32_t c0 = inputFeatureMapsChannelOffset.at(i);
            uint32_t c1 = c0 + numChannelsFeature;
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t readLocation = (x + y * width) * numChannelsFeaturePadded;
                    uint32_t writeLocation = (x + y * width) * numChannels;
                    for (uint32_t c = c0; c < c1; c++) {
                        renderedImageData[writeLocation + c] = mappedData[readLocation + (c - c0)];
                    }
                }
            }
            renderImageStagingBuffer->unmapMemory();
        }

        inputTensor = torch::from_blob(
                renderedImageData, inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(deviceType).memory_format(
                        c10::MemoryFormat::ChannelsLast));
        if (useBatchDimension) {
            inputTensor = inputTensor.permute({0, 3, 1, 2}); // (n, h, w, c) -> (n, c, h, w)
        } else {
            inputTensor = inputTensor.permute({2, 0, 1}); // (h, w, c) -> (c, h, w)
        }
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (pyTorchDevice == PyTorchDevice::CUDA) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        timelineValue++;

        if (false) {
            renderer->transitionImageLayout(inputImageVulkan->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            inputImageVulkan->getImage()->copyToBuffer(
                    inputImageBufferVk, renderer->getVkCommandBuffer());
        } else {
            // TODO: Testing
            /*renderImageStagingBuffers.resize(inputFeatureMapsUsed.size());
            sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
            for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
                uint32_t numChannelsFeaturePadded = FEATURE_MAP_NUM_CHANNELS_PADDED[int(inputFeatureMapsUsed.at(i))];
                renderImageStagingBuffers.at(i) = std::make_shared<sgl::vk::Buffer>(
                        device, sizeof(float) * width * height * numChannelsFeaturePadded,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

                const sgl::vk::TexturePtr& featureMap = inputFeatureMaps.at(i);
                renderer->transitionImageLayout(featureMap->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                featureMap->getImage()->copyToBuffer(
                        renderImageStagingBuffers.at(i), renderer->getVkCommandBuffer());
            }
            renderer->syncWithCpu();
            for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
                uint32_t numChannelsFeaturePadded = FEATURE_MAP_NUM_CHANNELS_PADDED[int(inputFeatureMapsUsed.at(i))];
                renderImageStagingBuffers.at(i) = std::make_shared<sgl::vk::Buffer>(
                        device, sizeof(float) * width * height * numChannelsFeaturePadded,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

                sgl::BitmapPtr bitmap(new sgl::Bitmap(width, height, 32));
                auto* pixels = bitmap->getPixels();
                const sgl::vk::BufferPtr& renderImageStagingBuffer = renderImageStagingBuffers.at(i);
                auto* mappedData = (float*)renderImageStagingBuffer->mapMemory();
                uint32_t numChannelsFeature = FEATURE_MAP_NUM_CHANNELS[int(inputFeatureMapsUsed.at(i))];
                for (uint32_t y = 0; y < height; y++) {
                    for (uint32_t x = 0; x < width; x++) {
                        uint32_t writeLocation = (x + y * width) * 4;
                        pixels[writeLocation + 0] = 0;
                        pixels[writeLocation + 1] = 0;
                        pixels[writeLocation + 2] = 0;
                        pixels[writeLocation + 3] = 255;
                    }
                }
                for (uint32_t y = 0; y < height; y++) {
                    for (uint32_t x = 0; x < width; x++) {
                        uint32_t readLocation = (x + y * width) * numChannelsFeaturePadded;
                        uint32_t writeLocation = (x + y * width) * 4;
                        for (uint32_t c = 0; c < numChannelsFeature; c++) {
                            pixels[writeLocation + c] = std::clamp(int(mappedData[readLocation + c] * 255), 0, 255);
                        }
                    }
                }
                renderImageStagingBuffer->unmapMemory();
                std::string filename = std::string() + FEATURE_MAP_NAMES[int(inputFeatureMapsUsed.at(i))] + ".png";
                bitmap->savePNG(filename.c_str());
            }*/

            for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
                const sgl::vk::TexturePtr& featureMap = inputFeatureMaps.at(i);
                renderer->transitionImageLayout(featureMap->getImage(), VK_IMAGE_LAYOUT_GENERAL);
            }
            // Use shader that combines all feature maps.
            featureCombinePass->render();
        }

        sgl::vk::CommandBufferPtr commandBufferPreDenoise = renderer->getCommandBuffer();
        sgl::vk::SemaphoreVkCudaDriverApiInteropPtr renderFinishedSemaphore = renderFinishedSemaphores.at(frameIndex);
#ifdef USE_TIMELINE_SEMAPHORES
        renderFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
#endif
        commandBufferPreDenoise->pushSignalSemaphore(renderFinishedSemaphore);
        renderer->endCommandBuffer();

        /*
         * 'model->forward()' uses device caching allocators, which may call functions like 'cudaStreamIsCapturing',
         * which enforce synchronization of the stream with the host if more memory needs to be allocated. Thus, we must
         * submit the Vulkan command buffers before this function, as otherwise, CUDA will wait forever on the render
         * finished semaphore!
         */
        renderer->submitToQueue();

        inputTensor = torch::from_blob(
                (void*)inputImageBufferCu->getCudaDevicePtr(), inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(deviceType).memory_format(
                        c10::MemoryFormat::ChannelsLast));
        if (useBatchDimension) {
            inputTensor = inputTensor.permute({0, 3, 1, 2}); // (n, h, w, c) -> (n, c, h, w)
        } else {
            inputTensor = inputTensor.permute({2, 0, 1}); // (h, w, c) -> (c, h, w)
        }

#ifdef USE_TIMELINE_SEMAPHORES
        renderFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);
#else
        renderFinishedSemaphore->waitSemaphoreCuda(stream);
#endif
    }
#endif

    std::vector<torch::jit::IValue> inputs;
    if (useFP16){
        inputTensor = inputTensor.toType(torch::kFloat16);
    }
    if (!inputTensor.is_contiguous()){
        //std::cout << "input not contiguous" << std::endl;
        inputTensor = inputTensor.contiguous();
    }
    inputs.emplace_back(inputTensor);
#ifdef USE_KPN_MODULE
    bool hasPreviousFrame = true;
#endif
    if (usePreviousFrame) {
        //std::cout << width << ", " << height << std::endl;
        //std::cout << previousTensor.sizes() << std::endl;

        if (previousTensor.sizes()[0] == 0 || zeroOutPreviousFrame) {
            //std::cout <<"new tensor: "<< width << ", " << height << std::endl;
#ifdef USE_KPN_MODULE
            hasPreviousFrame = false;
#endif
            if (useBatchDimension) {
                inputSizes = { 1, int64_t(4), int64_t(height), int64_t(width)};
                previousTensor = torch::zeros(inputSizes, torch::TensorOptions().dtype(torch::kFloat32).device(deviceType));
            } else {
                inputSizes = { int64_t(4), int64_t(height), int64_t(width)};
                previousTensor = torch::zeros(inputSizes, torch::TensorOptions().dtype(torch::kFloat32).device(deviceType));
            }
        }
        if (useFP16){
            previousTensor = previousTensor.toType(torch::kFloat16);
        }else{
            previousTensor = previousTensor.toType(torch::kFloat32);
        }
        inputs.emplace_back(previousTensor);
    }
    if (useFP16) {
        wrapper->module.to(torch::kFloat16);
    } else {
        wrapper->module.to(torch::kFloat32);
    }
    at::Tensor outputTensor;
    if (blend_inv_iter){
#ifdef USE_KPN_MODULE
        c10::IValue layer_outputs = wrapper->module.forward(inputs);
        at::Tensor lprev = layer_outputs.toTuple()->elements()[0].toTensor().to(torch::kFloat32);
        at::Tensor l2 = layer_outputs.toTuple()->elements()[1].toTensor().to(torch::kFloat32);
        at::Tensor l1 = layer_outputs.toTuple()->elements()[2].toTensor().to(torch::kFloat32);
        at::Tensor l0 = layer_outputs.toTuple()->elements()[3].toTensor().to(torch::kFloat32);
        inputTensor = inputTensor.to(torch::kFloat32);
        at::Tensor x0 = inputTensor.index({torch::indexing::Slice(),torch::indexing::Slice(0,4)});
        at::Tensor x1 = torch::avg_pool2d(x0, 2, 2);
        at::Tensor x2 = torch::avg_pool2d(x1, 2, 2);

        int kernelSize = 5;
        if (l0.size(1) < 25){
            kernelSize = 3;
        }

        torch::Tensor reproj_uv = inputTensor.index({torch::indexing::Slice(),torch::indexing::Slice(4,6)});
        torch::Tensor reproj0 = torch::grid_sampler_2d(previousTensor.to(torch::kFloat32), reproj_uv.permute({0,2,3,1}) * 2 - 1, 0,0, false);

        reproj0 = perPixelKernelForward(reproj0, torch::softmax(lprev, 1), kernelSize);
        torch::Tensor reproj1 = torch::avg_pool2d(reproj0, 2, 2);

        at::Tensor kp2 = perPixelKernelForward(x2, torch::softmax(l2, 1), kernelSize);
        at::Tensor kp1 = inv_iter_kernel(x1, kp2, x2, reproj1, l1, kernelSize, hasPreviousFrame);
        at::Tensor kp0 = inv_iter_kernel(x0, kp1, x1, reproj0, l0, kernelSize, hasPreviousFrame);

        outputTensor = kp0;
#endif
    } else{
        outputTensor = wrapper->module.forward(inputs).toTensor().to(torch::kFloat32);
    }
    previousTensor = outputTensor.clone().detach();
    //std::cout << "got " << previousTensor.sizes() << std::endl;
    uint32_t outputTensorWidth = 0;
    uint32_t outputTensorHeight = 0;
    uint32_t outputTensorChannels = 0;
    if (useBatchDimension) {
        if (outputTensor.sizes().size() != 4) {
            sgl::Logfile::get()->throwError(
                    "Error in PyTorchDenoiser::denoise: Expected 4 output dimensions, but got "
                    + std::to_string(outputTensor.sizes().size()) + ".");
        }
        if (outputTensor.size(0) != 1) {
            sgl::Logfile::get()->throwError(
                    "Error in PyTorchDenoiser::denoise: Expected a batch dimension of 1, but got "
                    + std::to_string(outputTensor.size(0)) + ".");
        }
        outputTensorChannels = uint32_t(outputTensor.size(1));
        outputTensorHeight = uint32_t(outputTensor.size(2));
        outputTensorWidth = uint32_t(outputTensor.size(3));
        outputTensor = outputTensor.permute({0, 2, 3, 1}); // (n, c, h, w) -> (n, h, w, c)
    } else {
        if (outputTensor.sizes().size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in PyTorchDenoiser::denoise: Expected 3 output dimensions, but got "
                    + std::to_string(outputTensor.sizes().size()) + ".");
        }
        outputTensorChannels = uint32_t(outputTensor.size(0));
        outputTensorHeight = uint32_t(outputTensor.size(1));
        outputTensorWidth = uint32_t(outputTensor.size(2));
        outputTensor = outputTensor.permute({1, 2, 0}); // (c, h, w) -> (h, w, c)
    }
    if (outputTensorChannels != 4) {
        //std::cout << "got " << outputTensorChannels << " channels" << std::endl;
        sgl::Logfile::get()->throwError("Error in PyTorchDenoiser::denoise: Mismatch in output tensor channels.");
    }
    if (outputTensorWidth != width || outputTensorHeight != height) {
        sgl::Logfile::get()->throwError("Error in PyTorchDenoiser::denoise: Mismatch in output tensor sizes.");
    }
    if (!outputTensor.is_contiguous()) {
        if (isFirstContiguousWarning) {
            sgl::Logfile::get()->writeWarning("Error in PyTorchDenoiser::denoise: Output tensor is not contiguous.");
            isFirstContiguousWarning = false;
        }
        outputTensor = outputTensor.contiguous();
    }

    // Read back denoised color image to Vulkan.
    if (pyTorchDevice == PyTorchDevice::CPU) {
        void* stagingBufferData = denoisedImageStagingBuffer->mapMemory();
        memcpy(stagingBufferData, outputTensor.data_ptr(), sizeof(float) * width * height * 4);
        denoisedImageStagingBuffer->unmapMemory();

        renderer->beginCommandBuffer();
        renderer->transitionImageLayout(
                outputImageVulkan->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        outputImageVulkan->getImage()->copyFromBuffer(
                denoisedImageStagingBuffer, renderer->getVkCommandBuffer());
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (pyTorchDevice == PyTorchDevice::CUDA) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
                outputImageBufferCu->getCudaDevicePtr(), (CUdeviceptr)outputTensor.data_ptr(),
                outputImageBufferVk->getSizeInBytes(), stream);

        sgl::vk::SemaphoreVkCudaDriverApiInteropPtr denoiseFinishedSemaphore =
                denoiseFinishedSemaphores.at(frameIndex);
#ifdef USE_TIMELINE_SEMAPHORES
        denoiseFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
#else
        denoiseFinishedSemaphore->signalSemaphoreCuda(stream);
#endif

        sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
#ifdef USE_TIMELINE_SEMAPHORES
        denoiseFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
#endif
        renderer->pushCommandBuffer(postRenderCommandBuffer);
        renderer->beginCommandBuffer();
        postRenderCommandBuffer->pushWaitSemaphore(
                denoiseFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        renderer->transitionImageLayout(outputImageVulkan, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        outputImageVulkan->getImage()->copyFromBuffer(
                outputImageBufferVk, renderer->getVkCommandBuffer());

        if (addBackground) {
            renderer->transitionImageLayout(backgroundImage, VK_IMAGE_LAYOUT_GENERAL);
            renderer->transitionImageLayout(outputImageVulkan, VK_IMAGE_LAYOUT_GENERAL);

            backgroundAddPass->render();
        }
    }
#endif
}

void PyTorchDenoiser::recreateSwapchain(uint32_t width, uint32_t height) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (renderedImageData) {
        delete[] renderedImageData;
        renderedImageData = nullptr;
    }
    if (denoisedImageData) {
        delete[] denoisedImageData;
        denoisedImageData = nullptr;
    }
    renderImageStagingBuffers = {};
    denoisedImageStagingBuffer = {};
#ifdef SUPPORT_CUDA_INTEROP
    outputImageBufferCu = {};
    outputImageBufferVk = {};
    postRenderCommandBuffers = {};
    renderFinishedSemaphores = {};
    denoiseFinishedSemaphores = {};
    timelineValue = 0;
#endif

    if (pyTorchDevice == PyTorchDevice::CPU) {
        renderImageStagingBuffers.resize(inputFeatureMapsUsed.size());
        for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
            uint32_t numChannelsFeaturePadded = FEATURE_MAP_NUM_CHANNELS_PADDED[int(inputFeatureMapsUsed.at(i))];
            renderImageStagingBuffers.at(i) = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(float) * width * height * numChannelsFeaturePadded,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }

        denoisedImageStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * width * height * 4,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        renderedImageData = new float[width * height * numChannels];
        denoisedImageData = new float[width * height * 4];
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (pyTorchDevice == PyTorchDevice::CUDA) {
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            bool success = sgl::vk::initializeCudaDeviceApiFunctionTable();
            if (!success) {
                sgl::Logfile::get()->throwError(
                        "Error in PyTorchDenoiser::recreateSwapchain: "
                        "sgl::vk::initializeCudaDeviceApiFunctionTable() failed.", false);
            }
        }

        inputImageBufferCu = {};
        inputImageBufferVk = std::make_shared<sgl::vk::Buffer>(
                device, width * height * numChannels * sizeof(float),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        inputImageBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(inputImageBufferVk);
        featureCombinePass->setOutputBuffer(inputImageBufferVk);

        outputImageBufferCu = {};
        outputImageBufferVk = std::make_shared<sgl::vk::Buffer>(
                device, width * height * 4 * sizeof(float),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        outputImageBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(outputImageBufferVk);

        sgl::vk::Swapchain *swapchain = sgl::AppSettings::get()->getSwapchain();
        size_t numImages = swapchain ? swapchain->getNumImages() : 1;

        sgl::vk::CommandPoolType commandPoolType;
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (!swapchain) {
            commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
        }
        for (size_t frameIdx = 0; frameIdx < numImages; frameIdx++) {
            postRenderCommandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
#ifdef USE_TIMELINE_SEMAPHORES
            denoiseFinishedSemaphores.push_back(std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                    device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
                    timelineValue));
            renderFinishedSemaphores.push_back(std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                    device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
                    timelineValue));
#else
            denoiseFinishedSemaphores.push_back(std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(device));
            renderFinishedSemaphores.push_back(std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(device));
#endif
        }
    }
#endif
    else {
        sgl::Logfile::get()->throwError(
                "Error in PyTorchDenoiser::recreateSwapchain: Unsupported PyTorch device type.",
                false);
    }
}

bool PyTorchDenoiser::loadModelFromFile(const std::string& modelPath) {
    torch::DeviceType deviceType = getTorchDeviceType(pyTorchDevice);
    torch::jit::ExtraFilesMap extraFilesMap;
    extraFilesMap["model_info.json"] = "";
    wrapper = std::make_shared<ModuleWrapper>();
    try {
        // std::shared_ptr<torch::jit::script::Module>
        wrapper->module = torch::jit::load(modelPath, deviceType, extraFilesMap);
        wrapper->module.to(deviceType);
    } catch (const c10::Error& e) {
        sgl::Logfile::get()->writeError("Error: Couldn't load the PyTorch module from \"" + modelPath + "\"!");
        sgl::Logfile::get()->writeError(std::string() + "What: " + e.what());
        wrapper = {};
        return false;
    } catch (const torch::jit::ErrorReport& e) {
        sgl::Logfile::get()->writeError("Error: Couldn't load the PyTorch module from \"" + modelPath + "\"!");
        sgl::Logfile::get()->writeError(std::string() + "What: " + e.what());
        sgl::Logfile::get()->writeError("Call stack: " + e.current_call_stack());
        wrapper = {};
        return false;
    }

    // Read the model JSON metadata next.
    auto inputFeatureMapsUsedOld = inputFeatureMapsUsed;
    inputFeatureMapsUsed.clear();
    inputFeatureMapsIndexMap.clear();
    auto it = extraFilesMap.find("model_info.json");
    if (it == extraFilesMap.end()) {
        sgl::Logfile::get()->writeError(
                "Error: Couldn't find model_info.json in the PyTorch module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }

    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> const charReader(builder.newCharReader());
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!charReader->parse(
            it->second.c_str(), it->second.c_str() + it->second.size(), &root, &errorString)) {
        sgl::Logfile::get()->writeError("Error in PyTorchDenoiser::loadModelFromFile: " + errorString);
        wrapper = {};
        return false;
    }

    /*
     * Set the blend type, if none provided, assume color output.
     */
    blend_inv_iter = false;
    if (root.isMember("blend")) {
        Json::Value& blendType = root["blend"];
        if (!blendType.isString()) {
            sgl::Logfile::get()->writeError(
                    "Error: Element 'blendtype' should be a string in \"" + modelPath + "\"!");
            wrapper = {};
            return false;
        }
        std::string bt = sgl::toLowerCopy(blendType.asString());
        if (bt == "inv_iter") {
#ifdef USE_KPN_MODULE
            blend_inv_iter = true;
#else
            sgl::Logfile::get()->writeError(
                    "Error: Blend type 'inv_iter' is used in \"" + modelPath
                    + "\", but the program is compiled without the KPN module.");
            wrapper = {};
            return false;
#endif
            std::cout << "blending inv_iter" << std::endl;
        }
    }

    // Check whether the previous frame needs to be provided.
    usePreviousFrame = false;
    if (root.isMember("use_previous_frame")) {
        Json::Value& usePreviousFrameElement = root["use_previous_frame"];
        if (!usePreviousFrameElement.isBool()) {
            sgl::Logfile::get()->writeError(
                    "Error: Element 'use_previous_frame' should be a bool in \"" + modelPath + "\"!");
            wrapper = {};
            return false;
        }
        usePreviousFrame = usePreviousFrameElement.asBool();
    }

    // Does the model use 3 or 4 input dimensions?
    useBatchDimension = false;
    if (wrapper->module.parameters().size() > 0) {
        auto paramSizes = (*wrapper->module.parameters().begin()).sizes();
        if (paramSizes.size() == 4) {
            useBatchDimension = true;
        }
    }
    if (root.isMember("use_batch_dimension")) {
        Json::Value& useBatchDimensionElement = root["use_batch_dimension"];
        if (!useBatchDimensionElement.isBool()) {
            sgl::Logfile::get()->writeError(
                    "Error: Element 'use_batch_dimension' should be a bool in \"" + modelPath + "\"!");
            wrapper = {};
            return false;
        }
        useBatchDimension = useBatchDimensionElement.asBool();
    }

    // Should FP16 be used by default (e.g., due to high memory consumption otherwise)?
    if (root.isMember("use_fp16")) {
        Json::Value& useFP16Element = root["use_fp16"];
        if (!useFP16Element.isBool()) {
            sgl::Logfile::get()->writeError(
                    "Error: Element 'use_fp16' should be a bool in \"" + modelPath + "\"!");
            wrapper = {};
            return false;
        }
        useFP16 = useFP16Element.asBool();
    }

    /*
     * Example: { "input_feature_maps": [ "color", "normal" ] }
     */
    if (!root.isMember("input_feature_maps")) {
        sgl::Logfile::get()->writeError(
                "Error: Array 'input_feature_maps' could not be found in the model_info.json file of the PyTorch "
                "module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }
    Json::Value& inputFeatureMapsNode = root["input_feature_maps"];
    if (!inputFeatureMapsNode.isArray()) {
        sgl::Logfile::get()->writeError(
                "Error: 'input_feature_maps' is not an array in the model_info.json file of the PyTorch "
                "module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }

    for (Json::Value& inputFeatureMapNode : inputFeatureMapsNode) {
        if (!inputFeatureMapNode.isString()) {
            sgl::Logfile::get()->writeError(
                    "Error: A child of the array 'input_feature_maps' in the model_info.json file of the PyTorch "
                    "module loaded from \"" + modelPath + "\" is not a string!");
            wrapper = {};
            return false;
        }
        std::string inputFeatureMapName = sgl::toLowerCopy(inputFeatureMapNode.asString());
        int i;
        for (i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
            if (sgl::toLowerCopy(std::string() + FEATURE_MAP_NAMES[i]) == inputFeatureMapName) {
                inputFeatureMapsUsed.push_back(FeatureMapType(i));
                break;
            }
        }
        if (i == IM_ARRAYSIZE(FEATURE_MAP_NAMES)) {
            std::string errorStringLogfile = "Error: Invalid feature map name '" + inputFeatureMapName;
            errorStringLogfile +=
                    "' found in the model_info.json file of the PyTorch module loaded from \"" + modelPath + "\"!";
            sgl::Logfile::get()->writeError(errorStringLogfile);
            wrapper = {};
            return false;
        }
    }

    for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
        inputFeatureMapsIndexMap.insert(std::make_pair(inputFeatureMapsUsed.at(i), i));
    }
    inputFeatureMaps.resize(inputFeatureMapsUsed.size());
    computeNumChannels();
    if (inputFeatureMapsUsed != inputFeatureMapsUsedOld && inputImageVulkan) {
        renderer->getDevice()->waitIdle();
        recreateSwapchain(
                inputImageVulkan->getImage()->getImageSettings().width,
                inputImageVulkan->getImage()->getImageSettings().height);
    }
    isFirstContiguousWarning = true;

    return true;
}

void PyTorchDenoiser::setPyTorchDevice(PyTorchDevice pyTorchDeviceNew) {
    if (pyTorchDeviceNew == pyTorchDevice) {
        return;
    }
    
    pyTorchDevice = pyTorchDeviceNew;
    if (inputImageVulkan) {
        renderer->getDevice()->waitIdle();
        recreateSwapchain(
                inputImageVulkan->getImage()->getImageSettings().width,
                inputImageVulkan->getImage()->getImageSettings().height);
    }
    if (wrapper) {
        wrapper->module.to(getTorchDeviceType(pyTorchDevice));
    }
}

bool PyTorchDenoiser::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    bool reRender = false;

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChoosePyTorchModelFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            // Is this line data set or a volume data file for the scattering line tracer?
            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            filename += selection.table[0].fileName;
            IGFD_Selection_DestroyContent(&selection);

            fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            modelFilePath = filename;
            loadModelFromFile(modelFilePath);
            reRender = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    propertyEditor.addInputAction("Model Path", &modelFilePath);
    if (propertyEditor.addButton("", "Load")) {
        loadModelFromFile(modelFilePath);
        reRender = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Open from Disk...")) {
        if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "LineDataSets/";
            if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
                fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
            }
        }
        IGFD_OpenModal(
                fileDialogInstance,
                "ChoosePyTorchModelFile", "Choose TorchScript Model File",
                ".*,.pt,.pth",
                fileDialogDirectory.c_str(),
                "", 1, nullptr,
                ImGuiFileDialogFlags_None);
    }

    PyTorchDevice pyTorchDeviceNew = pyTorchDevice;
    if (propertyEditor.addCombo(
            "Device", (int*)&pyTorchDeviceNew,
            PYTORCH_DEVICE_NAMES, IM_ARRAYSIZE(PYTORCH_DEVICE_NAMES))) {
        setPyTorchDevice(pyTorchDeviceNew);
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Add Background", &addBackground)) {
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Zero Out Previous Frame", &zeroOutPreviousFrame)) {
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Use FP16", &useFP16)) {
        reRender = true;
    }

    return reRender;
}



FeatureCombinePass::FeatureCombinePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

uint32_t FeatureCombinePass::getFeatureMapWriteOffset(FeatureMapType featureMapType) {
    auto it = inputFeatureMapsIndexMap.find(featureMapType);
    if (it == inputFeatureMapsIndexMap.end()) {
        return 0;
    }
    return inputFeatureMapsChannelOffset.at(it->second);
}

bool FeatureCombinePass::getUseFeatureMap(FeatureMapType featureMapType) {
    return inputFeatureMapsIndexMap.find(featureMapType) != inputFeatureMapsIndexMap.end();
}

std::string FeatureCombinePass::getFeatureMapImageName(FeatureMapType featureMapType) {
    std::string baseName = FEATURE_MAP_NAMES[int(featureMapType)];
    return sgl::toLowerCopy(baseName) + "Map";
}

void FeatureCombinePass::setUsedInputFeatureMaps(
        const std::vector<FeatureMapType>& _inputFeatureMapsUsed,
        const std::unordered_map<FeatureMapType, size_t>& _inputFeatureMapsIndexMap,
        const std::vector<uint32_t>& _inputFeatureMapsChannelOffset,
        uint32_t _numChannels) {
    setShaderDirty();
    inputFeatureMapsUsed = _inputFeatureMapsUsed;
    inputFeatureMapsIndexMap = _inputFeatureMapsIndexMap;
    inputFeatureMapsChannelOffset = _inputFeatureMapsChannelOffset;
    numChannels = _numChannels;
    inputFeatureMaps.resize(inputFeatureMapsUsed.size());
    uniformData.numChannelsOut = numChannels;
    uniformData.colorWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::COLOR);
    uniformData.normalWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::NORMAL);
    uniformData.depthWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::DEPTH);
    uniformData.positionWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::POSITION);
    uniformData.densityWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::DENSITY);
    uniformData.cloudOnlyWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::CLOUDONLY);
    uniformData.albedoWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::ALBEDO);
    uniformData.flowWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::FLOW);
    uniformData.reproj_uvWriteStartOffset = getFeatureMapWriteOffset(FeatureMapType::REPROJ_UV);

    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void FeatureCombinePass::setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) {
    inputFeatureMaps.at(inputFeatureMapsIndexMap.find(featureMapType)->second) = featureTexture;
    if (computeData) {
        computeData->setStaticImageViewOptional(
                featureTexture->getImageView(), getFeatureMapImageName(featureMapType));
    }
}

void FeatureCombinePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void FeatureCombinePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    for (int i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
        auto featureMapType = FeatureMapType(i);
        auto it = inputFeatureMapsIndexMap.find(featureMapType);
        if (it != inputFeatureMapsIndexMap.end()) {
            std::string baseName = FEATURE_MAP_NAMES[i];
            preprocessorDefines.insert(std::make_pair("USE_" + sgl::toUpperCopy(baseName), ""));
        }
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "FeatureMapMerge.Compute" }, preprocessorDefines);
}

void FeatureCombinePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    for (int i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
        auto featureMapType = FeatureMapType(i);
        auto it = inputFeatureMapsIndexMap.find(featureMapType);
        if (it != inputFeatureMapsIndexMap.end()) {
            computeData->setStaticImageView(
                    inputFeatureMaps.at(it->second)->getImageView(),
                    getFeatureMapImageName(featureMapType));
        }
    }
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void FeatureCombinePass::_render() {
    auto width = int(inputFeatureMaps.front()->getImage()->getImageSettings().width);
    auto height = int(inputFeatureMaps.front()->getImage()->getImageSettings().height);
    renderer->dispatch(
            computeData, sgl::iceil(width, computeBlockSize),
            sgl::iceil(height, computeBlockSize), 1);
}


BackgroundAddPass::BackgroundAddPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void BackgroundAddPass::setBackgroundImage(const sgl::vk::ImageViewPtr& _backgroundImage) {
    backgroundImage = _backgroundImage;
    setDataDirty();
}

void BackgroundAddPass::setTargetImage(sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    setDataDirty();
}

void BackgroundAddPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert({ "USE_ENVIRONMENT_MAP_IMAGE", "" });

    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "BackgroundPass.Compute" }, preprocessorDefines);
}

void BackgroundAddPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    //computeData->setStaticImageView(cloudOnlyImage, "cloudOnlyImage");
    computeData->setStaticImageView(backgroundImage, "backgroundImage");
    computeData->setStaticImageView(outputImage, "resultImage");
}

void BackgroundAddPass::_render() {
    auto width = int(outputImage->getImage()->getImageSettings().width);
    auto height = int(outputImage->getImage()->getImageSettings().height);

    renderer->dispatch(
            computeData,
            sgl::iceil(width, BLOCK_SIZE), sgl::iceil(height, BLOCK_SIZE), 1);
}