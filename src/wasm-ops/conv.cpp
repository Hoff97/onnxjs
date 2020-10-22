// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "conv.h"
#include "common.h"
#include "gemm.h"
#include "utils/shape_utils.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

// Wasm interop method
void conv_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  conv2D_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_FLOAT_PTR(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_INT32_PTR(data, dataIndex[8]),
      PARAM_INT32(data, dataIndex[9]), PARAM_INT32_PTR(data, dataIndex[10]),
      PARAM_INT32_PTR(data, dataIndex[11]), PARAM_INT32(data, dataIndex[12]));
}

// Core operator implementation
void conv2D_f32_imp(float *X, int *X_shape, float *W, int *W_shape, float *Y,
                    int *Y_shape, float *bias, int *dilations, int group,
                    int *pads, int *strides, int dimension) {
  const int N = X_shape[0]; // Batch size
  const int C = X_shape[1]; // Number of input channels

  const int M = W_shape[0]; // Output channels

  const int CG = C / group;

  std::vector<int32_t> inputSizes(X_shape,
                                  X_shape + dimension +
                                      2); // Input dimension sizes
  std::vector<int32_t> kernelSizes(W_shape,
                                   W_shape + dimension +
                                       2); // Kernel dimension sizes
  std::vector<int32_t> outputSizes(Y_shape,
                                   Y_shape + dimension +
                                       2); // Output dimension sizes

  const size_t kernelSize = ShapeUtils::size_from_dims(kernelSizes);
  const size_t inputSize = ShapeUtils::size_from_dims(inputSizes);
  const size_t outputSize = ShapeUtils::size_from_dims(outputSizes);

  const size_t outputDataSize = ShapeUtils::size_from_dims(outputSizes, 2);
  const size_t kernelDataSize = ShapeUtils::size_from_dims(kernelSizes, 2);

  std::vector<int32_t> outputStrides = ShapeUtils::compute_strides(outputSizes);
  std::vector<int32_t> inputStrides = ShapeUtils::compute_strides(inputSizes);
  std::vector<int32_t> kernelStrides = ShapeUtils::compute_strides(kernelSizes);

  std::vector<int32_t> outputIndices(dimension + 2, 0);
  size_t outputIndex = 0;

  std::vector<int32_t> inputIndices(dimension + 2, 0);
  size_t inputIndex = 0;

  std::vector<int32_t> kernelIndices(dimension + 2, 0);
  size_t kernelIndex = 0;

  // Iterate over all batches
  for (int32_t n = 0; n < N; n++) {
    outputIndices[0] = n;
    inputIndices[0] = n;
    // Iterate over all output channels
    for (int32_t m = 0; m < M; m++) {
      outputIndices[1] = m;
      kernelIndices[0] = m;
      if (bias != nullptr) {
        const float b = bias[m];

        size_t outIx = n * outputStrides[0] + m * outputStrides[1];
        const size_t end = outIx + outputDataSize;

        while (outIx < end) {
          Y[outIx] = b;
          outIx++;
        }
      }

      for (int32_t cg = 0; cg < CG; cg++) {
        const int32_t c = m * CG + cg;
        kernelIndices[1] = cg;
        inputIndices[1] = c;

        std::fill(outputIndices.begin() + 2, outputIndices.end(), 0);

        size_t outputBase = outputStrides[0] * n + outputStrides[1] * m;
        outputIndex = 0;
        while (outputIndex < outputDataSize) {

          std::fill(kernelIndices.begin() + 2, kernelIndices.end(), 0);
          kernelIndex = 0;
          size_t kernelBase = kernelStrides[0] * m + kernelStrides[1] * cg;
          while (kernelIndex < kernelDataSize) {
            bool skip = false;

            inputIndex = n * inputStrides[0] + c * inputStrides[1];
            for (size_t axis = 0; axis < dimension; axis++) {
              const int32_t ix = outputIndices[axis + 2] * strides[axis] -
                                 pads[axis] +
                                 kernelIndices[axis + 2] * dilations[axis];

              if (ix < 0 || ix >= inputSizes[axis + 2]) {
                skip = true;
                break;
              }

              inputIndices[axis + 2] = ix;
              inputIndex += inputStrides[axis + 2] * ix;
            }

            if (!skip) {
              Y[outputBase + outputIndex] +=
                  W[kernelBase + kernelIndex] * X[inputIndex];
            }

            ShapeUtils::increment_index(kernelIndices, kernelSizes,
                                        dimension + 2);
            kernelIndex++;
          }

          outputIndex++;
          ShapeUtils::increment_index(outputIndices, outputSizes,
                                      dimension + 2);
        }
      }
    }
  }
}
