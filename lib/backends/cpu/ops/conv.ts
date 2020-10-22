// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuConv extends Conv {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];
    const w = inputs[1];
    const b = inputs.length === 3 ? inputs[2] : undefined;

    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      this.kernelShape = wDims.slice(2);
    }

    // create output Tensor after determining output size (after adjusting pads based on 'autoPad' attribute)
    const outputDims = PoolConvUtil.computeConvOutputShape(
        x.dims, w.dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    const y = new Tensor(outputDims, x.type);

    // Should almost be the same up until here for all backends

    conv(y, x, w, b, this.dilations, this.group, this.pads, this.strides);
    return [y];
  }
}

export function conv(
    Y: Tensor, X: Tensor, W: Tensor, B: Tensor|undefined, dilations: ReadonlyArray<number>, group: number,
    pads: ReadonlyArray<number>, strides: ReadonlyArray<number>) {
  const N = X.dims[0];        // Batch size
  const C = X.dims[1];        // Number of input channels
  const D = X.dims.slice(2);  // Data dimensions

  const M = W.dims[0];        // Output channels
  const K = W.dims.slice(2);  // Kernel dimensions

  const G = group;
  const CG = C / G;

  const kernelSize = ShapeUtil.size(K);

  const R = Y.dims.slice(2);  // Output data dimensions
  const outputSize = ShapeUtil.size(R);

  const dataRank = R.length;

  // Iterate over all batches
  for (let n = 0; n < N; n++) {
    // Iterate over all output channels
    for (let m = 0; m < M; m++) {
      if (B) {
        const bias = B ? B.get([m]) as number : 0;

        const outputIndices = new Array(R.length).fill(0);
        outputIndices.unshift(n, m);

        for (let oIx = 0; oIx < outputSize; oIx++) {
          Y.set(outputIndices, bias);

          ShapeUtil.incrementIndex(outputIndices, Y.dims);
        }
      }

      for (let cg = 0; cg < CG; cg++) {
        const c = m * CG + cg;

        const outputIndices = new Array(R.length).fill(0);
        outputIndices.unshift(n, m);
        for (let oIx = 0; oIx < outputSize; oIx++) {
          let result = Y.get(outputIndices) as number;

          const kernelIndices = new Array(K.length).fill(0);
          kernelIndices.unshift(m, cg);
          for (let kIx = 0; kIx < kernelSize; kIx++) {
            const inputIx = [n, c];

            let skip = false;
            for (let axis = 0; axis < dataRank; axis++) {
              const stride = strides.length === 0 ? 1 : strides[axis];
              const pad = pads.length === 0 ? 0 : pads[axis];
              const dilation = dilations.length === 0 ? 1 : dilations[axis];

              const ix = outputIndices[axis + 2] * stride - pad + kernelIndices[axis + 2] * dilation;

              if (ix < 0 || ix >= D[axis]) {
                skip = true;
                break;
              }

              inputIx.push(ix);
            }

            if (!skip) {
              const Wi = W.get(kernelIndices) as number;
              const Xi = X.get(inputIx) as number;
              result += Wi * Xi;
            }

            ShapeUtil.incrementIndex(kernelIndices, W.dims);
          }

          Y.set(outputIndices, result);

          ShapeUtil.incrementIndex(outputIndices, Y.dims);
        }
      }
    }
  }
}
