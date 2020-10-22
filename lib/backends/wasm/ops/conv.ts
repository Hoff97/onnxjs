// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmConv extends Conv {
  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
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

    // determine number of threads needed to process
    // const numThreads = determineNumThreads(x.dims[0], this.group, w.dims[0], WasmBinding.workerNumber);

    const dataRank = x.dims.length - 2;  // Convolution dimension (1D, 2D, 3D...)

    const dilations = this.dilations.length > 0 ? this.dilations : new Array(dataRank).fill(1);
    const pads = this.pads.length > 0 ? this.pads : new Array(dataRank * 2).fill(0);
    const strides = this.strides.length > 0 ? this.strides : new Array(dataRank).fill(1);

    WasmBinding.getInstance().ccall(
        '_conv_f32',
        [x.floatData, 'float32ptr'],
        [x.dims, 'int32ptr'],
        [w.floatData, 'float32ptr'],
        [w.dims, 'int32ptr'],
        [y.floatData, 'float32ptr', 'out'],
        [y.dims, 'int32ptr'],
        [b ? b.floatData : null, 'float32ptr'],
        [dilations, 'int32ptr'],
        [this.group, 'int32'],
        [pads, 'int32ptr'],
        [strides, 'int32ptr'],
        [dataRank, 'int32'],
    );

    return [y];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
      return false;
    }

    if (inputs.length === 3 && inputs[2].type !== 'float32') {
      return false;
    }

    return true;
  }
}

// This function will determine the number of threads
// The strategy to parallelize is to parallelize on number of filter maps in the kernel
// (i.e.) number of output channels
/*function determineNumThreads(batchSize: number, group: number, numFilterMaps: number, numWebWorkers: number): number {
  // single threaded if:
  // 1) batch size is not 1 (data splitting logic across threads is specific to batch size being 1)
  // 2) multi-threading not supported yet for mulitple groups
  // 3) if number of filter maps is 1
  // 4) number of web workers is 0
  if (batchSize !== 1 || group !== 1 || numFilterMaps === 1 || numWebWorkers <= 0) {
    return 1;
  }

  // multi-threaded:
  // determine number of threads
  return Math.min(numFilterMaps, numWebWorkers + 1);
}*/
