// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData} from '../types';

import {ShapeUtil} from './../../../util';

const samplerNames = 'XWB';

export class WebGLConv extends Conv {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const x = inputs[0];
    const w = inputs[1];

    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      this.kernelShape = wDims.slice(2);
    }

    // create output Tensor after determining output size (after adjusting pads based on 'autoPad' attribute)
    const outputShape = PoolConvUtil.computeConvOutputShape(
        x.dims, w.dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);

    const shaderSource = this.getShaderSource(x.dims, w.dims, outputShape, inputs.length === 3);
    const inputLayouts = inputs.map(t => inferenceHandler.getOrCreateTextureLayout(t));
    return {
      inputLayouts,
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: samplerNames.slice(0, inputs.length).split(''),
      variables: [],
      shaderSource,
    };
  }

  getShaderSource(shapeX: readonly number[], shapeW: readonly number[], shapeY: readonly number[], bias: boolean) {
    const rank = shapeX.length;
    /*const dimension = rank - 2;

    const N = shapeX[0];
    const C = shapeX[1];
    const M = shapeW[0];*/
    const CG = shapeW[1];

    let initValue = 'float value = 0.0;';
    if (bias) {
      // Initialize value with bias
      initValue = `
        int biasIndex[1];
        biasIndex[0] = indices[1];
        float value = _B(biasIndex);
        `;
    }

    const kernelDataSize = ShapeUtil.size(shapeW.slice(2));

    const shaderSource = `
      float process(int indices[${rank}]) {
        ${initValue}

        int n = indices[0];
        int m = indices[1];

        ${this.initInputIndices(rank)}
        ${this.initKernelIndices(rank)}

        for (int cg = 0; cg < ${CG}; cg++) {
          int c = m * ${CG} + cg;

          inputIndices[1] = c;
          kernelIndices[1] = cg;

          int kernelIx = 0;
          while (kernelIx < ${kernelDataSize}) {
            bool skip = false;

            ${this.updateInputIndexScript(rank, shapeX)}

            if (!skip) {
              value += _X(inputIndices) * _W(kernelIndices);
            }

            ${this.incrementKernelIndexScript(rank, shapeW)}
            kernelIx++;
          }
        }

        return value;
      }`;

    return shaderSource;
  }

  updateInputIndexScript(rank: number, shapeX: readonly number[]) {
    let script = '';

    for (let axis = 0; axis < rank - 2; axis++) {
      const stride = this.strides.length === 0 ? 1 : this.strides[axis];
      const pad = this.pads.length === 0 ? 0 : this.pads[axis];
      const dilation = this.dilations.length === 0 ? 1 : this.dilations[axis];

      const dim = axis + 2;

      script += `inputIndices[${dim}] = indices[${dim}]*${stride} - ${pad} + kernelIndices[${dim}]*${dilation};
      if (inputIndices[${dim}] < 0 || inputIndices[${dim}] >= ${shapeX[dim]}) {
        skip = true;
      }`;
    }
    return script;
  }

  incrementKernelIndexScript(rank: number, shapeW: readonly number[]): string {
    let script = '';
    for (let i = rank - 1; i >= 2; i--) {
      script += `
      kernelIndices[${i}] += 1;
        if (kernelIndices[${i}] >= ${shapeW[i]}) {
          kernelIndices[${i}] = 0;
      `;
    }
    for (let i = rank - 1; i >= 2; i--) {
      script += '}\n';
    }

    return script;
  }

  initInputIndices(rank: number) {
    let result = `int inputIndices[${rank}];
    inputIndices[0] = n;`;
    for (let i = 1; i < rank; i++) {
      result += `\ninputIndices[${i}] = 0;`;
    }
    return result;
  }

  initKernelIndices(rank: number) {
    let result = `int kernelIndices[${rank}];
    kernelIndices[0] = m;`;
    for (let i = 1; i < rank; i++) {
      result += `\nkernelIndices[${i}] = 0;`;
    }
    return result;
  }

  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((v, i) => inferenceHandler.getOrCreateTextureData(v, programInfo.inputLayouts[i]));

    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}
