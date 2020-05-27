/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// TODO(annxingyuan): read version from tfjsWasm directly once
// https://github.com/tensorflow/tfjs/pull/2819 is merged.
import { version } from '@tensorflow/tfjs-backend-wasm/dist/version';

import { TRIANGULATION } from './triangulation';

tfjsWasm.setWasmPath(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
  version}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas,
  scatterGLHasInitialized = false, scatterGL;

const VIDEO_SIZE = 500;
const mobile = isMobile();
const stats = new Stats();
const state = {
  backend: 'wasm',
  maxFaces: 1,
  triangulateMesh: false//true
};

function setupDatGui() {
}

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function renderPrediction() {
  stats.begin();
  ctx.fillStyle = '#00FF00';
  ctx.strokeStyle = '#FF0000';

  const predictions = await model.estimateFaces(video);
  ctx.drawImage(
    video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);

  if (predictions.length > 0) {
    predictions.forEach(prediction => {
      const keypoints = prediction.scaledMesh;

      if (state.triangulateMesh) {
        for (let i = 0; i < TRIANGULATION.length / 3; i++) {
          const points = [
            TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2]
          ].map(index => keypoints[index]);

          drawPath(ctx, points, true);
        }
      } else {
        // for (let i = 0; i < keypoints.length; i++) {
        //   const x = keypoints[i][0];
        //   const y = keypoints[i][1];

        //   ctx.beginPath();
        //   ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
        //   ctx.fill();
        // }
        const boundingBox = prediction.boundingBox;
        console.log(boundingBox);
        ctx.fillStyle = '#FF0000';
        ctx.strokeStyle = '#FF0000';
        // console.log("boundingBox topLeft x=" + boundingBox.topLeft[0][0] + "; y=" + boundingBox.topLeft[0][1]);
        // ctx.arc(boundingBox.topLeft[0][0], boundingBox.topLeft[0][1], 1 /* radius */, 0, 2 * Math.PI);
        // ctx.fill();
        // console.log("boundingBox bottomRight x=" + boundingBox.bottomRight[0][0] + "; y=" + boundingBox.bottomRight[0][1]);
        // ctx.arc(boundingBox.bottomRight[0][0], boundingBox.bottomRight[0][1], 1 /* radius */, 0, 2 * Math.PI);
        // ctx.fill();

        //boundingBox
        ctx.fillStyle = '#0000FF';
        ctx.strokeStyle = '#0000FF';
        ctx.strokeRect(boundingBox.bottomRight[0][0], boundingBox.topLeft[0][1], boundingBox.topLeft[0][0] - boundingBox.bottomRight[0][0], boundingBox.bottomRight[0][1] - boundingBox.topLeft[0][1]);

        //silhouette
        ctx.fillStyle = '#FF0000';
        ctx.strokeStyle = '#FF0000';
        const silhouette = prediction.annotations.silhouette;
        for (let i = 0; i < silhouette.length; i++) {
          const x1 = silhouette[i][0];
          const y1 = silhouette[i][1];

          ctx.beginPath();
          ctx.arc(x1, y1, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }

        //Draw Lines on Silhouette
        // ctx.fillStyle = '#800080';
        // ctx.beginPath();
        // let xx = 0;
        // let yy = 0;
        // if(silhouette.length > 0)
        // {
        //   xx = silhouette[0][0];
        //   yy = silhouette[0][1];
        // }
        // ctx.stroke();
        // ctx.moveTo(xx, yy);   // Begin first sub-path
        // ctx.stroke();
        // for (let i = 0; i < silhouette.length; i++) {
        //   const x1 = silhouette[i][0];
        //   const y1 = silhouette[i][1];
        //   ctx.lineTo(x1, y1);
        //   ctx.stroke();
        // }
        // ctx.lineTo(xx, yy);
        // ctx.stroke();
        // ctx.fill();

        //lips
        ctx.fillStyle = '#FFFF00';
        ctx.strokeStyle = '#FFFF00';
        const lipsUpperOuter = prediction.annotations.lipsUpperOuter;
        for (let i = 0; i < lipsUpperOuter.length; i++) {
          const x2 = lipsUpperOuter[i][0];
          const y2 = lipsUpperOuter[i][1];

          ctx.beginPath();
          ctx.arc(x2, y2, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        const lipsLowerOuter = prediction.annotations.lipsLowerOuter;
        for (let i = 0; i < lipsLowerOuter.length; i++) {
          const x3 = lipsLowerOuter[i][0];
          const y3 = lipsLowerOuter[i][1];

          ctx.beginPath();
          ctx.arc(x3, y3, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.fillStyle = '#FFFF00';
        ctx.strokeStyle = '#FFFF00';
        const lipsUpperInner = prediction.annotations.lipsUpperInner;
        for (let i = 0; i < lipsUpperOuter.length; i++) {
          const x4 = lipsUpperInner[i][0];
          const y4 = lipsUpperInner[i][1];

          ctx.beginPath();
          ctx.arc(x4, y4, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        const lipsLowerInner = prediction.annotations.lipsLowerInner;
        for (let i = 0; i < lipsLowerInner.length; i++) {
          const x5 = lipsLowerInner[i][0];
          const y5 = lipsLowerInner[i][1];

          ctx.beginPath();
          ctx.arc(x5, y5, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }

        //Fill LipStick Color
        //ctx.fillStyle = '#FF4573';
        //ctx.strokeStyle = '#FF4573';
        //  ctx.fillStyle = '#FF2CB4';
        //  ctx.strokeStyle = '#FF2CB4';
         ctx.fillStyle = '#FF0000';
         ctx.strokeStyle = '#FF0000';
        drawLipStickColor(ctx, lipsUpperOuter, lipsLowerOuter, lipsUpperInner, lipsLowerInner);


        //Eyebrows
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#FFFFFF';
        const rightEyebrowUpper = prediction.annotations.rightEyebrowUpper;
        for (let i = 0; i < rightEyebrowUpper.length; i++) {
          const x6 = rightEyebrowUpper[i][0];
          const y6 = rightEyebrowUpper[i][1];

          ctx.beginPath();
          ctx.arc(x6, y6, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        const rightEyebrowLower = prediction.annotations.rightEyebrowLower;
        for (let i = 0; i < rightEyebrowLower.length; i++) {
          const x7 = rightEyebrowLower[i][0];
          const y7 = rightEyebrowLower[i][1];

          ctx.beginPath();
          ctx.arc(x7, y7, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        const leftEyebrowUpper = prediction.annotations.leftEyebrowUpper;
        for (let i = 0; i < leftEyebrowUpper.length; i++) {
          const x8 = leftEyebrowUpper[i][0];
          const y8 = leftEyebrowUpper[i][1];

          ctx.beginPath();
          ctx.arc(x8, y8, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
        const leftEyebrowLower = prediction.annotations.leftEyebrowLower;
        for (let i = 0; i < leftEyebrowLower.length; i++) {
          const x9 = leftEyebrowLower[i][0];
          const y9 = leftEyebrowLower[i][1];

          ctx.beginPath();
          ctx.arc(x9, y9, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }

        //Eyes
        ctx.fillStyle = '#00FF00';
        ctx.strokeStyle = '#00FF00';
        const rightEyeUpper0 = prediction.annotations.rightEyeUpper0;
        drawPoints(ctx, rightEyeUpper0);
        const rightEyeLower0 = prediction.annotations.rightEyeLower0;
        drawPoints(ctx, rightEyeLower0);
        // const rightEyeUpper1 = prediction.annotations.rightEyeUpper1;
        // drawPoints(ctx, rightEyeUpper1);
        // const rightEyeLower1 = prediction.annotations.rightEyeLower1;
        // drawPoints(ctx, rightEyeLower1);
        // const rightEyeUpper2 = prediction.annotations.rightEyeUpper2;
        // drawPoints(ctx, rightEyeUpper2);
        // const rightEyeLower2 = prediction.annotations.rightEyeLower2;
        // drawPoints(ctx, rightEyeLower2);
        // const rightEyeLower3 = prediction.annotations.rightEyeLower3;
        // drawPoints(ctx, rightEyeLower3);

        const leftEyeUpper0 = prediction.annotations.leftEyeUpper0;
        drawPoints(ctx, leftEyeUpper0);
        const leftEyeLower0 = prediction.annotations.leftEyeLower0;
        drawPoints(ctx, leftEyeLower0);
        // const leftEyeUpper1 = prediction.annotations.leftEyeUpper1;
        // drawPoints(ctx, leftEyeUpper1);
        // const leftEyeLower1 = prediction.annotations.leftEyeLower1;
        // drawPoints(ctx, leftEyeLower1);
        // const leftEyeUpper2 = prediction.annotations.leftEyeUpper2;
        // drawPoints(ctx, leftEyeUpper2);
        // const leftEyeLower2 = prediction.annotations.leftEyeLower2;
        // drawPoints(ctx, leftEyeLower2);
        // const leftEyeLower3 = prediction.annotations.leftEyeLower3;
        // drawPoints(ctx, leftEyeLower3);

        //Midway Between Eyes
        ctx.fillStyle = '#00FF00';
        ctx.strokeStyle = '#00FF00';
        const midwayBetweenEyes = prediction.annotations.midwayBetweenEyes;
        drawPoints(ctx, midwayBetweenEyes);

        //Nose
        ctx.fillStyle = '#FFC0CB';
        ctx.strokeStyle = '#FFC0CB';
        const noseTip = prediction.annotations.noseTip;
        drawPoints(ctx, noseTip);
        const noseBottom = prediction.annotations.noseBottom;
        drawPoints(ctx, noseBottom);
        const noseRightCorner = prediction.annotations.noseRightCorner;
        drawPoints(ctx, noseRightCorner);
        const noseLeftCorner = prediction.annotations.noseLeftCorner;
        drawPoints(ctx, noseLeftCorner);

        //Cheek
        ctx.fillStyle = '#00FF00';
        ctx.strokeStyle = '#00FF00';
        const rightCheek = prediction.annotations.rightCheek;
        drawPoints(ctx, rightCheek);
        const leftCheek = prediction.annotations.leftCheek;
        drawPoints(ctx, leftCheek);

      }
    });

  }

  stats.end();
  requestAnimationFrame(renderPrediction);
};

async function drawPoints(ctx, attribute) {
  for (let i = 0; i < attribute.length; i++) {
    const x = attribute[i][0];
    const y = attribute[i][1];

    ctx.beginPath();
    ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
    ctx.fill();
  }
}

async function drawLipStickColor(ctx, lipsUpperOuter, lipsLowerOuter, lipsUpperInner, lipsLowerInner)
{
  ctx.beginPath();
  drawLinesToPart(ctx, lipsUpperOuter);
  drawLinesToPart(ctx, lipsUpperInner);
  ctx.stroke();
  // if(lipsLowerOuter.length > 0 && lipsLowerInner.length >0)
  // {
  //   let lipsLowerOuterFirstX = lipsLowerOuter[0][0];
  //   let lipsLowerOuterFirstY = lipsLowerOuter[0][1];
  //   let lipsLowerInnerFirstX = lipsLowerInner[0][0];
  //   let lipsLowerInnerFirstY = lipsLowerInner[0][1];
  //   ctx.moveTo(lipsLowerOuterFirstX, lipsLowerOuterFirstY);
  //   ctx.lineTo(lipsLowerInnerFirstX, lipsLowerInnerFirstY);
  //   ctx.stroke();
  // }
  // if(lipsLowerOuter.length > 0 && lipsLowerInner.length >0)
  // {
  //   let lipsLowerOuterLastX = lipsLowerOuter[lipsLowerOuter.length - 1][0];
  //   let lipsLowerOuterLastY = lipsLowerOuter[lipsLowerOuter.length - 1][1];
  //   let lipsLowerInnerLastX = lipsLowerInner[lipsLowerInner.length - 1][0];
  //   let lipsLowerInnerLastY = lipsLowerInner[lipsLowerInner.length - 1][1];
  //   ctx.moveTo(lipsLowerOuterLastX, lipsLowerOuterLastY);
  //   ctx.lineTo(lipsLowerInnerLastX, lipsLowerInnerLastY);
  //   ctx.stroke();
  // }

  // if(lipsUpperOuter.length > 0 && lipsUpperInner.length >0)
  // {
  //   let lipsUpperOuterFirstX = lipsUpperOuter[0][0];
  //   let lipsUpperOuterFirstY = lipsUpperOuter[0][1];
  //   let lipsUpperInnerFirstX = lipsUpperInner[0][0];
  //   let lipsUpperInnerFirstY = lipsUpperInner[0][1];
  //   ctx.moveTo(lipsUpperOuterFirstX, lipsUpperOuterFirstY);
  //   ctx.lineTo(lipsUpperInnerFirstX, lipsUpperInnerFirstY);
  //   ctx.stroke();
  // }
  // if(lipsUpperOuter.length > 0 && lipsUpperInner.length >0)
  // {
  //   let lipsUpperOuterLastX = lipsUpperOuter[lipsUpperOuter.length - 1][0];
  //   let lipsUpperOuterLastY = lipsUpperOuter[lipsUpperOuter.length - 1][1];
  //   let lipsUpperInnerLastX = lipsUpperInner[lipsUpperInner.length - 1][0];
  //   let lipsUpperInnerLastY = lipsUpperInner[lipsUpperInner.length - 1][1];
  //   ctx.moveTo(lipsUpperOuterLastX, lipsUpperOuterLastY);
  //   ctx.lineTo(lipsUpperInnerLastX, lipsUpperInnerLastY);
  //   ctx.stroke();
  // }
  drawLinesToPart(ctx, lipsLowerOuter);
  drawLinesToPart(ctx, lipsLowerInner);
  ctx.fill();
}

async function drawLinesToPart(ctx, part)
{
  let xx = 0;
  let yy = 0;
  if(part.length > 0)
  {
    xx = part[0][0];
    yy = part[0][1];
  }
  ctx.stroke();
  ctx.moveTo(xx, yy);   // Begin first sub-path

  for (let i = 0; i < part.length; i++) {
    const xx1 = part[i][0];
    const yy1 = part[i][1];
    ctx.lineTo(xx1, yy1);
    ctx.stroke();
  }
}

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();

  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  //  ctx.fillStyle = '#32EEDB';
  ctx.fillStyle = '#00FF00';
  // ctx.strokeStyle = '#32EEDB';
  ctx.strokeStyle = '#FF0000';
  ctx.lineWidth = 0.5;

  model = await facemesh.load({ maxFaces: state.maxFaces });
  renderPrediction();

};

main();
