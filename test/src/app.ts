import { Renderer } from 'graphwagu-renderer';
import {loadGraphFromPath} from './utils';

let context: GPUCanvasContext;
let format: GPUTextureFormat;

async function main() {
  const canvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement;
  const label = document.getElementById('label') as HTMLLabelElement;
  const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
  const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;

  if (!navigator.gpu) {
    document.getElementById("no-webgpu")!.style.display = "block";
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize
    }}
  );
  if (!device) {
    document.getElementById("no-webgpu")!.style.display = "block";
    return;
  }

  context = canvas.getContext("webgpu")!;
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque'
  });

  const renderer = new Renderer(device, canvas, label);

  const { nodes, edges, sources, targets } = await loadGraphFromPath('/network.dot');

  console.log(`sources: ${sources}`);
  console.log(`targets: ${targets}`);


  renderer.setNodeEdgeData(nodes, edges, sources, targets);
  renderer.setCoolingFactor(0.985);
  renderer.setIdealLength(0.005);
  renderer.setTheta(2);
  renderer.setIterationCount(100000);
  renderer.setEnergy(0.1);
  
  // await renderer.runForceDirected();
  startBtn.addEventListener("click", async () => {
    renderer.forceDirected!.stopForce = false;
    await renderer.runForceDirected();
  });

  stopBtn.addEventListener("click", () => {
    renderer.forceDirected!.stopForces();
  });
}

window.addEventListener('load', main);
