import { GPUSorter } from './sort';
export declare function uploadToBuffer(encoder: GPUCommandEncoder, device: GPUDevice, buffer: GPUBuffer, values: Uint32Array): void;
export declare function downloadBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<Uint32Array>;
export declare function testSort(sorter: GPUSorter, device: GPUDevice): Promise<boolean>;
export declare function guessWorkgroupSize(device: GPUDevice): Promise<number | undefined>;
