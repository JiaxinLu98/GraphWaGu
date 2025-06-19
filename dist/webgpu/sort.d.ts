export declare class GPUSorter {
    zeroPipeline: GPUComputePipeline;
    histogramPipeline: GPUComputePipeline;
    prefixPipeline: GPUComputePipeline;
    scatterEvenPipeline: GPUComputePipeline;
    scatterOddPipeline: GPUComputePipeline;
    device: GPUDevice;
    bindGroupLayout: GPUBindGroupLayout;
    constructor(device: GPUDevice, subgroupSize: number);
    createKeyvalBuffers(length: number): [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer];
    createInternalMemBuffer(length: number): GPUBuffer;
    createSortBuffers(length: number): SortBuffers;
    recordCalculateHistogram(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup, length: number): void;
    recordPrefixHistogram(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup): void;
    recordScatterKeys(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup, length: number): void;
    sort(commandEncoder: GPUCommandEncoder, queue: GPUQueue, sortBuffers: SortBuffers, sortFirstN?: number): void;
    scatterBlocksRu(n: number): number;
    histoBlocksRu(n: number): number;
    keysBufferSize(n: number): number;
}
declare class SortBuffers {
    keysA: GPUBuffer;
    keysB: GPUBuffer;
    payloadA: GPUBuffer;
    payloadB: GPUBuffer;
    internalMemBuffer: GPUBuffer;
    uniformBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
    length: number;
    constructor(keysA: GPUBuffer, keysB: GPUBuffer, payloadA: GPUBuffer, payloadB: GPUBuffer, internalMemBuffer: GPUBuffer, uniformBuffer: GPUBuffer, bindGroup: GPUBindGroup, length: number);
    get keys(): GPUBuffer;
    get values(): GPUBuffer;
    keysValidSize(): number;
    destroy(): void;
}
export {};
