import { GPUSorter } from './sort';
export declare class ForceDirected {
    sorter: GPUSorter;
    paramsBuffer: GPUBuffer;
    nodeDataBuffer: GPUBuffer;
    edgeDataBuffer: GPUBuffer;
    forceDataBuffer: GPUBuffer;
    coolingFactor: number;
    device: GPUDevice;
    createTreePipeline: GPUComputePipeline;
    createSourceListPipeline: GPUComputePipeline;
    createTargetListPipeline: GPUComputePipeline;
    computeAttractivePipeline: GPUComputePipeline;
    computeForcesBHPipeline: GPUComputePipeline;
    applyForcesPipeline: GPUComputePipeline;
    iterationCount: number;
    mortonCodePipeline: GPUComputePipeline;
    mortonCodeBuffer: GPUBuffer;
    theta: number;
    l: number;
    stopForce: boolean;
    clusterSize: number;
    nodeLength: number;
    edgeLength: number;
    sourceEdgeDataBuffer: GPUBuffer;
    targetEdgeDataBuffer: GPUBuffer;
    constructor(device: GPUDevice);
    stopForces(): void;
    formatToD3Format(positionList: number[], edgeList: number[], nLength: number, eLength: number): {
        nodeArray: any[];
        edgeArray: any[];
    };
    setNodeEdgeData(nodes: number[], edges: number[]): void;
    runForces(coolingFactor?: number, l?: number, theta?: number, iterationCount?: number): Promise<void>;
}
