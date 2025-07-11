import {apply_forces} from './wgsl-shaders';
import {create_adjacency_matrix} from './wgsl-shaders';
import {create_targetlist} from './wgsl-shaders';
import {create_sourcelist} from './wgsl-shaders';
import {compute_attract_forces} from './wgsl-shaders';
import {compute_forces} from './wgsl-shaders';
import {compute_forcesBH} from './wgsl-shaders';
import {compute_attractive_new} from './wgsl-shaders';
import {morton_codes} from './wgsl-shaders';
import {create_tree} from './wgsl-shaders';
import { GPUSorter } from './sort';

export class ForceDirected {
    public sorter: GPUSorter;
    public paramsBuffer: GPUBuffer;
    public nodeDataBuffer: GPUBuffer;
    public edgeDataBuffer: GPUBuffer;
    public adjMatrixBuffer: GPUBuffer;
    public laplacianBuffer: GPUBuffer;
    public quadTreeBuffer: GPUBuffer;
    public forceDataBuffer: GPUBuffer;
    public coolingFactor: number = 0.985;
    public device: GPUDevice;
    public createMatrixPipeline: GPUComputePipeline;
    public createTreePipeline: GPUComputePipeline;
    public createSourceListPipeline: GPUComputePipeline;
    public createTargetListPipeline: GPUComputePipeline;
    public computeAttractiveNewPipeline: GPUComputePipeline;
    public computeForcesPipeline: GPUComputePipeline;
    public computeForcesBHPipeline: GPUComputePipeline;
    public computeAttractForcesPipeline: GPUComputePipeline;
    public applyForcesPipeline: GPUComputePipeline;
    public iterationCount: number = 10000;
    public threshold: number = 100;
    public force: number = 1000.0;
    public mortonCodePipeline: GPUComputePipeline;
    public mortonCodeBuffer: GPUBuffer;
    public energy: number = 0.1;
    public theta: number = 0.8;
    public stopForce: boolean = false;
    clusterSize: number;

    constructor(device: GPUDevice) {
        this.device = device;
        this.sorter = new GPUSorter(this.device, 32);
        this.clusterSize = 4;

        this.nodeDataBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.mortonCodeBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.edgeDataBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.adjMatrixBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.laplacianBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.quadTreeBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.forceDataBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.createMatrixPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: create_adjacency_matrix
                }),
                entryPoint: "main",
            },
        });

        this.createTreePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: create_tree.replace(/CHANGEME/g, this.clusterSize.toString())
                }),
                entryPoint: "main",
            },
        });

        this.mortonCodePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: morton_codes.replace(/CHANGEME/g, this.clusterSize.toString())
                }),
                entryPoint: "main",
            }
        });

        this.createSourceListPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: create_sourcelist
                }),
                entryPoint: "main",
            },
        });

        this.createTargetListPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: create_targetlist
                }),
                entryPoint: "main",
            },
        });

        this.computeAttractiveNewPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: compute_attractive_new
                }),
                entryPoint: "main",
            },
        });

        this.computeForcesPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: compute_forces,
                }),
                entryPoint: "main",
            },
        });

        this.computeForcesBHPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: compute_forcesBH.replace(/CHANGEME/g, this.clusterSize.toString()),
                }),
                entryPoint: "main",
            },
        });

        this.computeAttractForcesPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: compute_attract_forces,
                }),
                entryPoint: "main",
            },
        });

        this.applyForcesPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: apply_forces,
                }),
                entryPoint: "main",
            },
        });

        // Create a buffer to store the params, output, and min/max
        this.paramsBuffer = device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    formatToD3Format(positionList: number[], edgeList: number[], nLength: number, eLength: number) {
        const nodeArray = new Array(nLength);
        const edgeArray = new Array(eLength / 2);

        for (let i = 0; i < 4 * nLength; i = i + 4) {
            nodeArray[i / 4] = {
                index: i / 4,
                name: (i / 4).toString(),
                x: positionList[i + 1],
                y: positionList[i + 2]
            };
        }

        for (let i = 0; i < eLength; i = i + 2) {
            let sourceIndex = edgeList[i];
            let targetIndex = edgeList[i + 1];

            edgeArray[i / 2] = {};

            edgeArray[i / 2].index = i / 2;
            edgeArray[i / 2].source = {};
            edgeArray[i / 2].source.index = sourceIndex;
            edgeArray[i / 2].source.name = sourceIndex.toString();
            edgeArray[i / 2].source.x = nodeArray[sourceIndex].x;
            edgeArray[i / 2].source.y = nodeArray[sourceIndex].y;

            edgeArray[i / 2].target = {};
            edgeArray[i / 2].target.index = targetIndex;
            edgeArray[i / 2].target.name = targetIndex.toString();
            edgeArray[i / 2].target.x = nodeArray[targetIndex].x;
            edgeArray[i / 2].target.y = nodeArray[targetIndex].y;
        }

        return { nodeArray, edgeArray }
    }

    stopForces() {
        this.stopForce = true;
    }

    async runForces(
        nodeDataBuffer = this.nodeDataBuffer,
        edgeDataBuffer = this.edgeDataBuffer,
        mortonCodeBuffer = this.mortonCodeBuffer,
        nodeLength: number = 0, edgeLength: number = 0,
        coolingFactor = this.coolingFactor, l = 0.01,
        energy: number = 0.1, theta: number = 0.8,
        iterationCount = this.iterationCount,
        threshold = this.threshold,
        // label: HTMLLabelElement,
        sourceEdgeBuffer: GPUBuffer | null,
        targetEdgeBuffer: GPUBuffer | null,
        frame: FrameRequestCallback,
    ) {
        this.stopForce = false;
        // coolingFactor = 0.995;
        // l = 0.01;
        if (nodeLength === 0 || edgeLength === 0 || nodeDataBuffer === null || edgeDataBuffer === null) {
            alert("No data to run");
            return;
        }
        this.energy = energy;
        this.theta = theta;
        this.coolingFactor = coolingFactor;
        this.nodeDataBuffer = nodeDataBuffer;
        this.mortonCodeBuffer = mortonCodeBuffer;
        this.edgeDataBuffer = edgeDataBuffer;
        this.threshold = threshold;
        this.force = 100000;
        const rangeBuffer = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const bounding = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });

        let mapping = bounding.getMappedRange();
        new Int32Array(mapping).set([0, 1000, 0, 1000]);
        bounding.unmap();
        const bounding2 = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });

        let mapping2 = bounding2.getMappedRange();
        new Int32Array(mapping2).set([1000, -1000, 1000, -1000]);
        bounding2.unmap();
        // this.coolingFactor = 2.0;
        let commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(bounding, 0, rangeBuffer, 0, 4 * 4);
        this.device.queue.submit([commandEncoder.finish()]);
    
        const sortBuffers = this.sorter.createSortBuffers(nodeLength);

        // Set up params (node length, edge length) for creating adjacency matrix
        const uploadBuffer = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        mapping = uploadBuffer.getMappedRange();
        new Uint32Array(mapping).set([nodeLength, edgeLength]);
        new Float32Array(mapping).set([this.coolingFactor, l], 2);
        uploadBuffer.unmap();

        commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(uploadBuffer, 0, this.paramsBuffer, 0, 4 * 4);

        this.device.queue.submit([commandEncoder.finish()]);

        // Create a buffer to store the params, output, and min/max
        const treeInfoBuffer = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(
            treeInfoBuffer,
            8,
            new Float32Array([this.theta]),
            0,
            1
        );

        this.forceDataBuffer = this.device.createBuffer({
            size: nodeLength * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const quadTreeLength = nodeLength * 12 * 4 * 4;
        this.quadTreeBuffer = this.device.createBuffer({
            size: quadTreeLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const sourceListBuffer = this.device.createBuffer({
            size: edgeLength * 2,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const targetListBuffer = this.device.createBuffer({
            size: edgeLength * 2,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const edgeInfoBuffer = this.device.createBuffer({
            size: nodeLength * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const treeBuffer = this.device.createBuffer({
            size: Math.ceil(nodeLength * 2.1) * (12 + Math.max(4, this.clusterSize)) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const createSourceListBindGroup = this.device.createBindGroup({
            layout: this.createSourceListPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: sourceEdgeBuffer!,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: edgeInfoBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: sourceListBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.paramsBuffer
                    }
                }
            ]
        });

        const createTargetListBindGroup = this.device.createBindGroup({
            layout: this.createTargetListPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: targetEdgeBuffer!,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: edgeInfoBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: targetListBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.paramsBuffer
                    }
                }
            ]
        })
        this.device.queue.submit([commandEncoder.finish()]);
        commandEncoder = this.device.createCommandEncoder();
        // Run create source and target lists pass
        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setBindGroup(0, createSourceListBindGroup);
        computePassEncoder.setPipeline(this.createSourceListPipeline);
        computePassEncoder.dispatchWorkgroups(1, 1, 1);
        computePassEncoder.setBindGroup(0, createTargetListBindGroup);
        computePassEncoder.setPipeline(this.createTargetListPipeline);
        computePassEncoder.dispatchWorkgroups(1, 1, 1);
        computePassEncoder.end();
        const gpuReadBuffer = this.device.createBuffer({
            size: nodeLength * 4 * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        commandEncoder.copyBufferToBuffer(
            edgeInfoBuffer /* source buffer */,
            0 /* source offset */,
            gpuReadBuffer /* destination buffer */,
            0 /* destination offset */,
            nodeLength * 4 * 4 /* size */
        );
        this.device.queue.submit([commandEncoder.finish()]);
        // await this.device.queue.onSubmittedWorkDone();
        // await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        // const arrayBuffer = gpuReadBuffer.getMappedRange();
        // const list = new Uint32Array(arrayBuffer);
        // console.log(list);
        // return;

        // const iterationTimes: Array<number> = [];
        const applyBindGroup = this.device.createBindGroup({
            layout: this.applyForcesPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.nodeDataBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.forceDataBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.paramsBuffer
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: rangeBuffer
                    }
                }
            ],
        });

        const createTreeBindGroup = this.device.createBindGroup({
            layout: this.createTreePipeline.getBindGroupLayout(0),
            entries: [
                // Sort values buffer filled with indices
                {
                    binding: 0,
                    resource: {
                        buffer: sortBuffers.values,
                    }
                },  
                {
                    binding: 1,
                    resource: {
                        buffer: this.paramsBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: treeInfoBuffer,
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: rangeBuffer,
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: treeBuffer,
                    }
                },
            ]
        });
        const mortonCodeBindGroup = this.device.createBindGroup({
            layout: this.mortonCodePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.nodeDataBuffer,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.mortonCodeBuffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.paramsBuffer,
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: rangeBuffer,
                    }
                },
                // Sort values buffer filled with mor
                {
                    binding: 4,
                    resource: {
                        buffer: sortBuffers.values,
                    }
                },
                {
                    binding: 5,
                    resource: {
                        buffer: treeBuffer,
                    }
                },
            ]
        });
        // const batchBuffer = this.device.createBuffer({
        //     size: 4,
        //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
        // });
        let positionReadBuffer = this.device.createBuffer({
            size: nodeLength * 4 * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        let numIterations = 0;
        var totalTime = 0;
        var totalTree = 0;
        // const querySet = this.device.createQuerySet({
        //     type: "timestamp",
        //     count: 10,
        // });
        // const queryBuffer = this.device.createBuffer({
        //     size: 8,
        //     usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        // });
        // const readQueryBuffer = this.device.createBuffer({
        //     size: 8,
        //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        // });
        // let totalSum = 0;
        // let treeSum = 0;
        var total_sumsize = 0;
        var start, end;
        const debug = false;
        var totalStart = 0;
        while (numIterations < iterationCount && this.coolingFactor > 0.0001 && this.force >= 0) {
            if (numIterations == 1) {
                totalStart = performance.now();
            }
            const frameStart = performance.now();
            numIterations++;
            // Set up params (node length, edge length)
            const upload = this.device.createBuffer({
                size: 4 * 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true,
            });
            const mapping = upload.getMappedRange();
            new Uint32Array(mapping).set([nodeLength, edgeLength]);
            new Float32Array(mapping).set([this.coolingFactor, l], 2);
            upload.unmap();
            //this.device.createQuerySet({})
            let commandEncoder = this.device.createCommandEncoder();
            //commandEncoder.writeTimestamp();
            commandEncoder.copyBufferToBuffer(upload, 0, this.paramsBuffer, 0, 4 * 4);
            // commandEncoder.copyBufferToBuffer(clearBuffer, 0, this.quadTreeBuffer, 0, quadTreeLength);
            this.device.queue.submit([commandEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            start = performance.now();
            commandEncoder = this.device.createCommandEncoder();
            let computePassEncoder = commandEncoder.beginComputePass();
            computePassEncoder.setBindGroup(0, mortonCodeBindGroup);
            computePassEncoder.setPipeline(this.mortonCodePipeline);
            computePassEncoder.dispatchWorkgroups(Math.ceil(nodeLength / 128), 1, 1);
            computePassEncoder.end();
            commandEncoder.copyBufferToBuffer(this.mortonCodeBuffer, 0, sortBuffers.keys, 0, this.mortonCodeBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            end = performance.now();
            console.log(`Morton codes took ${end - start}ms`)
            // {
                // var dbgBuffer = this.device.createBuffer({
                //     size: mortonCodeBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(mortonCodeBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBuffer.mapAsync(GPUMapMode.READ);

                // var debugValsf = new Float32Array(dbgBuffer.getMappedRange());
                // console.log(debugValsf);

                // var dbgBufferu = this.device.createBuffer({
                //     size: mortonCodeBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(mortonCodeBuffer, 0, dbgBufferu, 0, dbgBufferu.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBufferu.mapAsync(GPUMapMode.READ);

                // var debugValsu = new Uint32Array(dbgBufferu.getMappedRange());
                // console.log(debugValsu);
            // }
            if (iterationCount == 2) {
                return;
            }

            start = performance.now();
            const sortEncoder = this.device.createCommandEncoder();
            this.sorter.sort(sortEncoder, this.device.queue, sortBuffers);
            this.device.queue.submit([sortEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            end = performance.now();
            console.log(`Sort took ${end - start} ms`);
            // {
            //     var dbgBuffer = this.device.createBuffer({
            //         size: sortBuffers.keys.size,
            //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            //     });

            //     commandEncoder = this.device.createCommandEncoder();
            //     commandEncoder.copyBufferToBuffer(sortBuffers.keys, 0, dbgBuffer, 0, dbgBuffer.size);
            //     this.device.queue.submit([commandEncoder.finish()]);
            //     await this.device.queue.onSubmittedWorkDone();

            //     await dbgBuffer.mapAsync(GPUMapMode.READ);

            //     var debugValsu = new Uint32Array(dbgBuffer.getMappedRange());
            //     console.log(debugValsu);
            // }
            let startTot = performance.now();
            var maxIndex = nodeLength;
            commandEncoder = this.device.createCommandEncoder();
            for (var i = 0; i < Math.log(nodeLength) / Math.log(this.clusterSize); i++) {
                // start = performance.now();
                this.device.queue.writeBuffer(
                    treeInfoBuffer,
                    0,
                    new Uint32Array([i]),
                    0,
                    1
                );
                commandEncoder = this.device.createCommandEncoder();
                computePassEncoder = commandEncoder.beginComputePass();
                computePassEncoder.setBindGroup(0, createTreeBindGroup);
                computePassEncoder.setPipeline(this.createTreePipeline);
                computePassEncoder.dispatchWorkgroups(Math.ceil(nodeLength / (128 * this.clusterSize**(i+1))), 1, 1);
                computePassEncoder.end();
                this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();
                maxIndex += Math.ceil(nodeLength / this.clusterSize**(i+1))
                // end = performance.now();
                // console.log(`Create Tree iter ${i} took ${end - start}ms`)
            }
            this.device.queue.writeBuffer(
                treeInfoBuffer,
                4,
                new Uint32Array([maxIndex]),
                0,
                1
            );
            this.device.queue.submit([commandEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            let endTot = performance.now();
            totalTree += endTot - startTot;
            console.log(`Create Tree took ${endTot - startTot}ms`)
            {
                // console.log(nodeLength);
                // var dbgBuffer = this.device.createBuffer({
                //     size: treeBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(treeBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBuffer.mapAsync(GPUMapMode.READ);

                // var debugValsf = new Float32Array(dbgBuffer.getMappedRange());
                // console.log(debugValsf);
                // console.log(maxIndex * 16);
                // var sum_size = 0;
                // for (var tt = nodeLength * 16 + 16; tt < maxIndex * 16 + 16; tt += 16) {
                //     sum_size += debugValsf[tt + 2];
                // }
                // console.log(sum_size / (maxIndex - nodeLength));
                // total_sumsize += sum_size / (maxIndex - nodeLength);
                // dbgBuffer.destroy();
                // var dbgBufferu = this.device.createBuffer({
                //     size: treeBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(treeBuffer, 0, dbgBufferu, 0, dbgBuffer.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBufferu.mapAsync(GPUMapMode.READ);

                // var debugValsu = new Uint32Array(dbgBufferu.getMappedRange());
                // console.log(debugValsu);
            }

            commandEncoder = this.device.createCommandEncoder();
            // const commandEncoder = this.device.createCommandEncoder();
            // start = performance.now();
            // Create BH bindgroup
            const computeForcesBHBindGroup = this.device.createBindGroup({
                layout: this.computeForcesBHPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.nodeDataBuffer,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.forceDataBuffer,
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.paramsBuffer,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: treeInfoBuffer
                        }
                    },
                    {
                        binding: 4,
                        resource: {
                            buffer: treeBuffer
                        }
                    }
                ],
            });

            // Run attract forces pass
            const attractBindGroup = this.device.createBindGroup({
                layout: this.computeAttractiveNewPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: edgeInfoBuffer,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: sourceListBuffer,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: targetListBuffer,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: this.forceDataBuffer,
                        }
                    },
                    {
                        binding: 4,
                        resource: {
                            buffer: this.nodeDataBuffer,
                        }
                    },
                    {
                        binding: 5,
                        resource: {
                            buffer: this.paramsBuffer,
                        },
                    },
                ],
            });
            if (debug) {await this.device.queue.onSubmittedWorkDone();}

            // Run attract forces pass
            computePassEncoder = commandEncoder.beginComputePass();
            computePassEncoder.setBindGroup(0, attractBindGroup);
            computePassEncoder.setPipeline(this.computeAttractiveNewPipeline);
            computePassEncoder.dispatchWorkgroups(Math.ceil(nodeLength / 128), 1, 1);
            computePassEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            start = performance.now();
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            end = performance.now();
            console.log(`attract force time: ${end - start}`)

            // Run compute forces BH pass
            start = performance.now();
            commandEncoder = this.device.createCommandEncoder();

            const pass = commandEncoder.beginComputePass();
            pass.setBindGroup(0, computeForcesBHBindGroup);
            pass.setPipeline(this.computeForcesBHPipeline);
            pass.dispatchWorkgroups(Math.ceil(nodeLength / 128), 1, 1);
            pass.end();
            this.device.queue.submit([commandEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            end = performance.now();
            console.log(`repulse force time: ${end - start}`)

            {
                // var dbgBuffer = this.device.createBuffer({
                //     size: treeBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(treeBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBuffer.mapAsync(GPUMapMode.READ);

                // var debugValsf = new Float32Array(dbgBuffer.getMappedRange());
                // console.log(debugValsf);


                // var dbgBufferu = this.device.createBuffer({
                //     size: stackBuffer.size,
                //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                // });

                // commandEncoder = this.device.createCommandEncoder();
                // commandEncoder.copyBufferToBuffer(stackBuffer, 0, dbgBufferu, 0, stackBuffer.size);
                // this.device.queue.submit([commandEncoder.finish()]);
                // await this.device.queue.onSubmittedWorkDone();

                // await dbgBufferu.mapAsync(GPUMapMode.READ);

                // var debugValsu = new Uint32Array(dbgBufferu.getMappedRange());
                // console.log(debugValsu);
            }

            commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(bounding2, 0, rangeBuffer, 0, 4 * 4);

            start = performance.now();
            computePassEncoder = commandEncoder.beginComputePass();
            //commandEncoder.writeTimestamp();


            // Run apply forces pass
            computePassEncoder.setBindGroup(0, applyBindGroup);
            computePassEncoder.setPipeline(this.applyForcesPipeline);
            computePassEncoder.dispatchWorkgroups(Math.ceil(nodeLength / (2 * 128)), 1, 1);
            computePassEncoder.end();

            // commandEncoder.copyBufferToBuffer(
            //     rangeBuffer /* source buffer */,
            //     0 /* source offset */,
            //     gpuReadBuffer /* destination buffer */,
            //     0 /* destination offset */,
            //     4 * 4 /* size */
            // );

            // commandEncoder.copyBufferToBuffer(
            //     this.nodeDataBuffer,
            //     0,
            //     positionReadBuffer,
            //     0,
            //     nodeLength * 4 * 4
            // );

            this.device.queue.submit([commandEncoder.finish()]);
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            end = performance.now();
            console.log(`apply forces time ${end - start}`)
            // iterationTimes.push(end - start);

            // this.maxForceResultBuffer.unmap();
            // Read all of the forces applied.
            // await gpuReadBuffer.mapAsync(GPUMapMode.READ);
            // const arrayBuffer = gpuReadBuffer.getMappedRange();
            // const output = new Int32Array(arrayBuffer);
            // console.log(output);

            // console.log(output[23]);
            // await gpuReadBuffer3.mapAsync(GPUMapMode.READ);
            // const arrayBuffer3 = gpuReadBuffer3.getMappedRange();
            // const output3 = new Float32Array(arrayBuffer3);
            // console.log(output3);
            // await gpuReadBuffer2.mapAsync(GPUMapMode.READ);
            // const arrayBuffer2 = gpuReadBuffer2.getMappedRange();
            // const output2 = new Uint32Array(arrayBuffer2);
            // console.log(output2);
            // for (var m = 0; m < output.length; m += 12) {
            //     const mass = output[m + 10];
            //     if (
            //         output2[m + 4] == 1000000 ||
            //         output2[m + 5] == 1000000 ||
            //         output2[m + 6] == 1000000 ||
            //         output2[m + 7] == 1000000 
            //     ) {
            //         console.log(m);
            //         break;
            //     }
            // }
            // if (output[11] > 0) {
            //     break;
            // }
            // this.coolingFactor = this.coolingFactor * 0.99;
            this.coolingFactor = this.coolingFactor * 0.9;
            if (numIterations % 20 == 0 && numIterations < iterationCount - 20) {
                if (!this.stopForce) {
                    this.coolingFactor = this.energy;
                }
            }
            console.log(`Iteration ${numIterations}`);
            // label.innerText = `Iteration ${numIterations}`;
            if (debug) {await this.device.queue.onSubmittedWorkDone();}
            const frameEnd = performance.now();
            console.log(`Total frame time: ${frameEnd - frameStart}`);
            totalTime += frameEnd - frameStart;
            if (numIterations % 10 == 0) {
                requestAnimationFrame(frame);
                await this.device.queue.onSubmittedWorkDone();
            }
        }
        await positionReadBuffer.mapAsync(GPUMapMode.READ);
        // let positionArrayBuffer = positionReadBuffer.getMappedRange();
        // let positionList = new Float32Array(positionArrayBuffer);
        await this.device.queue.onSubmittedWorkDone();
        const totalEnd = performance.now();
        console.log(`Average tree node size: ${total_sumsize / 100}`);

        // const iterAvg : number = iterationTimes.reduce(function(a, b) {return a + b}) / iterationTimes.length;
        console.log(`Completed in ${numIterations} iterations with total time ${totalEnd - totalStart} average iteration time ${(totalEnd - totalStart) / (numIterations - 1)}`);
        // label.innerText = `Completed in ${numIterations} iterations with total time ${totalEnd - totalStart} average iteration time ${(totalEnd - totalStart) / (numIterations - 1)}`;
        // let d3Format = this.formatToD3Format(
        //     positionList,
        //     edgeList,
        //     nodeLength,
        //     edgeLength
        //   );
        // let formattedNodeList = d3Format.nodeArray;
        // let formattedEdgeList = d3Format.edgeArray;

        // console.log(formattedNodeList, formattedEdgeList);
        // const element = document.createElement("a");
        // const textFile = new Blob([JSON.stringify(formattedEdgeList)], {type: 'application/json'});
        // element.href = URL.createObjectURL(textFile);
        // element.download = "BH_edges.json";
        // document.body.appendChild(element); 
        // element.click();
        // const element2 = document.createElement("a");
        // const textFile2 = new Blob([JSON.stringify(formattedNodeList)], {type: 'application/json'});
        // element.href = URL.createObjectURL(textFile2);
        // element.download = "BH_nodes.json";
        // document.body.appendChild(element2); 
        // element.click();


        requestAnimationFrame(frame);
    }
}