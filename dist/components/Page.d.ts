import { default as React, MutableRefObject } from 'react';
import { Renderer } from '../webgpu/render';
type PageState = {
    canvasRef: MutableRefObject<HTMLCanvasElement | null>;
    iterRef: MutableRefObject<HTMLLabelElement | null>;
    renderer: Renderer | null;
    renderTutorial: boolean;
    renderAlert: boolean;
};
declare class Page extends React.Component<{}, PageState> {
    constructor(props: {} | Readonly<{}>);
    componentDidMount(): Promise<void>;
    setNodeEdgeData(nodeData: Array<number>, edgeData: Array<number>, sourceEdges: Array<number>, targetEdges: Array<number>): void;
    setIdealLength(value: number): void;
    setEnergy(value: number): void;
    setTheta(value: number): void;
    setCoolingFactor(value: number): void;
    setIterationCount(value: number): void;
    toggleNodeLayer(): void;
    toggleEdgeLayer(): void;
    runForceDirected(): void;
    stopForceDirected(): void;
    takeScreenshot(): void;
    unmountTutorial(): void;
    render(): import("react/jsx-runtime").JSX.Element;
}
export default Page;
