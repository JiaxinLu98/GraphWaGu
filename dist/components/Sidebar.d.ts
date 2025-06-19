import { default as React } from 'react';
declare const datasets: {
    sf_ba6000: Promise<any>;
    'fe_4elt2.mtx': Promise<any>;
    'pkustk02.mtx': Promise<any>;
    'pkustk01.mtx': Promise<any>;
    'finance256.mtx': Promise<any>;
};
type SidebarProps = {
    setNodeEdgeData: (nodeData: Array<number>, edgeData: Array<number>, sourceEdges: Array<number>, targetEdges: Array<number>) => void;
    setCoolingFactor: (value: number) => void;
    setIdealLength: (value: number) => void;
    setTheta: (value: number) => void;
    setEnergy: (value: number) => void;
    setIterationCount: (value: number) => void;
    toggleNodeLayer: () => void;
    toggleEdgeLayer: () => void;
    runForceDirected: () => void;
    stopForceDirected: () => void;
    takeScreenshot: () => void;
};
type SidebarState = {
    nodeData: Array<number>;
    edgeData: Array<number>;
    sourceEdges: Array<number>;
    targetEdges: Array<number>;
    adjacencyMatrix: Array<Array<number>>;
};
type edge = {
    source: number;
    target: number;
};
type Graph = {
    nodes: Array<any>;
    edges: Array<edge>;
};
declare class Sidebar extends React.Component<SidebarProps, SidebarState> {
    constructor(props: SidebarProps | Readonly<SidebarProps>);
    handleSubmit(event: {
        preventDefault: () => void;
    }): void;
    loadGraph(graph: Graph): void;
    readJson(event: React.ChangeEvent<HTMLInputElement>): Promise<void>;
    chooseDataset(dataset: keyof typeof datasets): Promise<void>;
    render(): import("react/jsx-runtime").JSX.Element;
}
export default Sidebar;
