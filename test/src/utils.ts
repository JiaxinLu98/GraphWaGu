export async function loadGraphFromPath(path: string): Promise<{
  nodes: number[],
  edges: number[],
  sources: number[],
  targets: number[]
}> {
  const response = await fetch(path);
  // const graph = await response.json();
  const dotText = await response.text();

  const nodes: number[] = [];
  const edges: number[] = [];
  const sources: number[] = [];
  const targets: number[] = [];

  // Collect all nodes (string)
  const nodeSet = new Set<string>();
  const edgeList: [string, string][] = [];
  const lines = dotText.split('\n');

  for(const line of lines) {
    const trimmed = line.trim();
    const edgeMatch = trimmed.match(/^([\w\d]+)\s*->\s*([\w\d]+)\s*(\[[^\]]*\])?;$/);
    if(edgeMatch) {
      const source = edgeMatch[1];
      const target = edgeMatch[2];
      nodeSet.add(source);
      nodeSet.add(target);
      edgeList.push([source, target]);
      // console.log(`source: ${source}`);
      // console.log(`target: ${target}`);
      // console.log(`edges: ${edgeList}`);
    }
  }

  // Assign an integer number to each string ID
  const idMap = new Map<string, number>();
  let currentId = 0;
  for(const nodeId of nodeSet) {
    idMap.set(nodeId, currentId);
    currentId++;
    console.log(`string -> integer: ${nodeId} -> ${currentId - 1}`);
  }

  // Create the node
  for(let i = 0; i < idMap.size; i++) {
    const x = Math.random();
    const y = Math.random();
    nodes.push(0, x, y, 1.0);
  }

  const graphEdges = edgeList.map(([srcStr, tgtStr]) => {
    return {
      source: idMap.get(srcStr)!,
      target: idMap.get(tgtStr)!
    };
  });

  for (const edge of graphEdges) {
    edges.push(edge.source, edge.target);
  }

  const sortedBySource = [...graphEdges].sort((a, b) => a.source - b.source);
  for (const edge of sortedBySource) {
    sources.push(edge.source, edge.target);
  }

  const sortedByTarget = [...graphEdges].sort((a, b) => a.target - b.target);
  for (const edge of sortedByTarget) {
    targets.push(edge.source, edge.target);
  }

  // for (const node of graph.nodes) {
  //   const x = node.x ?? Math.random();
  //   const y = node.y ?? Math.random();
  //   nodes.push(0, x, y, 1.0);
  // }

  // for (const edge of graph.edges) {
  //   edges.push(edge.source, edge.target);
  // }

  // const sortedBySource = [...graph.edges].sort((a, b) => a.source - b.source);
  // for (const edge of sortedBySource) {
  //   sources.push(edge.source, edge.target);
  // }

  // const sortedByTarget = [...graph.edges].sort((a, b) => a.target - b.target);
  // for (const edge of sortedByTarget) {
  //   targets.push(edge.source, edge.target);
  // }

  return { nodes, edges, sources, targets };
}