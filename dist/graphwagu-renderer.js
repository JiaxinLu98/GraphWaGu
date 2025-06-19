var ue = Object.defineProperty;
var de = (h, e, t) => e in h ? ue(h, e, { enumerable: !0, configurable: !0, writable: !0, value: t }) : h[e] = t;
var s = (h, e, t) => (de(h, typeof e != "symbol" ? e + "" : e, t), t);
const fe = `struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Nodes {
    nodes : array<Node>,
};
struct Forces {
    forces : array<f32>,
};
struct Batch {
    batch_id : u32,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};
struct Range {
    x_min : atomic<i32>,
    x_max : atomic<i32>,
    y_min : atomic<i32>,
    y_max : atomic<i32>,
};
@group(0) @binding(0) var<storage, read_write> nodes : Nodes;
@group(0) @binding(1) var<storage, read_write> forces : Forces;
// @group(0) @binding(2) var<uniform> batch : Batch;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
@group(0) @binding(3) var<storage, read_write> bounding : Range;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= uniforms.nodes_length) {
        return;
    }
    var high : f32 = 8.0;
    var low : f32 = -7.0;
    var batch_index : u32 = global_id.x;
    for (var iter = 0u; iter < 2u; iter = iter + 1u) {
        // nodes.nodes[batch_index].x = nodes.nodes[batch_index].x + forces.forces[batch_index * 2u];
        // nodes.nodes[batch_index].y = nodes.nodes[batch_index].y + forces.forces[batch_index * 2u + 1u]; 
        if (forces.forces[batch_index * 2u] > uniforms.cooling_factor) {
            // atomicStore(&bounding.y_max, i32(batch_index));
            forces.forces[batch_index * 2u] = 0.0;    
        }
        if (forces.forces[batch_index * 2u + 1u] > uniforms.cooling_factor) {
            // atomicStore(&bounding.y_min, i32(batch_index));
            forces.forces[batch_index * 2u + 1u] = 0.0;    
        }
        // var x : f32 = min(high, max(low, nodes.nodes[batch_index].x + forces.forces[batch_index * 2u]));
        // var y : f32 = min(high, max(low, nodes.nodes[batch_index].y + forces.forces[batch_index * 2u + 1u]));
        var x : f32 = nodes.nodes[batch_index].x + forces.forces[batch_index * 2u];
        var y : f32 = nodes.nodes[batch_index].y + forces.forces[batch_index * 2u + 1u];

        // var centering : vec2<f32> = normalize(vec2<f32>(0.5, 0.5) - vec2<f32>(x, y));
        // var dist : f32 = distance(vec2<f32>(0.5, 0.5), vec2<f32>(x, y));
        // x = x + centering.x * (0.1 * uniforms.cooling_factor * dist);
        // y = y + centering.y * (0.1 * uniforms.cooling_factor * dist);
        // Randomize position slightly to prevent exact duplicates after clamping
        // if (x == high) {
        //     x = x - f32(batch_index) / 500000.0; 
        // } 
        // if (y == high) {
        //     y = y - f32(batch_index) / 500000.0; 
        // }
        // if (x == low) {
        //     x = x + f32(batch_index) / 500000.0; 
        // }
        // if (y == low) {
        //     y = y + f32(batch_index) / 500000.0; 
        // }
        nodes.nodes[batch_index].x = x;
        nodes.nodes[batch_index].y = y;
        forces.forces[batch_index * 2u] = 0.0;
        forces.forces[batch_index * 2u + 1u] = 0.0;
        atomicMin(&bounding.x_min, i32(floor(x * 1000.0)));
        atomicMax(&bounding.x_max, i32(ceil(x * 1000.0)));
        atomicMin(&bounding.y_min, i32(floor(y * 1000.0)));
        atomicMax(&bounding.y_max, i32(ceil(y * 1000.0)));


        // var test : f32 = forces.forces[0]; 
        // var test2 : f32 = nodes.nodes[0].x;
        batch_index = batch_index + (uniforms.nodes_length / 2u);
    }
}
`, ce = `struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Nodes {
    nodes : array<Node>,
};
struct Forces {
    forces : array<f32>,
};
struct UintArray {
    a : array<u32>,
};
struct EdgeInfo {
    source_start : u32,
    source_degree : u32,
    dest_start : u32,
    dest_degree : u32,
}
struct EdgeInfoArray {
    a : array<EdgeInfo>,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};

@group(0) @binding(0) var<storage, read_write> edge_info : EdgeInfoArray;
@group(0) @binding(1) var<storage, read> source_list : UintArray;
@group(0) @binding(2) var<storage, read> dest_list : UintArray;
@group(0) @binding(3) var<storage, read_write> forces : Forces;
@group(0) @binding(4) var<storage, read> nodes : Nodes;
@group(0) @binding(5) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= uniforms.nodes_length) {
        return;
    }
    let l : f32 = uniforms.ideal_length;
    var node : Node = nodes.nodes[global_id.x];
    var a_force : vec2<f32> = vec2<f32>(0.0, 0.0);
    var info : EdgeInfo = edge_info.a[global_id.x];
    // Accumulate forces where node is the source
    for (var i : u32 = info.source_start; i < info.source_start + info.source_degree; i = i + 1u) {
        var node2 : Node = nodes.nodes[source_list.a[i]];
        var dist : f32 = distance(vec2<f32>(node.x, node.y), vec2<f32>(node2.x, node2.y));
        if(dist > 0.0000001) {
            var dir : vec2<f32> = normalize(vec2<f32>(node2.x, node2.y) - vec2<f32>(node.x, node.y));
            a_force = a_force + ((dist * dist) / l) * dir;
        }
    }
    // Accumulate forces where node is the dest
    for (var i : u32 = info.dest_start; i < info.dest_start + info.dest_degree; i = i + 1u) {
        var node2 : Node = nodes.nodes[dest_list.a[i]];
        var dist : f32 = distance(vec2<f32>(node.x, node.y), vec2<f32>(node2.x, node2.y));
        if(dist > 0.0000001) {
            var dir : vec2<f32> = normalize(vec2<f32>(node2.x, node2.y) - vec2<f32>(node.x, node.y));
            a_force = a_force + ((dist * dist) / l) * dir;
        }
    }
    forces.forces[global_id.x * 2u] = a_force.x;
    forces.forces[global_id.x * 2u + 1u] = a_force.y;
}`, le = `const cluster_size = CHANGEMEu;
struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Edges {
    edges : array<u32>,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};
struct TreeInfo {
    step : u32,
    max_index : u32,
    theta : f32,
};
struct Range {
    x_min : i32,
    x_max : i32,
    y_min : i32,
    y_max : i32,
};
struct TreeNode {
    // x, y, width, height
    boundary : vec4<f32>,
    CoM : vec2<f32>,
    mass : f32,
    test : u32,
    code : u32,
    level : u32,
    test2: u32,
    test3: u32,
    pointers : array<u32, cluster_size>,
};

@group(0) @binding(0) var<storage, read> nodes : array<Node>;
@group(0) @binding(1) var<storage, read_write> forces : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
@group(0) @binding(3) var<uniform> tree_info : TreeInfo;
@group(0) @binding(4) var<storage, read> tree : array<TreeNode>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var stack = array<u32, 64>();
    let l : f32 = uniforms.ideal_length;
    var index : u32 = global_id.x;
    if (index >= uniforms.nodes_length) {
        return;
    }
    let node = nodes[index];
    var theta : f32 = tree_info.theta;
    var r_force : vec2<f32> = vec2<f32>(0.0, 0.0);
    var a_force : vec2<f32> = vec2<f32>(forces[index * 2u], forces[index * 2u + 1u]);
    var tree_idx : u32 = tree_info.max_index;
    var counter : u32 = 0u;
    var out : u32 = 0u;
    loop {
        out = out + 1u;
        // if (out == 1000u) {
        //     break;
        // }
        var tree_node = tree[tree_idx];
        let dist : f32 = distance(vec2<f32>(node.x, node.y), tree_node.CoM);
        let s : f32 = 2.0 * tree_node.boundary.w;
        if (theta > s / dist) {
            var dir : vec2<f32> = normalize(vec2<f32>(node.x, node.y) - tree_node.CoM);
            r_force = r_force + tree_node.mass * ((l * l) / dist) * dir;
        } else {
            for (var i : u32 = 0u; i < cluster_size; i = i + 1u) {
                let child : u32 = tree_node.pointers[i];
                if (child == 0 || tree[child].mass < 1.0) {
                    continue;
                } else {
                    if (tree[child].mass > 1.0) {
                        stack[counter] = child;
                        counter = counter + 1u;
                    } else {
                        let dist : f32 = distance(vec2<f32>(node.x, node.y), tree[child].CoM);
                        if (dist > 0.0) {
                            var dir : vec2<f32> = normalize(vec2<f32>(node.x, node.y) - tree[child].CoM);
                            r_force = r_force + ((l * l) / dist) * dir;
                        }
                    }
                }
            }
        }
        counter--;
        if (counter < 0u) {
            break;
        }
        tree_idx = stack[counter];
        if (tree_idx == 0u) {
            break;
        } 
    }
    var force : vec2<f32> = (a_force + r_force);
    var localForceMag: f32 = length(force); 
    if (localForceMag>0.000000001) {
        force = normalize(force) * min(uniforms.cooling_factor, length(force));
    }
    else{
        force.x = 0.0;
        force.y = 0.0;
    }
    if (force.x > uniforms.cooling_factor) {
        force.x = 0.0;
    }
    if (force.y > uniforms.cooling_factor) {
        force.y = 0.0;
    }
    forces[index * 2u] = force.x;
    forces[index * 2u + 1u] = force.y;
}
`, _e = `struct Edges {
    edges : array<u32>,
};
struct UintArray {
    a : array<u32>,
};
struct EdgeInfo {
    source_start : u32,
    source_degree : u32,
    dest_start : u32,
    dest_degree : u32,
}
struct EdgeInfoArray {
    a : array<EdgeInfo>,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};

@group(0) @binding(0) var<storage, read_write> edges : Edges;
@group(0) @binding(1) var<storage, read_write> edge_info : EdgeInfoArray;
@group(0) @binding(2) var<storage, read_write> source_list : UintArray;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var counter : u32 = 0u;
    var source : u32 = 0u;
    // expects edges to be sorted by source id
    for (var i : u32 = 0u; i < uniforms.edges_length; i = i + 2u) {
        var new_source : u32 = edges.edges[i];
        var dest : u32 = edges.edges[i + 1u];
        edge_info.a[new_source].source_degree = edge_info.a[new_source].source_degree + 1u;
        source_list.a[counter] = dest;
        if (new_source != source || i == 0u) {
            edge_info.a[new_source].source_start = counter;
        }
        counter = counter + 1u;
        source = new_source;
    }
}`, ge = `struct Edges {
    edges : array<u32>,
};
struct UintArray {
    a : array<u32>,
};
struct EdgeInfo {
    source_start : u32,
    source_degree : u32,
    dest_start : u32,
    dest_degree : u32,
}
struct EdgeInfoArray {
    a : array<EdgeInfo>,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};

@group(0) @binding(0) var<storage, read_write> edges : Edges;
@group(0) @binding(1) var<storage, read_write> edge_info : EdgeInfoArray;
@group(0) @binding(2) var<storage, read_write> dest_list : UintArray;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var counter : u32 = 0u;
    var dest : u32 = 0u;
    // expects edges to be sorted by dest id
    for (var i : u32 = 0u; i < uniforms.edges_length; i = i + 2u) {
        var source : u32 = edges.edges[i];
        var new_dest : u32 = edges.edges[i + 1u];
        edge_info.a[new_dest].dest_degree = edge_info.a[new_dest].dest_degree + 1u;
        dest_list.a[counter] = source;
        if (new_dest != dest || i == 0u) {
            edge_info.a[new_dest].dest_start = counter;
        }
        counter = counter + 1u;
        dest = new_dest;
    }
}`, me = `const cluster_size = CHANGEMEu;
struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};
struct TreeInfo {
    step : u32,
    max_index : u32,
    theta: f32,
    cluster_size: u32,
};
struct Range {
    x_min : i32,
    x_max : i32,
    y_min : i32,
    y_max : i32,
};
struct TreeNode {
    // x, y, width, height
    boundary : vec4<f32>,
    CoM : vec2<f32>,
    mass : f32,
    test : u32,
    code : u32,
    level : u32,
    test2: u32,
    test3: u32,
    pointers : array<u32, cluster_size>,
};

@group(0) @binding(0) var<storage, read> indices : array<u32>;
@group(0) @binding(1) var<uniform> uniforms : Uniforms;
@group(0) @binding(2) var<uniform> tree_info : TreeInfo;
@group(0) @binding(3) var<storage, read_write> bounding : Range;
@group(0) @binding(4) var<storage, read_write> tree : array<TreeNode>;

// Find the level above where two Morton codes first disagree
fn find_morton_split_level(morton1: u32, morton2: u32) -> u32 {
    // XOR the Morton codes to find differing bits
    let diff = morton1 ^ morton2;
    
    // If codes are identical, return 16
    if (diff == 0u) {
        return 16u;
    }
    
    // Find position of highest different bit
    var highest_diff_bit = 31u;
    var temp = diff;
    
    // Count leading zeros
    while ((temp & 0x80000000u) == 0u) {
        temp = temp << 1u;
        highest_diff_bit = highest_diff_bit - 1u;
    }
    
    // Convert bit position to level
    let level = 16u - (highest_diff_bit + 2u) / 2u;
    return level;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let x_min = f32(bounding.x_min) / 1000.0;
    let x_max = f32(bounding.x_max) / 1000.0;
    let y_min = f32(bounding.y_min) / 1000.0;
    let y_max = f32(bounding.y_max) / 1000.0;
    let step = tree_info.step;
    var idx = global_id.x * cluster_size;
    var start = f32(uniforms.nodes_length);
    var end = uniforms.nodes_length;
    for (var i = 0u; i < step; i++) {
        idx += u32(start);
        start = ceil(start / f32(cluster_size));
        end += u32(start);
    }
    if (idx >= end) {
        return;
    }
    var pointers = array<u32, cluster_size>();
    if (step == 0u) {
        for (var i = 0u; i < cluster_size; i++) {
            if (idx + i >= end) {
                pointers[i] = 0;
            } else {
                pointers[i] = indices[idx + i] + 1;
            }
        }
    } else {
        for (var i = 0u; i < cluster_size; i++) {
             if (idx + i >= end) {
                pointers[i] = 0;
            } else {
                pointers[i] = idx + i + 1;
            }
        }
    }
    var node = tree[pointers[0]];
    var code = node.code;
    var level = node.level;
    var mass = node.mass;
    var CoM = node.CoM;
    for (var i = 1u; i < cluster_size; i++) {
        if (idx + i >= end) {
            break;
        }
        node = tree[pointers[i]];
        level = min(find_morton_split_level(code, node.code), min(level, node.level));
        CoM = (mass * CoM + node.mass * node.CoM) / (mass + node.mass);
        mass = mass + node.mass;
    }
    tree[end + global_id.x + 1] = TreeNode(
        vec4<f32>(0.0, 0.0, (1.0 / f32(1u << level)) * (x_max - x_min), (1.0 / f32(1u << level)) * (y_max - y_min)),
        CoM,
        mass, 
        0u, 
        code, level, 0u, 0u,
        pointers,
    );
    //  PROBLEM WITH POINTERS ARRAY
    // let node1 = tree[pointers[0]];
    // let node2 = tree[pointers[1]];
    // let node3 = tree[pointers[2]];
    // let node4 = tree[pointers[3]];
    // let morton1 = node1.code;
    // let morton2 = node2.code;
    // let morton3 = node3.code;
    // let morton4 = node4.code;
    // if (idx == end - 1) {
    //     // Just write the node out without merging with anything
    //     tree[end + global_id.x + 1] = node1;
    //     return;
    // }
    // if (idx == end - 2) {
    //     let level = min(find_morton_split_level(morton1, morton2), min(node1.level, node2.level));
    //     tree[end + global_id.x + 1] = TreeNode(
    //         vec4<f32>(0.0, 0.0, 1.0 / f32(1u << level), 1.0 / f32(1u << level)),
    //         (node1.mass * node1.CoM + node2.mass * node2.CoM) / (node1.mass + node2.mass),
    //         node1.mass + node2.mass, 
    //         morton2, 
    //         pointers,
    //         morton1, level
    //     );
    //     return;
    // }
    // if (idx == end - 3) {
    //     let level = min(min(find_morton_split_level(morton3, morton2), min(find_morton_split_level(morton1, morton2), min(node1.level, node2.level))), node3.level);
    //     tree[end + global_id.x + 1] = TreeNode(
    //         vec4<f32>(0.0, 0.0, 1.0 / f32(1u << level), 1.0 / f32(1u << level)),
    //         (node1.mass * node1.CoM + node2.mass * node2.CoM + node3.mass * node3.CoM) / (node1.mass + node2.mass + node3.mass),
    //         node1.mass + node2.mass + node3.mass, 
    //         morton2, 
    //         pointers,
    //         morton1, level
    //     );
    //     return;
    // }
    // let level12 = min(find_morton_split_level(morton1, morton2), min(node1.level, node2.level));
    // let level34 = min(find_morton_split_level(morton3, morton4), min(node3.level, node4.level));
    // let level = min(find_morton_split_level(morton2, morton3), min(level12, level34));
    // tree[end + global_id.x + 1] = TreeNode(
    //     vec4<f32>(0.0, 0.0, 1.0 / f32(1u << level), 1.0 / f32(1u << level)),
    //     (node1.mass * node1.CoM + node2.mass * node2.CoM + node3.mass * node3.CoM + node4.mass * node4.CoM) / (node1.mass + node2.mass + node3.mass + node4.mass),
    //     node1.mass + node2.mass + node3.mass + node4.mass, 
    //     morton2, 
    //     pointers,
    //     morton1, level
    // );
}
`, pe = `@fragment
fn main()->@location(0) vec4<f32>{
    return vec4<f32>(0.0, 0.0, 0.0, 0.1);
}`, he = `//this builtin(position) clip_position tells that clip_position is the value we want to use for our vertex position or clip position
//it's not needed to create a struct, we could just do [[builtin(position)]] clipPosition
struct VertexOutput{
    @builtin(position) clip_position: vec4<f32>,
};
struct Uniforms {
  view_box : vec4<f32>,
};
struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Nodes {
    nodes : array<Node>,
};
struct Edges {
    edges : array<u32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> nodes : Nodes;
@group(0) @binding(2) var<storage, read> edges : Edges;
@vertex
fn main(@builtin(instance_index) index : u32, @location(0) position: vec2<f32>)-> VertexOutput {
    var out : VertexOutput;
    var node : Node = nodes.nodes[edges.edges[2u * index + u32(position.x)]];
    var inv_zoom : f32 = uniforms.view_box.z - uniforms.view_box.x;
    var expected_x : f32 = 0.5 * (1.0 - inv_zoom); 
    var expected_y : f32 = 0.5 * (1.0 - inv_zoom);
    // view_box expected to be between 0 and 1, panning need to be doubled as clip space is (-1, 1)
    var x : f32 = ((2.0 * node.x - 1.0) - 2.0 * (uniforms.view_box.x - expected_x)) / inv_zoom;
    var y : f32 = ((2.0 * node.y - 1.0) - 2.0 * (uniforms.view_box.y - expected_y)) / inv_zoom;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}`, ve = `const cluster_size = CHANGEMEu;
struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Uniforms {
    nodes_length : u32,
    edges_length : u32,
    cooling_factor : f32,
    ideal_length : f32,
};
struct Range {
    x_min : i32,
    x_max : i32,
    y_min : i32,
    y_max : i32,
};
struct TreeNode {
    // x, y, width, height
    boundary : vec4<f32>,
    CoM : vec2<f32>,
    mass : f32,
    test : u32,
    code : u32,
    level : u32,
    test2: u32,
    test3: u32,
    pointers : array<u32, cluster_size>,
};

@group(0) @binding(0) var<storage, read> nodes : array<Node>;
@group(0) @binding(1) var<storage, read_write> morton_codes : array<u32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
@group(0) @binding(3) var<storage, read_write> bounding : Range;
@group(0) @binding(4) var<storage, read_write> morton_indices : array<u32>;
@group(0) @binding(5) var<storage, read_write> tree : array<TreeNode>;

// Spreads bits by inserting 0s between each bit
fn spread_bits(x: u32) -> u32 {
    var x_mut = x & 0x0000FFFF;  // Mask to ensure we only use lower 16 bits
    x_mut = (x_mut | (x_mut << 8)) & 0x00FF00FF;
    x_mut = (x_mut | (x_mut << 4)) & 0x0F0F0F0F;
    x_mut = (x_mut | (x_mut << 2)) & 0x33333333;
    x_mut = (x_mut | (x_mut << 1)) & 0x55555555;
    return x_mut;
}

// Converts float in [0,1] to fixed-point integer
// TODO: precision lost here
fn float_to_fixed(f: f32) -> u32 {
    return u32(f * 65535.0);  // 2^16 - 1
}

// Convert morton code to quadrant boundaries
fn morton_to_rectangle(morton: u32, level: u32) -> vec4<f32> {    
    // Initialize normalized coordinates
    var x = 0.0;
    var y = 0.0;
    var size = 1.0;
    
    // Process each pair of bits from most significant to least
    for(var i = 0u; i < level; i++) {
        size *= 0.5; // Each level divides size by 2
        let shift = (15u - i) * 2u;
        let bits = (morton >> shift) & 3u; // Get pair of bits
        
        // Update position based on quadrant
        switch bits {
            case 0u: { // 00: bottom left
                // Position stays the same
            }
            case 1u: { // 01: bottom right
                x += size;
            }
            case 2u: { // 10: top left
                y += size;
            }
            case 3u: { // 11: top right
                x += size;
                y += size;
            }
            default: {}
        }
    }
    
    // Convert from normalized coordinates to world space
    let x_min = f32(bounding.x_min) / 1000.0;
    let x_max = f32(bounding.x_max) / 1000.0;
    let y_min = f32(bounding.y_min) / 1000.0;
    let y_max = f32(bounding.y_max) / 1000.0;
    
    let world_x = x * (x_max - x_min) + x_min;
    let world_y = y * (y_max - y_min) + y_min;
    let world_w = size * (x_max - x_min);
    let world_h = size * (y_max - y_min);
    
    return vec4<f32>(world_x, world_y, world_w, world_h);
}

fn rotate_bits(n: u32, rx: u32, ry: u32, order: u32) -> u32 {
    if (ry == 0u) {
        if (rx == 1u) {
            // Reflect about y=x
            let mask = (1u << order) - 1u;
            return mask - n;
        }
    }
    return n;
}

fn hilbert_xy_to_d(x_in: u32, y_in: u32) -> u32 {
    var d: u32 = 0u;
    var x: u32 = x_in;
    var y: u32 = y_in;
    var rx: u32;
    var ry: u32;
    
    // Process 16 bits of input coordinates
    for(var i: u32 = 0u; i < 16u; i += 1u) {
        let s = 15u - i;
        
        // Extract current bit of x and y from highest positions
        rx = (x >> 15u) & 1u;
        ry = (y >> 15u) & 1u;
        
        // Add position to result
        d |= ((3u * rx) ^ ry) << (2u * s);
        
        // Rotate coordinates if needed for next iteration
        if (ry == 0u) {
            if (rx == 1u) {
                // Reflect about y=x
                x = (1u << 16u) - 1u - x;
                y = (1u << 16u) - 1u - y;
            }
            // Swap x and y
            let t = x;
            x = y;
            y = t;
        }
        
        // Shift coordinates for next iteration
        x = (x << 1u) & 0xFFFFu;
        y = (y << 1u) & 0xFFFFu;
    }
    
    return d;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.nodes_length) {
        return;
    }
    let node = nodes[idx];
    
    // Convert floats to fixed-point
    let x_min = f32(bounding.x_min) / 1000.0;
    let x_max = f32(bounding.x_max) / 1000.0;
    let y_min = f32(bounding.y_min) / 1000.0;
    let y_max = f32(bounding.y_max) / 1000.0;
    let x_fixed = float_to_fixed((node.x - x_min) / (x_max - x_min));
    let y_fixed = float_to_fixed((node.y - y_min) / (y_max - y_min));
    
    // Compute Morton code by interleaving bits
    let morton = spread_bits(x_fixed) | (spread_bits(y_fixed) << 1);
    let hilbert = hilbert_xy_to_d(x_fixed, y_fixed);
    let code = hilbert;
    
    morton_codes[idx] = code;
    // morton_codes[idx] = morton;
    morton_indices[idx] = idx;
    // tree[idx + 1u] = TreeNode(
    //     morton_to_rectangle(morton, 16),
    //     vec2<f32>(node.x, node.y),
    //     1.0, 0.0, vec4<u32>(0u),
    //     morton, 16u
    // );
    tree[idx + 1u] = TreeNode(
        vec4<f32>(0.0, 0.0, (1.0 / f32(1u << 16u)) * (x_max - x_min), (1.0 / f32(1u << 16u)) * (y_max - y_min)),
        vec2<f32>(node.x, node.y),
        1.0, 0u,
        code, 16u, 0u, 0u,
        array<u32, cluster_size>()
    );
}
`, be = `fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

@fragment
fn main(@location(0) position: vec2<f32>, @location(1) @interpolate(flat) center: vec2<f32>, @location(2) color: vec3<f32>) -> @location(0) vec4<f32> {
    if (distance(position, center) > 0.002) {
        discard;
    }
    return vec4<f32>(color.x, color.y, color.z, 1.0);
}
`, xe = `struct Node {
    value : f32,
    x : f32,
    y : f32,
    size : f32,
};
struct Nodes {
    nodes : array<Node>,
};
struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) @interpolate(flat) center : vec2<f32>,
    @location(2) color: vec3<f32>,
};
struct Uniforms {
  view_box : vec4<f32>,
};
struct Edges {
    edges : array<u32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> nodes : Nodes;
@group(0) @binding(2) var<storage, read> morton_codes : array<u32>;

fn u32_to_color(value: u32) -> vec3<f32> {
    // First convert u32 to f32 in [0,1] range
    // We need to be careful about precision here
    // Break the u32 into two parts to maintain precision
    let upper = f32(value >> 16u);
    let lower = f32(value & 0xFFFFu);
    
    // Combine the parts with appropriate scaling
    let normalized = (upper * 65536.0 + lower) / 4294967295.0;
    
    // Define the color gradient
    // Here we'll use a simple RGB gradient: blue -> cyan -> green -> yellow -> red
    let positions = array<f32, 5>(0.0, 0.25, 0.5, 0.75, 1.0);
    let colors = array<vec3<f32>, 5>(
        vec3<f32>(0.0, 0.0, 1.0),  // Blue
        vec3<f32>(0.0, 1.0, 1.0),  // Cyan
        vec3<f32>(0.0, 1.0, 0.0),  // Green
        vec3<f32>(1.0, 1.0, 0.0),  // Yellow
        vec3<f32>(1.0, 0.0, 0.0)   // Red
    );
    
    // Find the segment
    var i = 0;
    while i < 4 && normalized > positions[i + 1] {
        i = i + 1;
    }
    
    // Calculate interpolation factor
    let t = (normalized - positions[i]) / (positions[i + 1] - positions[i]);
    
    // Interpolate between colors
    let color = mix(colors[i], colors[i + 1], t);
    
    return color;
}

@vertex
fn main(@builtin(instance_index) index : u32, @location(0) position : vec2<f32>)
     -> VertexOutput {
    var node_center : vec2<f32> = 2.0 * vec2<f32>(nodes.nodes[index].x, nodes.nodes[index].y) - vec2<f32>(1.0);
    var translation : vec2<f32> = position * 0.01;
    var out_position : vec2<f32> = node_center + translation;
    var output : VertexOutput;
    var inv_zoom : f32 = uniforms.view_box.z - uniforms.view_box.x;
    var expected_x : f32 = 0.5 * (1.0 - inv_zoom); 
    var expected_y : f32 = 0.5 * (1.0 - inv_zoom);
    // view_box expected to be between 0 and 1, panning need to be doubled as clip space is (-1, 1)
    var x : f32 = (out_position.x - 2.0 * (uniforms.view_box.x - expected_x)) / inv_zoom;
    var y : f32 = (out_position.y - 2.0 * (uniforms.view_box.y - expected_y)) / inv_zoom;
    output.Position = vec4<f32>(x, y, 0.0, 1.0);
    output.position = out_position;
    // flat interpolated position will give bottom right corner, so translate to center
    output.center = node_center;
    let test = morton_codes[index];
    output.color = u32_to_color(test);
    return output;
}`, ye = `// shader implementing gpu radix sort. More information in the beginning of gpu_rs.rs
// info: 

// also the workgroup sizes are added in these prepasses
// before the pipeline is started the following constant definitionis are prepended to this shadercode

// const histogram_sg_size
// const histogram_wg_size
// const rs_radix_log2
// const rs_radix_size
// const rs_keyval_size
// const rs_histogram_block_rows
// const rs_scatter_block_rows

struct GeneralInfo {
    num_keys: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0)
var<storage, read_write> infos: GeneralInfo;
@group(0) @binding(1)
var<storage, read_write> histograms : array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys : array<u32>;
@group(0) @binding(3)
var<storage, read_write> keys_b : array<u32>;
@group(0) @binding(4)
var<storage, read_write> payload_a : array<u32>;
@group(0) @binding(5)
var<storage, read_write> payload_b : array<u32>;

// layout of the histograms buffer
//   +---------------------------------+ <-- 0
//   | histograms[keyval_size]         |
//   +---------------------------------+ <-- keyval_size                           * histo_size
//   | partitions[scatter_blocks_ru-1] |
//   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
//   | workgroup_ids[keyval_size]      |
//   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

// --------------------------------------------------------------------------------------------------------------
// Filling histograms and keys with default values (also resets the pass infos for odd and even scattering)
// --------------------------------------------------------------------------------------------------------------
@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = 0u;
        infos.odd_pass = 1u;    // has to be one, as on the first call to even pass + 1 % 2 is calculated
    }
    // here the histograms are set to zero and the partitions are set to 0xfffffffff to avoid sorting problems
    let scatter_wg_size = histogram_wg_size;
    let scatter_block_kvs = scatter_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (infos.num_keys + scatter_block_kvs - 1u) / scatter_block_kvs;

    let histo_size = rs_radix_size;
    var n = (rs_keyval_size + scatter_blocks_ru - 1u) * histo_size;
    let b = n;
    if infos.num_keys < infos.padded_size {
        n += infos.padded_size - infos.num_keys;
    }

    let line_size = nwg.x * {histogram_wg_size}u;
    for (var cur_index = gid.x; cur_index < n; cur_index += line_size){
        if cur_index >= n {
            return;
        }
            
        if cur_index  < rs_keyval_size * histo_size {
            atomicStore(&histograms[cur_index], 0u);
        }
        else if cur_index < b {
            atomicStore(&histograms[cur_index], 0u);
        }
        else {
            keys[infos.num_keys + cur_index - b] = 0xFFFFFFFFu;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------
// Calculating the histograms
// --------------------------------------------------------------------------------------------------------------
var<workgroup> smem : array<atomic<u32>, rs_radix_size>;
var<private> kv : array<u32, rs_histogram_block_rows>;
fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        atomicStore(&smem[lid], 0u);
    }
}
fn histogram_pass(pass_: u32, lid: u32) {
    zero_smem(lid);
    workgroupBarrier();

    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = bitcast<u32>(kv[j]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }

    workgroupBarrier();
    let histogram_offset = rs_radix_size * pass_ + lid;
    if lid < rs_radix_size && atomicLoad(&smem[lid]) >= 0u {
        atomicAdd(&histograms[histogram_offset], atomicLoad(&smem[lid]));
    }
}

// the workgrpu_size can be gotten on the cpu by by calling pipeline.get_bind_group_layout(0).unwrap().get_local_workgroup_size();
fn fill_kv(wid: u32, lid: u32) {
    let rs_block_keyvals: u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys[pos];
    }
}
fn fill_kv_keys_b(wid: u32, lid: u32) {
    let rs_block_keyvals: u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys_b[pos];
    }
}
@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // efficient loading of multiple values
    fill_kv(wid.x, lid.x);
    
    // Accumulate and store histograms for passes
    histogram_pass(3u, lid.x);
    histogram_pass(2u, lid.x);
    histogram_pass(1u, lid.x);
    histogram_pass(0u, lid.x);
}

// --------------------------------------------------------------------------------------------------------------
// Prefix sum over histogram
// --------------------------------------------------------------------------------------------------------------
fn prefix_reduce_smem(lid: u32) {
    var offset = 1u;
    for (var d = rs_radix_size >> 1u; d > 0u; d = d >> 1u) { // sum in place tree
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            atomicAdd(&smem[bi], atomicLoad(&smem[ai]));
        }
        offset = offset << 1u;
    }

    if lid == 0u {
        atomicStore(&smem[rs_radix_size - 1u], 0u);
    } // clear the last element

    for (var d = 1u; d < rs_radix_size; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;

            let t = atomicLoad(&smem[ai]);
            atomicStore(&smem[ai], atomicLoad(&smem[bi]));
            atomicAdd(&smem[bi], t);
        }
    }
}
@compute @workgroup_size({prefix_wg_size})
fn prefix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // the work group  id is the pass, and is inverted in the next line, such that pass 3 is at the first position in the histogram buffer
    let histogram_base = (rs_keyval_size - 1u - wid.x) * rs_radix_size;
    let histogram_offset = histogram_base + lid.x;
    
    // the following coode now corresponds to the prefix calc code in fuchsia/../shaders/prefix.h
    // however the implementation is taken from https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf listing 2 (better overview, nw subgroup arithmetic)
    // this also means that only half the amount of workgroups is spawned (one workgroup calculates for 2 positioons)
    // the smemory is used from the previous section
    atomicStore(&smem[lid.x], atomicLoad(&histograms[histogram_offset]));
    atomicStore(&smem[lid.x + {prefix_wg_size}u], atomicLoad(&histograms[histogram_offset + {prefix_wg_size}u]));

    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    atomicStore(&histograms[histogram_offset], atomicLoad(&smem[lid.x]));
    atomicStore(&histograms[histogram_offset + {prefix_wg_size}u], atomicLoad(&smem[lid.x + {prefix_wg_size}u]));
}

// --------------------------------------------------------------------------------------------------------------
// Scattering the keys
// --------------------------------------------------------------------------------------------------------------
// General note: Only 2 sweeps needed here
var<workgroup> scatter_smem: array<u32, rs_mem_dwords>; // note: rs_mem_dwords is caclulated in the beginngin of gpu_rs.rs
//            | Dwords                                    | Bytes
//  ----------+-------------------------------------------+--------
//  Lookback  | 256                                       | 1 KB
//  Histogram | 256                                       | 1 KB
//  Prefix    | 4-84                                      | 16-336
//  Reorder   | RS_WORKGROUP_SIZE * RS_SCATTER_BLOCK_ROWS | 2-8 KB
fn partitions_base_offset() -> u32 { return rs_keyval_size * rs_radix_size;}
fn smem_prefix_offset() -> u32 { return rs_radix_size + rs_radix_size;}
fn rs_prefix_sweep_0(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_0_offset + idx];}
fn rs_prefix_sweep_1(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_1_offset + idx];}
fn rs_prefix_sweep_2(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_2_offset + idx];}
fn rs_prefix_load(lid: u32, idx: u32) -> u32 { return scatter_smem[rs_radix_size + lid + idx];}
fn rs_prefix_store(lid: u32, idx: u32, val: u32) { scatter_smem[rs_radix_size + lid + idx] = val;}
fn is_first_local_invocation(lid: u32) -> bool { return lid == 0u;}

fn histogram_load(digit: u32) -> u32 {
    return atomicLoad(&smem[digit]);
}

fn histogram_store(digit: u32, count: u32) {
    atomicStore(&smem[digit], count);
}


const rs_partition_mask_status : u32 = 0xC0000000u;
const rs_partition_mask_count : u32 = 0x3FFFFFFFu;
var<private> kr : array<u32, rs_scatter_block_rows>;
var<private> pv : array<u32, rs_scatter_block_rows>;

fn fill_kv_even(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals: u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_a[pos];
    }
}

fn fill_kv_odd(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals: u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_b[pos];
    }
}
fn scatter(pass_: u32, lid: vec3<u32>, gid: vec3<u32>, wid: vec3<u32>, nwg: vec3<u32>, partition_status_invalid: u32, partition_status_reduction: u32, partition_status_prefix: u32) {
    let partition_mask_invalid = partition_status_invalid << 30u;
    let partition_mask_reduction = partition_status_reduction << 30u;
    let partition_mask_prefix = partition_status_prefix << 30u;
    // kv_filling is done in the scatter_even and scatter_odd functions to account for front and backbuffer switch
    // in the reference there is a nulling of the smmem here, was moved to line 251 as smem is used in the code until then

    // The following implements conceptually the same as the
    // Emulate a "match" operation with broadcasts for small subgroup sizes (line 665 ff in scatter.glsl)
    // The difference however is, that instead of using subrgoupBroadcast each thread stores
    // its current number in the smem at lid.x, and then looks up their neighbouring values of the subgroup
    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {scatter_wg_size}u / histogram_sg_size;
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = bitcast<u32>(kv[i]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicStore(&smem[lid.x], digit);
        var count = 0u;
        var rank = 0u;
        
        for (var j = 0u; j < histogram_sg_size; j++) {
            if atomicLoad(&smem[subgroup_offset + j]) == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }
        
        kr[i] = (count << 16u) | rank;
    }
    
    zero_smem(lid.x);   // now zeroing the smmem as we are now accumulating the histogram there
    workgroupBarrier();

    // The final histogram is stored in the smem buffer
    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = bitcast<u32>(kv[j]);
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;

                if rank == count {
                    histogram_store(digit, (prev + count));
                }
                
                // TODO: check if the barrier here is needed
            }            
        }
        workgroupBarrier();
    }
    // kr filling is now done and contains the total offset for each value to be able to 
    // move the values into order without having any collisions
    
    // we do not check for single work groups (is currently not assumed to occur very often)
    let partition_offset = lid.x + partitions_base_offset();    // is correct, the partitions pointer does not change
    let partition_base = wid.x * rs_radix_size;
    if wid.x == 0u {
        // special treating for the first workgroup as the data might be read back by later workgroups
        // corresponds to rs_first_prefix_store
        let hist_offset = pass_ * rs_radix_size + lid.x;
        if lid.x < rs_radix_size {
            // let exc = histograms[hist_offset];
            let exc = atomicLoad(&histograms[hist_offset]);
            let red = histogram_load(lid.x);// scatter_smem[rs_keyval_size + lid.x];
            
            scatter_smem[lid.x] = exc;
            
            let inc = exc + red;

            atomicStore(&histograms[partition_offset], inc | partition_mask_prefix);
        }
    }
    else {
        // standard case for the "inbetween" workgroups
        
        // rs_reduction_store, only for inbetween workgroups
        if lid.x < rs_radix_size && wid.x < nwg.x - 1u {
            let red = histogram_load(lid.x);
            atomicStore(&histograms[partition_offset + partition_base], red | partition_mask_reduction);
        }
        
        // rs_loopback_store
        if lid.x < rs_radix_size {
            var partition_base_prev = partition_base - rs_radix_size;
            var exc                 = 0u;

            // Note: Each workgroup invocation can proceed independently.
            // Subgroups and workgroups do NOT have to coordinate.
            while true {
                //let prev = atomicLoad(&histograms[partition_offset]);// histograms[partition_offset + partition_base_prev];
                let prev = atomicLoad(&histograms[partition_base_prev + partition_offset]);// histograms[partition_offset + partition_base_prev];
                if (prev & rs_partition_mask_status) == partition_mask_invalid {
                    continue;
                }
                exc += prev & rs_partition_mask_count;
                if (prev & rs_partition_mask_status) != partition_mask_prefix {
                    // continue accumulating reduction
                    partition_base_prev -= rs_radix_size;
                    continue;
                }

                // otherwise save the exclusive scan and atomically transform the
                // reduction into an inclusive prefix status math: reduction + 1 = prefix
                scatter_smem[lid.x] = exc;

                if wid.x < nwg.x - 1u { // only store when inbetween, skip for last workgrup
                    atomicAdd(&histograms[partition_offset + partition_base], exc | (1u << 30u));
                }
                break;
            }
        }
    }
    // special case for last workgroup is also done in the "inbetween" case
    
    // compute exclusive prefix scan of histogram
    // corresponds to rs_prefix
    // TODO make sure that the data is put into smem
    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    // convert keyval rank to local index, corresponds to rs_rank_to_local
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc   = histogram_load(digit);
        let idx   = exc + kr[i];
        
        kr[i] |= (idx << 16u);
    }
    workgroupBarrier();
    
    // reorder kv[] and kr[], corresponds to rs_reorder
    let smem_reorder_offset = rs_radix_size;
    let smem_base = smem_reorder_offset + lid.x;  // as we are in smem, the radix_size offset is not needed

    // keyvalues ----------------------------------------------
    // store keyval to sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        
        scatter_smem[smem_idx] = bitcast<u32>(kv[j]);
    }
    workgroupBarrier();

    // Load keyval dword from sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        kv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();
    // payload ----------------------------------------------
    // store payload to sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        
        scatter_smem[smem_idx] = pv[j];
    }
    workgroupBarrier();

    // Load payload dword from sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        pv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();
    
    // store the digit-index to sorted location
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let smem_idx = smem_reorder_offset + (kr[i] >> 16u) - 1u;
        scatter_smem[smem_idx] = kr[i];
    }
    workgroupBarrier();

    // Load kr[] from sorted location -- we only need the rank
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        kr[i] = scatter_smem[smem_base + i * {scatter_wg_size}u] & 0xFFFFu;
    }
    
    // convert local index to a global index, corresponds to rs_local_to_global
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc   = scatter_smem[digit];

        kr[i] += exc - 1u;
    }
    
    // the storing is done in the scatter_even and scatter_odd functions as the front and back buffer changes
}

@compute @workgroup_size({scatter_wg_size})
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.odd_pass = (infos.odd_pass + 1u) % 2u; // for this to work correctly the odd_pass has to start 1
    }
    let cur_pass = infos.even_pass * 2u;
    
    // load from keys, store to keys_b
    fill_kv_even(wid.x, lid.x);

    let partition_status_invalid = 0u;
    let partition_status_reduction = 1u;
    let partition_status_prefix = 2u;
    scatter(cur_pass, lid, gid, wid, nwg, partition_status_invalid, partition_status_reduction, partition_status_prefix);

    // store keyvals to their new locations, corresponds to rs_store
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_b[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_b[kr[i]] = pv[i];
    }
}
@compute @workgroup_size({scatter_wg_size})
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = (infos.even_pass + 1u) % 2u; // for this to work correctly the even_pass has to start at 0
    }
    let cur_pass = infos.odd_pass * 2u + 1u;

    // load from keys_b, store to keys
    fill_kv_odd(wid.x, lid.x);

    let partition_status_invalid = 2u;
    let partition_status_reduction = 3u;
    let partition_status_prefix = 0u;
    scatter(cur_pass, lid, gid, wid, nwg, partition_status_invalid, partition_status_reduction, partition_status_prefix);

    // store keyvals to their new locations, corresponds to rs_store
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_a[kr[i]] = pv[i];
    }

    // the indirect buffer is reset after scattering via write buffer, see record_scatter_indirect for details
}
`;
class we {
  constructor() {
    s(this, "mousemove");
    s(this, "press");
    s(this, "wheel");
    this.mousemove = null, this.press = null, this.wheel = null;
  }
  registerForCanvas(e) {
    let t = null;
    const a = this;
    e.addEventListener("mousemove", function(o) {
      o.preventDefault();
      const r = e.getBoundingClientRect(), i = [o.clientX - r.left, o.clientY - r.top];
      t ? a.mousemove && a.mousemove(t, i, o) : t = [o.clientX - r.left, o.clientY - r.top], t = i;
    }), e.addEventListener("mousedown", function(o) {
      o.preventDefault();
      const r = e.getBoundingClientRect(), i = [o.clientX - r.left, o.clientY - r.top];
      a.press && a.press(i, o);
    }), e.addEventListener("wheel", function(o) {
      o.preventDefault(), a.wheel && a.wheel(-o.deltaY);
    }), e.oncontextmenu = function(o) {
      o.preventDefault();
    };
  }
}
const M = 256, Be = 128, V = 256, Y = 8, A = 1 << Y, T = 32 / Y, I = 15, N = I, $ = M * N, K = M * I, R = 4, ke = R;
class Pe {
  constructor(e, t) {
    s(this, "zeroPipeline");
    s(this, "histogramPipeline");
    s(this, "prefixPipeline");
    s(this, "scatterEvenPipeline");
    s(this, "scatterOddPipeline");
    s(this, "device");
    s(this, "bindGroupLayout");
    this.device = e;
    let a = t, o = Math.floor(A / a), r = Math.floor(o / a), n = A + N * V, _ = 0, d = _ + o, u = d + r;
    console.log(u), this.bindGroupLayout = this.device.createBindGroupLayout({
      label: "radix sort bind group layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        }
      ]
    });
    const g = this.device.createPipelineLayout({
      label: "radix sort pipeline layout",
      bindGroupLayouts: [this.bindGroupLayout]
    });
    let f = `
        const histogram_sg_size: u32 = ${a}u;
        const histogram_wg_size: u32 = ${M}u;
        const rs_radix_log2: u32 = ${Y}u;
        const rs_radix_size: u32 = ${A}u;
        const rs_keyval_size: u32 = ${T}u;
        const rs_histogram_block_rows: u32 = ${I}u;
        const rs_scatter_block_rows: u32 = ${N}u;
        const rs_mem_dwords: u32 = ${n}u;
        const rs_mem_sweep_0_offset: u32 = ${_}u;
        const rs_mem_sweep_1_offset: u32 = ${d}u;
        const rs_mem_sweep_2_offset: u32 = ${u}u;
        ${ye}
        `;
    f = f.replace(/{histogram_wg_size}/g, M.toString()).replace(/{prefix_wg_size}/g, Be.toString()).replace(/{scatter_wg_size}/g, V.toString());
    const c = this.device.createShaderModule({
      label: "Radix sort shader",
      code: f
    });
    this.zeroPipeline = this.device.createComputePipeline({
      label: "zero_histograms",
      layout: g,
      compute: {
        module: c,
        entryPoint: "zero_histograms"
      }
    }), this.histogramPipeline = this.device.createComputePipeline({
      label: "calculate_histogram",
      layout: g,
      compute: {
        module: c,
        entryPoint: "calculate_histogram"
      }
    }), this.prefixPipeline = this.device.createComputePipeline({
      label: "prefix_histogram",
      layout: g,
      compute: {
        module: c,
        entryPoint: "prefix_histogram"
      }
    }), this.scatterEvenPipeline = this.device.createComputePipeline({
      label: "scatter_even",
      layout: g,
      compute: {
        module: c,
        entryPoint: "scatter_even"
      }
    }), this.scatterOddPipeline = this.device.createComputePipeline({
      label: "scatter_odd",
      layout: g,
      compute: {
        module: c,
        entryPoint: "scatter_odd"
      }
    });
  }
  createKeyvalBuffers(e) {
    let t = this.keysBufferSize(e) * T;
    console.log(this.keysBufferSize(e)), console.log(t);
    let a = this.device.createBuffer({
      label: "radix sort keys buffer",
      size: t * R,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    }), o = this.device.createBuffer({
      label: "radix sort keys auxiliary buffer",
      size: t * R,
      usage: GPUBufferUsage.STORAGE
    }), r = e * R, i = this.device.createBuffer({
      label: "radix sort payload buffer",
      size: r,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    }), n = this.device.createBuffer({
      label: "radix sort payload auxiliary buffer",
      size: r,
      usage: GPUBufferUsage.STORAGE
    });
    return [a, o, i, n];
  }
  // calculates and allocates a buffer that is sufficient for holding all needed information for
  // sorting. This includes the histograms and the temporary scatter buffer
  // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
  createInternalMemBuffer(e) {
    let t = this.scatterBlocksRu(e), a = A * 4, o = (T + t) * a;
    return this.device.createBuffer({
      label: "Internal radix sort buffer",
      size: o,
      usage: GPUBufferUsage.STORAGE
    });
  }
  createSortBuffers(e) {
    const [t, a, o, r] = this.createKeyvalBuffers(e), i = this.createInternalMemBuffer(e);
    let n = {
      num_keys: e,
      padded_size: this.keysBufferSize(e),
      even_pass: 0,
      odd_pass: 0
    }, _ = this.device.createBuffer({
      label: "radix sort uniform buffer",
      size: 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const d = new Uint32Array([
      n.num_keys,
      n.padded_size,
      n.even_pass,
      n.odd_pass
    ]);
    this.device.queue.writeBuffer(_, 0, d);
    const u = this.device.createBindGroup({
      label: "radix sort bind group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: _ } },
        { binding: 1, resource: { buffer: i } },
        { binding: 2, resource: { buffer: t } },
        { binding: 3, resource: { buffer: a } },
        { binding: 4, resource: { buffer: o } },
        { binding: 5, resource: { buffer: r } }
      ]
    });
    return new ze(t, a, o, r, i, _, u, e);
  }
  recordCalculateHistogram(e, t, a) {
    const o = this.histoBlocksRu(a), r = e.beginComputePass({ label: "zeroing histogram" });
    r.setPipeline(this.zeroPipeline), r.setBindGroup(0, t), r.dispatchWorkgroups(o, 1, 1), r.end();
    const i = e.beginComputePass({ label: "calculate histogram" });
    i.setPipeline(this.histogramPipeline), i.setBindGroup(0, t), i.dispatchWorkgroups(o, 1, 1), i.end();
  }
  // There does not exist an indirect histogram dispatch as the number of prefixes is determined by the amount of passes
  recordPrefixHistogram(e, t) {
    const a = e.beginComputePass({ label: "prefix histogram" });
    a.setPipeline(this.prefixPipeline), a.setBindGroup(0, t), a.dispatchWorkgroups(ke, 1, 1), a.end();
  }
  recordScatterKeys(e, t, a) {
    const o = this.scatterBlocksRu(a), r = e.beginComputePass({ label: "Scatter keyvals" });
    r.setBindGroup(0, t), r.setPipeline(this.scatterEvenPipeline), r.dispatchWorkgroups(o, 1, 1), r.setPipeline(this.scatterOddPipeline), r.dispatchWorkgroups(o, 1, 1), r.setPipeline(this.scatterEvenPipeline), r.dispatchWorkgroups(o, 1, 1), r.setPipeline(this.scatterOddPipeline), r.dispatchWorkgroups(o, 1, 1), r.end();
  }
  /// Writes sort commands to command encoder.
  /// If sort_first_n is not none one the first n elements are sorted
  /// otherwise everything is sorted.
  ///
  /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will be be corrupted
  sort(e, t, a, o) {
    const r = o ?? a.length;
    t.writeBuffer(a.uniformBuffer, 0, new Uint32Array([r])), this.recordCalculateHistogram(e, a.bindGroup, r), this.recordPrefixHistogram(e, a.bindGroup), this.recordScatterKeys(e, a.bindGroup, r);
  }
  scatterBlocksRu(e) {
    return Math.ceil(e / $);
  }
  histoBlocksRu(e) {
    return Math.ceil(this.scatterBlocksRu(e) * $ / K);
  }
  keysBufferSize(e) {
    return this.histoBlocksRu(e) * K;
  }
}
class ze {
  constructor(e, t, a, o, r, i, n, _) {
    this.keysA = e, this.keysB = t, this.payloadA = a, this.payloadB = o, this.internalMemBuffer = r, this.uniformBuffer = i, this.bindGroup = n, this.length = _;
  }
  get keys() {
    return this.keysA;
  }
  get values() {
    return this.payloadA;
  }
  keysValidSize() {
    return this.length * T;
  }
  destroy() {
    this.keysA.destroy(), this.keysB.destroy(), this.payloadA.destroy(), this.payloadB.destroy(), this.internalMemBuffer.destroy(), this.uniformBuffer.destroy();
  }
}
class Ue {
  constructor(e) {
    s(this, "sorter");
    s(this, "paramsBuffer");
    s(this, "nodeDataBuffer");
    s(this, "edgeDataBuffer");
    s(this, "forceDataBuffer");
    s(this, "coolingFactor", 0.985);
    s(this, "device");
    s(this, "createTreePipeline");
    s(this, "createSourceListPipeline");
    s(this, "createTargetListPipeline");
    s(this, "computeAttractivePipeline");
    s(this, "computeForcesBHPipeline");
    s(this, "applyForcesPipeline");
    s(this, "iterationCount", 1);
    s(this, "mortonCodePipeline");
    s(this, "mortonCodeBuffer");
    s(this, "theta", 0.8);
    s(this, "l", 0.01);
    s(this, "stopForce", !1);
    s(this, "clusterSize");
    s(this, "nodeLength");
    s(this, "edgeLength");
    s(this, "sourceEdgeDataBuffer");
    s(this, "targetEdgeDataBuffer");
    this.device = e, this.sorter = new Pe(this.device, 32), this.clusterSize = 4, this.nodeLength = 0, this.edgeLength = 0, this.nodeDataBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }), this.mortonCodeBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }), this.edgeDataBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }), this.sourceEdgeDataBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }), this.targetEdgeDataBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }), this.forceDataBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), this.createTreePipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: me.replace(/CHANGEME/g, this.clusterSize.toString())
        }),
        entryPoint: "main"
      }
    }), this.mortonCodePipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: ve.replace(/CHANGEME/g, this.clusterSize.toString())
        }),
        entryPoint: "main"
      }
    }), this.createSourceListPipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: _e
        }),
        entryPoint: "main"
      }
    }), this.createTargetListPipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: ge
        }),
        entryPoint: "main"
      }
    }), this.computeAttractivePipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: ce
        }),
        entryPoint: "main"
      }
    }), this.computeForcesBHPipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: le.replace(/CHANGEME/g, this.clusterSize.toString())
        }),
        entryPoint: "main"
      }
    }), this.applyForcesPipeline = e.createComputePipeline({
      layout: "auto",
      compute: {
        module: e.createShaderModule({
          code: fe
        }),
        entryPoint: "main"
      }
    }), this.paramsBuffer = e.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  }
  stopForces() {
    this.stopForce = !0;
  }
  formatToD3Format(e, t, a, o) {
    const r = new Array(a), i = new Array(o / 2);
    for (let n = 0; n < 4 * a; n = n + 4)
      r[n / 4] = {
        index: n / 4,
        name: (n / 4).toString(),
        x: e[n + 1],
        y: e[n + 2]
      };
    for (let n = 0; n < o; n = n + 2) {
      let _ = t[n], d = t[n + 1];
      i[n / 2] = {}, i[n / 2].index = n / 2, i[n / 2].source = {}, i[n / 2].source.index = _, i[n / 2].source.name = _.toString(), i[n / 2].source.x = r[_].x, i[n / 2].source.y = r[_].y, i[n / 2].target = {}, i[n / 2].target.index = d, i[n / 2].target.name = d.toString(), i[n / 2].target.x = r[d].x, i[n / 2].target.y = r[d].y;
    }
    return { nodeArray: r, edgeArray: i };
  }
  setNodeEdgeData(e, t) {
    this.nodeLength = e.length / 4, this.edgeLength = t.length, this.nodeDataBuffer = this.device.createBuffer({
      size: e.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    }), new Float32Array(this.nodeDataBuffer.getMappedRange()).set(e), this.nodeDataBuffer.unmap(), this.mortonCodeBuffer = this.device.createBuffer({
      size: e.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), this.edgeDataBuffer = this.device.createBuffer({
      size: t.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.edgeDataBuffer.getMappedRange()).set(t), this.edgeDataBuffer.unmap();
    const a = [];
    for (let i = 0; i < t.length; i += 2)
      a.push({ source: t[i], target: t[i + 1] });
    const o = a.sort((i, n) => i.source - n.source).flatMap((i) => [i.source, i.target]), r = a.slice().sort((i, n) => i.target - n.target).flatMap((i) => [i.source, i.target]);
    console.log(o), console.log(r), this.sourceEdgeDataBuffer = this.device.createBuffer({
      size: o.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.sourceEdgeDataBuffer.getMappedRange()).set(o), this.sourceEdgeDataBuffer.unmap(), this.targetEdgeDataBuffer = this.device.createBuffer({
      size: r.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.targetEdgeDataBuffer.getMappedRange()).set(r), this.targetEdgeDataBuffer.unmap();
  }
  async runForces(e = this.coolingFactor, t = this.l, a = this.theta, o = this.iterationCount) {
    if (this.stopForce = !1, this.nodeLength === 0 || this.edgeLength === 0 || this.nodeDataBuffer === null || this.edgeDataBuffer === null) {
      console.log("No data to run");
      return;
    }
    this.l = t, this.theta = a, this.coolingFactor = e, this.iterationCount = o;
    const r = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    }), i = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    });
    let n = i.getMappedRange();
    new Int32Array(n).set([0, 1e3, 0, 1e3]), i.unmap();
    const _ = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    });
    let d = _.getMappedRange();
    new Int32Array(d).set([1e3, -1e3, 1e3, -1e3]), _.unmap();
    let u = this.device.createCommandEncoder();
    u.copyBufferToBuffer(i, 0, r, 0, 4 * 4), this.device.queue.submit([u.finish()]);
    const g = this.sorter.createSortBuffers(this.nodeLength), f = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    });
    n = f.getMappedRange(), new Uint32Array(n).set([this.nodeLength, this.edgeLength]), new Float32Array(n).set([this.coolingFactor, t], 2), f.unmap(), u = this.device.createCommandEncoder(), u.copyBufferToBuffer(f, 0, this.paramsBuffer, 0, 4 * 4), this.device.queue.submit([u.finish()]);
    const c = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(
      c,
      8,
      new Float32Array([this.theta]),
      0,
      1
    ), this.forceDataBuffer = this.device.createBuffer({
      size: this.nodeLength * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const y = this.device.createBuffer({
      size: this.edgeLength * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), v = this.device.createBuffer({
      size: this.edgeLength * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), p = this.device.createBuffer({
      size: this.nodeLength * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), k = this.device.createBuffer({
      size: Math.ceil(this.nodeLength * 2.1) * (12 + Math.max(4, this.clusterSize)) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), z = this.device.createBindGroup({
      layout: this.createSourceListPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.sourceEdgeDataBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: p
          }
        },
        {
          binding: 2,
          resource: {
            buffer: y
          }
        },
        {
          binding: 3,
          resource: {
            buffer: this.paramsBuffer
          }
        }
      ]
    }), w = this.device.createBindGroup({
      layout: this.createTargetListPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.targetEdgeDataBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: p
          }
        },
        {
          binding: 2,
          resource: {
            buffer: v
          }
        },
        {
          binding: 3,
          resource: {
            buffer: this.paramsBuffer
          }
        }
      ]
    });
    this.device.queue.submit([u.finish()]), u = this.device.createCommandEncoder();
    const B = u.beginComputePass();
    B.setBindGroup(0, z), B.setPipeline(this.createSourceListPipeline), B.dispatchWorkgroups(1, 1, 1), B.setBindGroup(0, w), B.setPipeline(this.createTargetListPipeline), B.dispatchWorkgroups(1, 1, 1), B.end();
    const Z = this.device.createBuffer({
      size: this.nodeLength * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    u.copyBufferToBuffer(
      p,
      0,
      Z,
      0,
      this.nodeLength * 4 * 4
      /* size */
    ), this.device.queue.submit([u.finish()]);
    const J = this.device.createBindGroup({
      layout: this.applyForcesPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.forceDataBuffer
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
            buffer: r
          }
        }
      ]
    }), Q = this.device.createBindGroup({
      layout: this.createTreePipeline.getBindGroupLayout(0),
      entries: [
        // Sort values buffer filled with indices
        {
          binding: 0,
          resource: {
            buffer: g.values
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.paramsBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: c
          }
        },
        {
          binding: 3,
          resource: {
            buffer: r
          }
        },
        {
          binding: 4,
          resource: {
            buffer: k
          }
        }
      ]
    }), ee = this.device.createBindGroup({
      layout: this.mortonCodePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.mortonCodeBuffer
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
            buffer: r
          }
        },
        // Sort values buffer filled with mor
        {
          binding: 4,
          resource: {
            buffer: g.values
          }
        },
        {
          binding: 5,
          resource: {
            buffer: k
          }
        }
      ]
    });
    let te = this.device.createBuffer({
      size: this.nodeLength * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    }), P = 0;
    for (var b, x, O = 0; P < o && this.coolingFactor > 1e-4; ) {
      P == 1 && (O = performance.now());
      const re = performance.now();
      P++;
      const F = this.device.createBuffer({
        size: 4 * 4,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: !0
      }), H = F.getMappedRange();
      new Uint32Array(H).set([this.nodeLength, this.edgeLength]), new Float32Array(H).set([this.coolingFactor, t], 2), F.unmap();
      let l = this.device.createCommandEncoder();
      l.copyBufferToBuffer(F, 0, this.paramsBuffer, 0, 4 * 4), this.device.queue.submit([l.finish()]), b = performance.now(), l = this.device.createCommandEncoder();
      let m = l.beginComputePass();
      m.setBindGroup(0, ee), m.setPipeline(this.mortonCodePipeline), m.dispatchWorkgroups(Math.ceil(this.nodeLength / 128), 1, 1), m.end(), l.copyBufferToBuffer(this.mortonCodeBuffer, 0, g.keys, 0, this.mortonCodeBuffer.size), this.device.queue.submit([l.finish()]), x = performance.now(), console.log(`Morton codes took ${x - b}ms`);
      {
        var G = this.device.createBuffer({
          size: this.mortonCodeBuffer.size,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        l = this.device.createCommandEncoder(), l.copyBufferToBuffer(this.mortonCodeBuffer, 0, G, 0, G.size), this.device.queue.submit([l.finish()]), await this.device.queue.onSubmittedWorkDone(), await G.mapAsync(GPUMapMode.READ);
        var D = new Uint32Array(G.getMappedRange());
        console.log(D);
      }
      b = performance.now();
      const W = this.device.createCommandEncoder();
      this.sorter.sort(W, this.device.queue, g), this.device.queue.submit([W.finish()]), x = performance.now(), console.log(`Sort took ${x - b} ms`);
      {
        var S = this.device.createBuffer({
          size: g.keys.size,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        l = this.device.createCommandEncoder(), l.copyBufferToBuffer(g.keys, 0, S, 0, S.size), this.device.queue.submit([l.finish()]), await this.device.queue.onSubmittedWorkDone(), await S.mapAsync(GPUMapMode.READ);
        var D = new Uint32Array(S.getMappedRange());
        console.log(D);
      }
      let ie = performance.now();
      var j = this.nodeLength;
      l = this.device.createCommandEncoder();
      for (var U = 0; U < Math.log(this.nodeLength) / Math.log(this.clusterSize); U++)
        this.device.queue.writeBuffer(
          c,
          0,
          new Uint32Array([U]),
          0,
          1
        ), l = this.device.createCommandEncoder(), m = l.beginComputePass(), m.setBindGroup(0, Q), m.setPipeline(this.createTreePipeline), m.dispatchWorkgroups(Math.ceil(this.nodeLength / (128 * this.clusterSize ** (U + 1))), 1, 1), m.end(), this.device.queue.submit([l.finish()]), j += Math.ceil(this.nodeLength / this.clusterSize ** (U + 1));
      this.device.queue.writeBuffer(
        c,
        4,
        new Uint32Array([j]),
        0,
        1
      ), this.device.queue.submit([l.finish()]);
      let oe = performance.now();
      console.log(`Create Tree took ${oe - ie}ms`), l = this.device.createCommandEncoder();
      const se = this.device.createBindGroup({
        layout: this.computeForcesBHPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.nodeDataBuffer
            }
          },
          {
            binding: 1,
            resource: {
              buffer: this.forceDataBuffer
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
              buffer: c
            }
          },
          {
            binding: 4,
            resource: {
              buffer: k
            }
          }
        ]
      }), ae = this.device.createBindGroup({
        layout: this.computeAttractivePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: p
            }
          },
          {
            binding: 1,
            resource: {
              buffer: y
            }
          },
          {
            binding: 2,
            resource: {
              buffer: v
            }
          },
          {
            binding: 3,
            resource: {
              buffer: this.forceDataBuffer
            }
          },
          {
            binding: 4,
            resource: {
              buffer: this.nodeDataBuffer
            }
          },
          {
            binding: 5,
            resource: {
              buffer: this.paramsBuffer
            }
          }
        ]
      });
      m = l.beginComputePass(), m.setBindGroup(0, ae), m.setPipeline(this.computeAttractivePipeline), m.dispatchWorkgroups(Math.ceil(this.nodeLength / 128), 1, 1), m.end(), this.device.queue.submit([l.finish()]), b = performance.now(), x = performance.now(), console.log(`attract force time: ${x - b}`), b = performance.now(), l = this.device.createCommandEncoder();
      const E = l.beginComputePass();
      E.setBindGroup(0, se), E.setPipeline(this.computeForcesBHPipeline), E.dispatchWorkgroups(Math.ceil(this.nodeLength / 128), 1, 1), E.end(), this.device.queue.submit([l.finish()]), x = performance.now(), console.log(`repulse force time: ${x - b}`), l = this.device.createCommandEncoder(), l.copyBufferToBuffer(_, 0, r, 0, 4 * 4), b = performance.now(), m = l.beginComputePass(), m.setBindGroup(0, J), m.setPipeline(this.applyForcesPipeline), m.dispatchWorkgroups(Math.ceil(this.nodeLength / (2 * 128)), 1, 1), m.end(), this.device.queue.submit([l.finish()]), x = performance.now(), console.log(`apply forces time ${x - b}`), this.coolingFactor = this.coolingFactor * 0.975;
      const ne = performance.now();
      console.log(`Total frame time: ${ne - re}`), P % 10 == 0 && await this.device.queue.onSubmittedWorkDone();
    }
    await te.mapAsync(GPUMapMode.READ), await this.device.queue.onSubmittedWorkDone();
    const q = performance.now();
    console.log(`Completed in ${P} iterations with total time ${q - O} average iteration time ${(q - O) / (P - 1)}`);
  }
}
function L(h, e, t = GPUBufferUsage.STORAGE) {
  const a = {
    size: Math.max(Math.ceil(e.byteLength / 4) * 4, 16),
    usage: t,
    mappedAtCreation: !0
  }, o = h.createBuffer(a), r = o.getMappedRange();
  return (e instanceof Uint32Array ? new Uint32Array(r) : new Float32Array(r)).set(e), o.unmap(), o;
}
var C = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {}, X = { exports: {} };
(function(h, e) {
  (function(t, a) {
    a();
  })(C, function() {
    function t(d, u) {
      return typeof u > "u" ? u = { autoBom: !1 } : typeof u != "object" && (console.warn("Deprecated: Expected third argument to be a object"), u = { autoBom: !u }), u.autoBom && /^\s*(?:text\/\S*|application\/xml|\S*\/\S*\+xml)\s*;.*charset\s*=\s*utf-8/i.test(d.type) ? new Blob(["\uFEFF", d], { type: d.type }) : d;
    }
    function a(d, u, g) {
      var f = new XMLHttpRequest();
      f.open("GET", d), f.responseType = "blob", f.onload = function() {
        _(f.response, u, g);
      }, f.onerror = function() {
        console.error("could not download file");
      }, f.send();
    }
    function o(d) {
      var u = new XMLHttpRequest();
      u.open("HEAD", d, !1);
      try {
        u.send();
      } catch {
      }
      return 200 <= u.status && 299 >= u.status;
    }
    function r(d) {
      try {
        d.dispatchEvent(new MouseEvent("click"));
      } catch {
        var u = document.createEvent("MouseEvents");
        u.initMouseEvent("click", !0, !0, window, 0, 0, 0, 80, 20, !1, !1, !1, !1, 0, null), d.dispatchEvent(u);
      }
    }
    var i = typeof window == "object" && window.window === window ? window : typeof self == "object" && self.self === self ? self : typeof C == "object" && C.global === C ? C : void 0, n = i.navigator && /Macintosh/.test(navigator.userAgent) && /AppleWebKit/.test(navigator.userAgent) && !/Safari/.test(navigator.userAgent), _ = i.saveAs || (typeof window != "object" || window !== i ? function() {
    } : "download" in HTMLAnchorElement.prototype && !n ? function(d, u, g) {
      var f = i.URL || i.webkitURL, c = document.createElement("a");
      u = u || d.name || "download", c.download = u, c.rel = "noopener", typeof d == "string" ? (c.href = d, c.origin === location.origin ? r(c) : o(c.href) ? a(d, u, g) : r(c, c.target = "_blank")) : (c.href = f.createObjectURL(d), setTimeout(function() {
        f.revokeObjectURL(c.href);
      }, 4e4), setTimeout(function() {
        r(c);
      }, 0));
    } : "msSaveOrOpenBlob" in navigator ? function(d, u, g) {
      if (u = u || d.name || "download", typeof d != "string")
        navigator.msSaveOrOpenBlob(t(d, g), u);
      else if (o(d))
        a(d, u, g);
      else {
        var f = document.createElement("a");
        f.href = d, f.target = "_blank", setTimeout(function() {
          r(f);
        });
      }
    } : function(d, u, g, f) {
      if (f = f || open("", "_blank"), f && (f.document.title = f.document.body.innerText = "downloading..."), typeof d == "string")
        return a(d, u, g);
      var c = d.type === "application/octet-stream", y = /constructor/i.test(i.HTMLElement) || i.safari, v = /CriOS\/[\d]+/.test(navigator.userAgent);
      if ((v || c && y || n) && typeof FileReader < "u") {
        var p = new FileReader();
        p.onloadend = function() {
          var w = p.result;
          w = v ? w : w.replace(/^data:[^;]*;/, "data:attachment/file;"), f ? f.location.href = w : location = w, f = null;
        }, p.readAsDataURL(d);
      } else {
        var k = i.URL || i.webkitURL, z = k.createObjectURL(d);
        f ? f.location = z : location.href = z, f = null, setTimeout(function() {
          k.revokeObjectURL(z);
        }, 4e4);
      }
    });
    i.saveAs = _.saveAs = _, h.exports = _;
  });
})(X);
var Ce = X.exports;
class Se {
  constructor(e, t, a) {
    s(this, "device");
    s(this, "forceDirected", null);
    s(this, "nodeBindGroup", null);
    s(this, "edgeBindGroup", null);
    s(this, "uniform2DBuffer", null);
    s(this, "nodeDataBuffer", null);
    s(this, "edgeDataBuffer", null);
    s(this, "sourceEdgeDataBuffer", null);
    s(this, "targetEdgeDataBuffer", null);
    s(this, "viewBoxBuffer", null);
    s(this, "nodePipeline", null);
    s(this, "edgePipeline", null);
    s(this, "nodeLength", 1);
    s(this, "edgeLength", 1);
    s(this, "nodeToggle", !0);
    s(this, "edgeToggle", !0);
    s(this, "canvasSize", null);
    s(this, "idealLength", 5e-3);
    s(this, "coolingFactor", 0.985);
    s(this, "iterRef");
    s(this, "frame");
    s(this, "edgeList", []);
    s(this, "mortonCodeBuffer", null);
    s(this, "energy", 0.1);
    s(this, "theta", 2);
    s(this, "canvasRef");
    s(this, "viewExtreme");
    s(this, "iterationCount", 1e3);
    s(this, "context", null);
    s(this, "edgePositionBuffer", null);
    s(this, "nodePositionBuffer", null);
    if (this.iterRef = a, this.device = e, this.canvasRef = t, this.viewExtreme = [-1, -1, 2, 2], t.current === null)
      return;
    this.context = t.current.getContext("webgpu");
    const o = window.devicePixelRatio || 1;
    t.current.width = 800 * o, t.current.height = 800 * o;
    const r = "rgba8unorm";
    this.canvasSize = [
      t.current.width,
      t.current.height
    ], this.context.configure({
      device: e,
      format: r,
      alphaMode: "opaque"
    }), this.edgeDataBuffer = e.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: !0
    });
    let i = [0, 0, 0.01, 0.01];
    new Float32Array(this.edgeDataBuffer.getMappedRange()).set(i), this.edgeDataBuffer.unmap(), this.edgePipeline = e.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: e.createShaderModule({
          code: he
        }),
        entryPoint: "main",
        buffers: [
          {
            arrayStride: 2 * 4 * 1,
            attributes: [
              {
                format: "float32x2",
                offset: 0,
                shaderLocation: 0
              }
            ]
          }
        ]
      },
      fragment: {
        module: e.createShaderModule({
          code: pe
        }),
        entryPoint: "main",
        targets: [
          {
            format: r,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }
            }
          }
        ]
      },
      primitive: {
        topology: "line-list"
        //triangle-list is default   
      },
      multisample: {
        count: 4
      }
    });
    const n = new Float32Array([
      1,
      -1,
      -1,
      -1,
      -1,
      1,
      1,
      -1,
      -1,
      1,
      1,
      1
    ]);
    this.nodePositionBuffer = L(e, n, GPUBufferUsage.VERTEX);
    const _ = new Float32Array([0, 0, 1, 1]);
    this.edgePositionBuffer = L(e, _, GPUBufferUsage.VERTEX);
    const d = new Float32Array([0.5, 0.5, 0.5, 0.5]);
    this.nodeDataBuffer = L(e, d, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), this.mortonCodeBuffer = e.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    });
    let u = [0];
    new Float32Array(this.mortonCodeBuffer.getMappedRange()).set(u), this.mortonCodeBuffer.unmap(), this.nodePipeline = e.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: e.createShaderModule({
          code: xe
        }),
        entryPoint: "main",
        buffers: [
          {
            arrayStride: 2 * 4,
            attributes: [
              {
                format: "float32x2",
                offset: 0,
                shaderLocation: 0
              }
            ]
          }
        ]
      },
      fragment: {
        module: e.createShaderModule({
          code: be
        }),
        entryPoint: "main",
        targets: [
          {
            format: r,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }
            }
          }
        ]
      },
      primitive: {
        topology: "triangle-list"
      },
      multisample: {
        count: 4
      }
    }), this.forceDirected = new Ue(e), this.viewBoxBuffer = e.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    }), this.nodeBindGroup = e.createBindGroup({
      layout: this.nodePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.viewBoxBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: this.mortonCodeBuffer
          }
        }
      ]
    }), this.edgeBindGroup = e.createBindGroup({
      layout: this.edgePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.viewBoxBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: this.edgeDataBuffer
          }
        }
      ]
    });
    const f = e.createTexture({
      size: [t.current.width, t.current.height],
      sampleCount: 4,
      format: r,
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    }).createView(), c = this;
    this.frame = async () => {
      if (!t.current)
        return;
      const y = {
        colorAttachments: [
          {
            view: f,
            resolveTarget: c.context.getCurrentTexture().createView(),
            clearValue: { r: 1, g: 1, b: 1, a: 1 },
            loadOp: "clear",
            storeOp: "discard"
          }
        ]
      }, v = e.createCommandEncoder(), p = v.beginRenderPass(y);
      this.edgeToggle && (p.setPipeline(this.edgePipeline), p.setVertexBuffer(0, c.edgePositionBuffer), p.setBindGroup(0, this.edgeBindGroup), p.draw(2, this.edgeLength / 2, 0, 0)), this.nodeToggle && (p.setPipeline(this.nodePipeline), p.setVertexBuffer(0, c.nodePositionBuffer), p.setBindGroup(0, this.nodeBindGroup), p.draw(6, this.nodeLength, 0, 0)), p.end(), e.queue.submit([v.finish()]);
    }, this.frame();
  }
  async takeScreenshot() {
    if (!this.canvasRef.current)
      return;
    const e = this.canvasRef.current.width, t = this.canvasRef.current.height, a = 4, o = e * t * a, r = this.device.createBuffer({
      size: o,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    }), i = this.device.createTexture({
      size: { width: e, height: t, depthOrArrayLayers: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      sampleCount: 4
    }), n = this.device.createTexture({
      size: { width: e, height: t, depthOrArrayLayers: 1 },
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    }), _ = {
      colorAttachments: [
        {
          view: i.createView(),
          resolveTarget: n.createView(),
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
          loadOp: "clear",
          storeOp: "discard"
        }
      ]
    }, d = this.device.createCommandEncoder(), u = d.beginRenderPass(_);
    this.edgeToggle && (u.setPipeline(this.edgePipeline), u.setVertexBuffer(0, this.edgePositionBuffer), u.setBindGroup(0, this.edgeBindGroup), u.draw(2, this.edgeLength / 2, 0, 0)), this.nodeToggle && (u.setPipeline(this.nodePipeline), u.setVertexBuffer(0, this.nodePositionBuffer), u.setBindGroup(0, this.nodeBindGroup), u.draw(6, this.nodeLength, 0, 0)), u.end(), d.copyTextureToBuffer(
      {
        texture: n,
        mipLevel: 0,
        origin: { x: 0, y: 0, z: 0 }
      },
      {
        buffer: r,
        bytesPerRow: e * a,
        rowsPerImage: t
      },
      {
        width: e,
        height: t,
        depthOrArrayLayers: 1
      }
    ), this.device.queue.submit([d.finish()]), await this.device.queue.onSubmittedWorkDone(), await r.mapAsync(GPUMapMode.READ);
    const g = new Uint8Array(r.getMappedRange()), f = document.createElement("canvas");
    f.width = e, f.height = t;
    const c = f.getContext("2d"), y = c.createImageData(e, t);
    y.data.set(g), c.putImageData(y, 0, 0), f.toBlob(function(v) {
      Ce.saveAs(v, "out.png");
    }, "image/png"), r.unmap(), i.destroy();
  }
  setNodeEdgeData(e, t, a, o) {
    this.edgeList = t, this.nodeDataBuffer = this.device.createBuffer({
      size: e.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: !0
    }), new Float32Array(this.nodeDataBuffer.getMappedRange()).set(e), this.nodeDataBuffer.unmap(), this.mortonCodeBuffer = this.device.createBuffer({
      size: e.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    }), this.edgeDataBuffer = this.device.createBuffer({
      size: t.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.edgeDataBuffer.getMappedRange()).set(t), this.edgeDataBuffer.unmap(), this.edgeBindGroup = this.device.createBindGroup({
      layout: this.edgePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.viewBoxBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: this.edgeDataBuffer
          }
        }
      ]
    }), this.nodeBindGroup = this.device.createBindGroup({
      layout: this.nodePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.viewBoxBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.nodeDataBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: this.mortonCodeBuffer
          }
        }
      ]
    }), this.edgeLength = t.length, this.nodeLength = e.length / 4, this.viewExtreme = [Math.min(-1, -(this.nodeLength / 1e5)), Math.min(-1, -(this.nodeLength / 1e5)), Math.max(2, 2 * (this.nodeLength / 1e5)), Math.max(2, 2 * (this.nodeLength / 1e5))], this.device.queue.writeBuffer(this.viewBoxBuffer, 0, new Float32Array(this.viewExtreme), 0, 4), this.setController(), this.sourceEdgeDataBuffer = this.device.createBuffer({
      size: t.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.sourceEdgeDataBuffer.getMappedRange()).set(a), this.sourceEdgeDataBuffer.unmap(), this.targetEdgeDataBuffer = this.device.createBuffer({
      size: t.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: !0
    }), new Uint32Array(this.targetEdgeDataBuffer.getMappedRange()).set(o), this.targetEdgeDataBuffer.unmap(), requestAnimationFrame(this.frame);
  }
  setCoolingFactor(e) {
    this.coolingFactor = e;
  }
  setIdealLength(e) {
    this.idealLength = e;
  }
  setEnergy(e) {
    this.energy = e;
  }
  setIterationCount(e) {
    this.iterationCount = e;
  }
  setTheta(e) {
    this.theta = e;
  }
  async runForceDirected() {
  }
  async stopForceDirected() {
    this.forceDirected.stopForces();
  }
  toggleNodeLayer() {
    this.nodeToggle = !this.nodeToggle;
  }
  toggleEdgeLayer() {
    this.edgeToggle = !this.edgeToggle;
  }
  setController() {
    let e = this.viewExtreme, t = this.viewExtreme;
    const a = new we();
    a.mousemove = (o, r, i) => {
      if (i.buttons === 1) {
        const n = [(r[0] - o[0]) * (e[2] - e[0]) / this.canvasSize[0], (o[1] - r[1]) * (e[3] - e[1]) / this.canvasSize[1]];
        t = [t[0] - n[0], t[1] - n[1], t[2] - n[0], t[3] - n[1]], (Math.abs(t[0] - e[0]) > 0.03 * (e[2] - e[0]) || Math.abs(t[1] - e[1]) > 0.03 * (e[3] - e[1])) && (e = t, this.device.queue.writeBuffer(this.viewBoxBuffer, 0, new Float32Array(e), 0, 4), requestAnimationFrame(this.frame));
      }
    }, a.wheel = (o) => {
      const r = [o / 1e3, o / 1e3];
      t = [t[0] + r[0], t[1] + r[1], t[2] - r[0], t[3] - r[1]], t[2] - t[0] > 0.01 && t[3] - t[1] > 0.01 ? (e = t, this.device.queue.writeBuffer(this.viewBoxBuffer, 0, new Float32Array(e), 0, 4), requestAnimationFrame(this.frame)) : t = e;
    }, a.registerForCanvas(this.canvasRef.current);
  }
}
export {
  Se as Renderer
};
//# sourceMappingURL=graphwagu-renderer.js.map
