let srcGraphSelectNodes = new Set();
let dstGraphSelectNodes = new Set();
// 记录选中节点
function recSelectNodes(net, selectNodes) {
    net.on("selectNode", (params) => {
        params.nodes.forEach(node => selectNodes.add(node));
    });
    net.on("deselectNode", (params) => {
        params.nodes.forEach(node => selectNodes.delete(node));
    });
    net.on("click", function(params) {
        // 点击空白处
        if (params.nodes.length === 0 && params.edges.length === 0) {
            selectNodes.clear();
        }
    });
}

const srcGraphIframe = document.getElementById('src-graph-iframe');
const dstGraphIframe = document.getElementById('dst-graph-iframe');
// iframe加载完
srcGraphIframe.addEventListener("load", () => {
    const iframeWindow = srcGraphIframe.contentWindow;
    window.srcGraphNetwork = iframeWindow.network;
    srcGraphSelectNodes.clear();
    recSelectNodes(window.srcGraphNetwork, srcGraphSelectNodes);
});
dstGraphIframe.addEventListener("load", () => {
    const iframeWindow = dstGraphIframe.contentWindow;
    window.dstGraphNetwork = iframeWindow.network;
    dstGraphSelectNodes.clear();
    recSelectNodes(window.dstGraphNetwork, dstGraphSelectNodes);
});

// 发送 POST 请求
async function sendPostRequest(url, data) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return response.json();
}

// 刷新 iframe
function reloadIframe(srcOrDst) {
    if (srcOrDst === "src") {
        srcGraphIframe.contentWindow.location.reload();
    } else {
        dstGraphIframe.contentWindow.location.reload();
    }
}

// 设置文件
function setFile(fileId, setFileButtonId, fileName) {
    const setFileButton = document.getElementById(setFileButtonId);
    setFileButton.addEventListener("click", async () => {
        const fileInput = document.getElementById(fileId);
        const formData = new FormData();
        formData.append(fileName, fileInput.files[0]);
        await fetch("/set_file", {
            method: "POST",
            body: formData
        });
        reloadIframe(fileName === "src_file" ? "src" : "dst");
    });
}

// 使用执行序文件中行号设置锚点
function setAnchorLineNum(anchorLineNumId, setAnchorLineNumButtonId, srcOrDst) {
    const setAnchorLineNumButton = document.getElementById(setAnchorLineNumButtonId);
    setAnchorLineNumButton.addEventListener("click", async () => {
        const anchorLineNum = document.getElementById(anchorLineNumId).value;
        await sendPostRequest("/set_anchor_line_num", {
            anchor_line_num: anchorLineNum,
            src_or_dst: srcOrDst
        });
        reloadIframe(srcOrDst);
    });
}

// 使用图中选中节点作为描点
function setAnchor(setAnchorButtonId, selectNodes, srcOrDst) {
    const setAnchorButton = document.getElementById(setAnchorButtonId);
    setAnchorButton.addEventListener("click", async () => {
        if (selectNodes.size !== 1) {
            alert("Please select one node");
            return;
        }
        const selectNode = Array.from(selectNodes)[0];
        await sendPostRequest("/set_anchor", {
            anchor_id: selectNode,
            src_or_dst: srcOrDst
        });
        reloadIframe(srcOrDst);
    });
}

// 切换至整图模式
function changeToWholeGraph() {
    const changeToWholeGraphButton = document.getElementById("change-to-whole-graph-button");
    changeToWholeGraphButton.addEventListener("click", async () => {
        await sendPostRequest("/change_to_whole_graph", {});
        reloadIframe("src");
        reloadIframe("dst");
    });
}

// 融合节点
async function fuseNodes(fuseNodesId, peerNodeId, fusedNodeOperatorType, srcOrDst) {
    const response = await sendPostRequest("/fuse_nodes", {
        fuse_nodes_id: fuseNodesId,
        peer_node_id: peerNodeId,
        fused_node_operator_type: fusedNodeOperatorType,
        src_or_dst: srcOrDst,
    });
    return response.fused_node_id;
}

// 融合一侧节点
function fuseOneSideNodes(fuseNodesButtonId, peerSelectNodes, selectNodes, srcOrDst) {
    const fuseNodesButton = document.getElementById(fuseNodesButtonId);
    fuseNodesButton.addEventListener("click", async () => {
        if (selectNodes.size < 1) {
            alert("Please select at least one node!");
            return;
        }
        if (peerSelectNodes.size !== 1) {
            alert("Please select one peer node!");
            return;
        }
        const fuseNodesId = Array.from(selectNodes);
        const peerNodeId = Array.from(peerSelectNodes)[0];
        await fuseNodes(fuseNodesId, peerNodeId, "", srcOrDst);
        reloadIframe(srcOrDst);
    });
}

// 融合两侧节点
function fuseTwoSideNodes() {
    const fuseNodesButton = document.getElementById("fuse-nodes-button");
    fuseNodesButton.addEventListener("click", async () => {
        if (srcGraphSelectNodes.size < 1 || dstGraphSelectNodes.size < 1) {
            alert("Please select at least one node on both graphs!");
            return;
        }
        const srcFuseNodesId = Array.from(srcGraphSelectNodes);
        const dstFuseNodesId = Array.from(dstGraphSelectNodes);
        const fusedNodeId = await fuseNodes(srcFuseNodesId, dstFuseNodesId[0], "FusedNode", "src");
        await fuseNodes(dstFuseNodesId, fusedNodeId, "FusedNode", "dst");
        reloadIframe("src");
        reloadIframe("dst");
    });
}

// 设置层数
function setLayerNum() {
    const setLayerNumButton = document.getElementById("set-layer-num-button");
    setLayerNumButton.addEventListener("click", async () => {
        const layerNum = document.getElementById("layer-num").value;
        await sendPostRequest("/set_layer_num", {
            layer_num: layerNum
        });
        reloadIframe("src");
        reloadIframe("dst");
    });
}

// 选择匹配的点
function setMatchNodes() {
    const setMatchNodesButton = document.getElementById("set-match-nodes-button");
    setMatchNodesButton.addEventListener("click", async () => {
        if (srcGraphSelectNodes.size !== 1 || dstGraphSelectNodes.size !== 1) {
            alert("Please select one node on both graphs!");
            return;
        }
        const srcNodeId = Array.from(srcGraphSelectNodes)[0];
        const dstNodeId = Array.from(dstGraphSelectNodes)[0];
        await sendPostRequest("/set_match_nodes", {
            src_node_id: srcNodeId,
            dst_node_id: dstNodeId
        });
        reloadIframe("src");
        reloadIframe("dst");
    });
}

// 比较两图
function compareGraphs() {
    const compareButton = document.getElementById("compare-button");
    compareButton.addEventListener("click", async () => {
        await sendPostRequest("/compare_graphs", {});
        reloadIframe("src");
        reloadIframe("dst");
    });
}

// 初始化
document.addEventListener("DOMContentLoaded", () => {
    setFile("src-file", "set-src-file-button", "src_file");
    setFile("dst-file", "set-dst-file-button", "dst_file");

    setAnchorLineNum("src-anchor-line-num", "set-src-anchor-line-num-button", "src");
    setAnchorLineNum("dst-anchor-line-num", "set-dst-anchor-line-num-button", "dst");

    setAnchor("set-src-anchor-button", srcGraphSelectNodes, "src");
    setAnchor("set-dst-anchor-button", dstGraphSelectNodes, "dst");

    changeToWholeGraph();

    fuseOneSideNodes("fuse-src-nodes-button", dstGraphSelectNodes, srcGraphSelectNodes, "src");
    fuseOneSideNodes("fuse-dst-nodes-button", srcGraphSelectNodes, dstGraphSelectNodes, "dst");
    fuseTwoSideNodes();

    setLayerNum();

    setMatchNodes();

    compareGraphs();
});