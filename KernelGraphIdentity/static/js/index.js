const createZoomDisplay = () => {
    const zoomDisplay = document.createElement('div');
    Object.assign(zoomDisplay.style, {
        position: 'fixed',
        bottom: '10px',
        right: '10px',
        backgroundColor: 'rgba(0,0,0,0.7)',
        color: 'white',
        padding: '5px 10px',
        borderRadius: '5px'
    });
    return zoomDisplay;
}

const showSrcNodeInfo = async (params) => {
    const nodeInfo = document.getElementById("src-node-info");
    if (params.nodes.length === 0) {
        nodeInfo.value = "";
        return;
    }
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const response = await sendPostRequest("/get_node_info", {
            node_id: nodeId,
            src_or_dst: "src"
        });
        nodeInfo.value = response.info;
    }
}

const showDstNodeInfo = async (params) => {
    const nodeInfo = document.getElementById("dst-node-info");
    if (params.nodes.length === 0) {
        nodeInfo.value = "";
        return;
    }
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const response = await sendPostRequest("/get_node_info", {
            node_id: nodeId,
            src_or_dst: "dst"
        });
        nodeInfo.value = response.info;
    }
}

const srcGraphIframe = document.getElementById('src-graph-iframe');
const dstGraphIframe = document.getElementById('dst-graph-iframe');
// iframe加载完
srcGraphIframe.addEventListener("load", () => {
    const iframeWindow = srcGraphIframe.contentWindow;
    const graphNetwork = iframeWindow.network;
    window.srcGraphNetwork = graphNetwork;

    // 显示缩放比例
    const zoomDisplay = createZoomDisplay();
    iframeWindow.document.body.appendChild(zoomDisplay);
    const updateZoomDisplay = () => {
        const scale = graphNetwork.getScale() * 100;
        zoomDisplay.textContent = `缩放比例: ${scale.toFixed(2)}%`;
    }
    updateZoomDisplay();
    window.srcGraphNetwork.on("zoom", updateZoomDisplay);

    // 显示节点信息
    window.srcGraphNetwork.on("click", async function(params) {
        await showSrcNodeInfo(params);
    });
});
dstGraphIframe.addEventListener("load", () => {
    const iframeWindow = dstGraphIframe.contentWindow;
    const graphNetwork = iframeWindow.network;
    window.dstGraphNetwork = graphNetwork;

    // 显示缩放比例
    const zoomDisplay = createZoomDisplay();
    iframeWindow.document.body.appendChild(zoomDisplay);
    const updateZoomDisplay = () => {
        const scale = graphNetwork.getScale() * 100;
        zoomDisplay.textContent = `缩放比例: ${scale.toFixed(2)}%`;
    }
    updateZoomDisplay();
    window.dstGraphNetwork.on("zoom", updateZoomDisplay);

    // 显示节点信息
    window.dstGraphNetwork.on("click", async function(params) {
        await showDstNodeInfo(params);
    });
});

// 发送 POST 请求
async function sendPostRequest(url, data) {
    const overlay = document.getElementById('loading-overlay');
    try {
        overlay.style.display = 'flex'; // 显示蒙板
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return response.json();
    } finally {
        overlay.style.display = 'none'; // 隐藏蒙板
    }
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
async function setFile(fileId, fileName) {
    const overlay = document.getElementById('loading-overlay');
    try {
        overlay.style.display = 'flex'; // 显示蒙板
        const fileInput = document.getElementById(fileId);
        const formData = new FormData();
        formData.append(fileName, fileInput.files[0]);
        await fetch("/set_file", {
            method: "POST",
            body: formData
        });
        reloadIframe(fileName === "src_file" ? "src" : "dst");
    } finally {
        overlay.style.display = 'none'; // 隐藏蒙板
    }
}

// 使用执行序文件中行号设置锚点
async function setAnchorLineNum(anchorLineNumId, srcOrDst) {
    const anchorLineNum = document.getElementById(anchorLineNumId).value;
    const response = await sendPostRequest("/set_anchor_line_num", {
        anchor_line_num: anchorLineNum,
        src_or_dst: srcOrDst
    });
    if (response.message === "Invalid anchor line number!") {
        alert("无效的锚点行号！");
        return;
    }
    if (response.message === "Select node is isolated!") {
        alert("选择的锚点是孤立节点，请使用 比较孤立节点 功能比较孤立节点差异！");
        return;
    }
    reloadIframe(srcOrDst);
}

// 使用图中选中节点作为描点
async function setAnchor(srcOrDst) {
    let selectNodes = window.dstGraphNetwork.getSelectedNodes();
    if (srcOrDst === "src") {
        selectNodes = window.srcGraphNetwork.getSelectedNodes();
    }
    if (selectNodes.length !== 1) {
        alert("请选择一个节点作为锚点！");
        return;
    }
    const selectNode = Array.from(selectNodes)[0];
    await sendPostRequest("/set_anchor", {
        anchor_id: selectNode,
        src_or_dst: srcOrDst
    });
    reloadIframe(srcOrDst);
}

// 切换至整图模式
async function changeToWholeGraph() {
    await sendPostRequest("/change_to_whole_graph", {});
    reloadIframe("src");
    reloadIframe("dst");
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

// 融合节点是否导致成环检测
async function fuseNodesCycleCheck(fuseNodesId, srcOrDst) {
    const response = await sendPostRequest("/fuse_nodes_cycle_check", {
        fuse_nodes_id: fuseNodesId,
        src_or_dst: srcOrDst,
    });
    if (!response.has_cycle) {
        return false;
    }
    if (srcOrDst === "src") {
        alert("左侧融合操作不生效，因为该融合会导致图中出现环！");
    } else {
        alert("右侧融合操作不生效，因为该融合会导致图中出现环！");
    }
    return true;
}

// 融合两侧节点
async function fuseTwoSideNodes() {
    const srcFuseNodesId = window.srcGraphNetwork.getSelectedNodes();
    const dstFuseNodesId = window.dstGraphNetwork.getSelectedNodes();
    if (srcFuseNodesId.length < 1 || dstFuseNodesId.length < 1) {
        alert("请在两张图中都至少选择一个节点作为待替换子图！");
        return;
    }
    if (srcFuseNodesId.length === 1) { // 只需融合一侧
        if (await fuseNodesCycleCheck(dstFuseNodesId, "dst")) {
            return;
        }
        const srcFuseNodeId = srcFuseNodesId[0];
        await fuseNodes(dstFuseNodesId, srcFuseNodeId, "", "dst");
        reloadIframe("dst");
        return;
    }
    if (dstFuseNodesId.length === 1) { // 只需融合一侧
        if (await fuseNodesCycleCheck(srcFuseNodesId, "src")) {
            return;
        }
        const dstFuseNodeId = dstFuseNodesId[0];
        await fuseNodes(srcFuseNodesId, dstFuseNodeId, "", "src");
        reloadIframe("src");
        return;
    }
    if (await fuseNodesCycleCheck(srcFuseNodesId, "src")) {
        return;
    }
    if (await fuseNodesCycleCheck(dstFuseNodesId, "dst")) {
        return;
    }
    const fusedNodeId = await fuseNodes(srcFuseNodesId, dstFuseNodesId[0], "FusedNode", "src");
    await fuseNodes(dstFuseNodesId, fusedNodeId, "FusedNode", "dst");
    reloadIframe("src");
    reloadIframe("dst");
}

// 设置层数
async function setLayerNum() {
    const layerNum = document.getElementById("layer-num").value;
    await sendPostRequest("/set_layer_num", {
        layer_num: layerNum
    });
    reloadIframe("src");
    reloadIframe("dst");
}

// 选择匹配的点
async function setMatchNodes() {
    const srcGraphSelectNodes = window.srcGraphNetwork.getSelectedNodes();
    const dstGraphSelectNodes = window.dstGraphNetwork.getSelectedNodes();
    if (srcGraphSelectNodes.length !== 1 || dstGraphSelectNodes.length !== 1) {
        alert("请在两张图中都选择一个节点作为一对匹配的点！");
        return;
    }
    const srcNodeId = srcGraphSelectNodes[0];
    const dstNodeId = dstGraphSelectNodes[0];
    await sendPostRequest("/set_match_nodes", {
        src_node_id: srcNodeId,
        dst_node_id: dstNodeId
    });
    reloadIframe("src");
    reloadIframe("dst");
}

// 删除匹配的点
async function deleteMatchNodes() {
    const srcGraphSelectNodes = window.srcGraphNetwork.getSelectedNodes();
    const dstGraphSelectNodes = window.dstGraphNetwork.getSelectedNodes();
    if (srcGraphSelectNodes.length < 1 && dstGraphSelectNodes.length < 1) {
        alert("请选择要删除的匹配的点！");
        return;
    }
    await sendPostRequest("/delete_match_nodes", {
        src_nodes_id: srcGraphSelectNodes,
        dst_nodes_id: dstGraphSelectNodes
    });
    reloadIframe("src");
    reloadIframe("dst");
}

// 向上比较两图
async function upCompareGraphs() {
    await sendPostRequest("/up_compare_graphs", {});
    reloadIframe("src");
    reloadIframe("dst");
}

// 向下比较两图
async function downCompareGraphs() {
    await sendPostRequest("/down_compare_graphs", {});
    reloadIframe("src");
    reloadIframe("dst");
}

// 删除节点
async function deleteNodes() {
    const srcGraphSelectNodes = window.srcGraphNetwork.getSelectedNodes();
    const dstGraphSelectNodes = window.dstGraphNetwork.getSelectedNodes();
    if (srcGraphSelectNodes.length < 1 && dstGraphSelectNodes.length < 1) {
        alert("请选择要删除的节点！");
        return;
    }
    await sendPostRequest("/delete_nodes", {
        src_delete_nodes_id: srcGraphSelectNodes,
        dst_delete_nodes_id: dstGraphSelectNodes
    });
    if (srcGraphSelectNodes.length > 0) {
        reloadIframe("src");
    }
    if (dstGraphSelectNodes.length > 0) {
        reloadIframe("dst");
    }
}

function getSelectEdges(network, edgesId) {
    return edgesId.map(edgeId => {
        const edge = network.body.data.edges.get(edgeId);
        return {
            src_node_id: edge.from,
            dst_node_id: edge.to
        };
    });
}

// 删除边
async function deleteEdges() {
    const srcGraphSelectEdgesId = window.srcGraphNetwork.getSelectedEdges();
    const dstGraphSelectEdgesId = window.dstGraphNetwork.getSelectedEdges();
    if (srcGraphSelectEdgesId.length < 1 && dstGraphSelectEdgesId.length < 1) {
        alert("请选择要删除的边！");
        return;
    }
    const srcGraphSelectEdges = getSelectEdges(window.srcGraphNetwork, srcGraphSelectEdgesId);
    const dstGraphSelectEdges = getSelectEdges(window.dstGraphNetwork, dstGraphSelectEdgesId);
    await sendPostRequest("/delete_edges", {
        src_edges: srcGraphSelectEdges,
        dst_edges: dstGraphSelectEdges
    });
    if (srcGraphSelectEdges.length > 0) {
        reloadIframe("src");
    }
    if (dstGraphSelectEdges.length > 0) {
        reloadIframe("dst");
    }
}

// 聚焦指定节点
async function focusNode(e) {
    e.preventDefault(); // 阻止表单默认提交行为

    const nodeLineNumInput = document.getElementById("focus-node-line-num").value;
    if (!nodeLineNumInput) {
        alert("请输入要聚焦的节点行号！");
        return;
    }
    const nodeLineNum = parseInt(nodeLineNumInput, 10);

    const focusNodeGraph = document.getElementById("focus-node-graph").value;
    let network = window.srcGraphNetwork;
    if (focusNodeGraph === "dst") {
        network = window.dstGraphNetwork;
    }

    const response = await sendPostRequest("/get_node_id_by_line_num", {
        node_line_num: nodeLineNum,
        src_or_dst: focusNodeGraph
    });
    const nodeId = response.node_id;

    if (network.body.data.nodes.get(nodeId) === null) {
        alert(`未找到行号为 "${nodeLineNum}" 的节点！`);
        return;
    }

    // 聚焦到该节点
    network.focus(nodeId, {
        scale: 1,
        animation: true,
        duration: 1000
    });

    // 高亮该节点
    network.selectNodes([nodeId]);
}

async function undo() {
    const response = await sendPostRequest("/undo", {});
    if (response.message === "No previous operation!") {
        alert("没有可撤销的操作！");
        return;
    }
    reloadIframe("src");
    reloadIframe("dst");
}

function createDiffOverlay() {
    const diffOverlay = document.createElement('div');
    Object.assign(diffOverlay.style, {
        position: 'fixed',
        top: '0px',
        left: '0px',
        width: '100%',
        height: '80%',
        background: 'rgba(0,0,0,0)',
        zIndex: '1000',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '20px',
        color: 'white'
    });
    return diffOverlay;
}

function createDiffContent() {
    const content = document.createElement('div');
    Object.assign(content.style, {
        flex: '1',
        overflowY: 'auto',
        width: '80%',
        backgroundColor: '#333',
        padding: '20px',
        borderRadius: '5px',
        boxSizing: 'border-box',
        whiteSpace: 'pre-wrap'
    });
    return content;
}

function createDiffCloseBtn() {
    const closeBtn = document.createElement('button');
    Object.assign(closeBtn.style, {
        marginTop: '20px',
        padding: '10px 30px',
        background: '#007bff',
        border: 'none',
        borderRadius: '5px',
        color: 'white',
        cursor: 'pointer'
    });
    closeBtn.textContent = '关闭';
    return closeBtn;
}

function showIsolatedNodesDiffResult(response) {
    const diffOverlay = createDiffOverlay();

    const content = createDiffContent();
    content.innerHTML = response.diff_result;
    diffOverlay.appendChild(content);

    const closeBtn = createDiffCloseBtn();
    closeBtn.onclick = () => document.body.removeChild(diffOverlay);
    diffOverlay.appendChild(closeBtn);

    document.body.appendChild(diffOverlay);
}

// 比较孤立节点差异
async function compareIsolatedNodes() {
    const response = await sendPostRequest("/compare_isolated_nodes", {});
    showIsolatedNodesDiffResult(response);
}

function updateProgress(data) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressMessage = document.getElementById('progress-message');
    const percentage = Math.round((data.current / data.total) * 100);
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${percentage}%`;
    progressMessage.textContent = data.message;
}

let eventSource;
function initProgress() {
    eventSource = new EventSource('/get_progress');
    eventSource.onmessage = function(e) {
        const data = JSON.parse(e.data);
        updateProgress(data);
    };
}

// 初始化
document.addEventListener("DOMContentLoaded", () => {
    let setFileButton = document.getElementById("set-src-file-button");
    setFileButton.addEventListener("click", async () => {
        await setFile("src-file", "src_file");
    });
    setFileButton = document.getElementById("set-dst-file-button");
    setFileButton.addEventListener("click", async () => {
        await setFile("dst-file", "dst_file");
    });

    let setAnchorLineNumButton = document.getElementById("set-src-anchor-line-num-button");
    setAnchorLineNumButton.addEventListener("click", async () => {
        await setAnchorLineNum("src-anchor-line-num", "src")
    });
    setAnchorLineNumButton = document.getElementById("set-dst-anchor-line-num-button");
    setAnchorLineNumButton.addEventListener("click", async () => {
        await setAnchorLineNum("dst-anchor-line-num", "dst")
    });

    let setAnchorButton = document.getElementById("set-src-anchor-button");
    setAnchorButton.addEventListener("click", async () => {
        await setAnchor("src");
    });
    setAnchorButton = document.getElementById("set-dst-anchor-button");
    setAnchorButton.addEventListener("click", async () => {
        await setAnchor("dst");
    });

    const changeToWholeGraphButton = document.getElementById("change-to-whole-graph-button");
    changeToWholeGraphButton.addEventListener("click", async () => {
        await changeToWholeGraph();
    });

    const fuseNodesButton = document.getElementById("fuse-nodes-button");
    fuseNodesButton.addEventListener("click", async () => {
        await fuseTwoSideNodes();
    });

    const setLayerNumButton = document.getElementById("set-layer-num-button");
    setLayerNumButton.addEventListener("click", async () => {
        await setLayerNum();
    });

    const setMatchNodesButton = document.getElementById("set-match-nodes-button");
    setMatchNodesButton.addEventListener("click", async () => {
        await setMatchNodes();
    });

    const deleteMatchNodesButton = document.getElementById("delete-match-nodes-button");
    deleteMatchNodesButton.addEventListener("click", async () => {
        await deleteMatchNodes();
    });

    const upCompareButton = document.getElementById("up-compare-button");
    upCompareButton.addEventListener("click", async () => {
        await upCompareGraphs();
    });

    const downCompareButton = document.getElementById("down-compare-button");
    downCompareButton.addEventListener("click", async () => {
        await downCompareGraphs();
    });

    const deleteNodesButton = document.getElementById("delete-nodes-button");
    deleteNodesButton.addEventListener("click", async () => {
        await deleteNodes();
    });

    const deleteEdgesButton = document.getElementById("delete-edges-button");
    deleteEdgesButton.addEventListener("click", async () => {
        await deleteEdges();
    });

    const focusNodeForm = document.getElementById("focus-node-form");
    focusNodeForm.addEventListener("submit", async e => {
        await focusNode(e);
    });

    const undoButton = document.getElementById("undo-button");
    undoButton.addEventListener("click", async () => {
        await undo();
    });

    const compareIsolatedNodesButton = document.getElementById("compare-isolated-nodes-button");
    compareIsolatedNodesButton.addEventListener("click", async () => {
        await compareIsolatedNodes();
    });

    initProgress();
});