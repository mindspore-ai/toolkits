import json
import os
import time
import webbrowser
import socket
from flask import Flask, jsonify, render_template, request
from flask import Response, stream_with_context
from ExecutingOrderPreCheckUI import ExecutingOrderPreCheckUI
from ExecutingOrder.NodeManager import INVALID_NODE_ID
from UIHistory import UIHistory

UPDATE_PROGRESS_INTERVAL = 1  # 1s更新一次进度


ui = ExecutingOrderPreCheckUI()
ui_history = UIHistory()
app = Flask(__name__)


def generate_progress(pre_check_ui: ExecutingOrderPreCheckUI):
    while True:
        yield f"data: {json.dumps(pre_check_ui.get_progress())}\n\n"
        time.sleep(UPDATE_PROGRESS_INTERVAL)


@app.route('/get_progress')
def get_progress():
    return Response(stream_with_context(generate_progress(ui)), mimetype="text/event-stream")


@app.route('/')
def get_blank():
    return render_template('blank.html')


@app.route('/set_canvas_size', methods=['POST'])
def set_canvas_size():
    data = request.json
    canvas_height = int(data["canvas_height"])
    canvas_width = int(data["canvas_width"])
    ui.set_canvas_size(canvas_height, canvas_width)

    return jsonify({"message": "Set canvas size successfully!"})


@app.route("/get_index")
def get_index():
    return render_template("index.html")


@app.route("/set_file", methods=["POST"])
def set_file():
    ui_history.save_ui_state(ui)
    ui.set_show_all_layer(False)

    src_file = request.files.get("src_file")
    if src_file is not None:
        file_path = ui.get_src_executing_order_file_path()
        src_file.save(file_path)
        ui.reload_src_executing_order()

    dst_file = request.files.get("dst_file")
    if dst_file is not None:
        file_path = ui.get_dst_executing_order_file_path()
        dst_file.save(file_path)
        ui.reload_dst_executing_order()

    return jsonify({"message": "Set file successfully!"})


@app.route("/set_anchor_line_num", methods=["POST"])
def set_anchor_line_num():
    ui_history.save_ui_state(ui)
    ui.set_show_all_layer(False)

    data = request.json
    anchor_line_num = int(data["anchor_line_num"])
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        new_anchor = ui.get_src_node_by_line_num(anchor_line_num)
        if new_anchor is None:
            return jsonify({"message": "Invalid anchor line number!"})
        if new_anchor in ui.get_src_executing_order().get_node_manager().get_isolated_nodes():
            return jsonify({"message": "Select node is isolated!"})
        ui.update_src_first_level_anchor(new_anchor)
    else:
        new_anchor = ui.get_dst_node_by_line_num(anchor_line_num)
        if new_anchor is None:
            return jsonify({"message": "Invalid anchor line number!"})
        if new_anchor in ui.get_dst_executing_order().get_node_manager().get_isolated_nodes():
            return jsonify({"message": "Select node is isolated!"})
        ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Set anchor line number successfully!"})


@app.route("/set_anchor", methods=["POST"])
def set_anchor():
    ui_history.save_ui_state(ui)
    ui.set_show_all_layer(False)

    data = request.json
    anchor_id = data["anchor_id"]
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        new_anchor = ui.get_src_node_by_node_id(anchor_id)
        ui.update_src_first_level_anchor(new_anchor)
    else:
        new_anchor = ui.get_dst_node_by_node_id(anchor_id)
        ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Set anchor successfully!"})


@app.route("/change_to_whole_graph", methods=["POST"])
def change_to_whole_graph():
    ui_history.save_ui_state(ui)
    ui.set_show_all_layer(True)

    new_anchor = ui.get_src_sink_virtual_node()
    ui.update_src_first_level_anchor(new_anchor)
    new_anchor = ui.get_dst_sink_virtual_node()
    ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Change to whole graph successfully!"})


@app.route("/fuse_nodes_cycle_check", methods=["POST"])
def fuse_nodes_cycle_check():
    data = request.json
    fuse_nodes_id = data["fuse_nodes_id"]
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        has_cycle = ui.fuse_src_nodes_cycle_check(fuse_nodes_id)
    else:
        has_cycle = ui.fuse_dst_nodes_cycle_check(fuse_nodes_id)

    return jsonify({"has_cycle": has_cycle})


@app.route("/fuse_nodes", methods=["POST"])
def fuse_nodes():
    ui_history.save_ui_state(ui)

    data = request.json
    fuse_nodes_id = data["fuse_nodes_id"]
    peer_node_id = data["peer_node_id"]
    fused_node_operator_type = data["fused_node_operator_type"]
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        peer_node = ui.get_dst_node_by_node_id(peer_node_id)
        fused_node_id = ui.fuse_src_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)
    else:
        peer_node = ui.get_src_node_by_node_id(peer_node_id)
        fused_node_id = ui.fuse_dst_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

    return jsonify({"fused_node_id": fused_node_id})


@app.route("/set_layer_num", methods=["POST"])
def set_layer_num():
    ui_history.save_ui_state(ui)

    data = request.json
    layer_num = int(data["layer_num"])

    ui.set_layer_num(layer_num)

    return jsonify({"message": "Set layer number successfully!"})


@app.route("/set_match_nodes", methods=["POST"])
def set_match_nodes():
    ui_history.save_ui_state(ui)

    data = request.json
    src_node_id = data["src_node_id"]
    src_node = ui.get_src_node_by_node_id(src_node_id)
    dst_node_id = data["dst_node_id"]
    dst_node = ui.get_dst_node_by_node_id(dst_node_id)

    ui.add_second_level_anchor(src_node, dst_node)

    return jsonify({"message": "Set match nodes successfully!"})


@app.route("/delete_match_nodes", methods=["POST"])
def delete_match_nodes():
    ui_history.save_ui_state(ui)

    data = request.json
    src_nodes_id = data["src_nodes_id"]
    dst_nodes_id = data["dst_nodes_id"]

    ui.delete_second_level_anchor(src_nodes_id, dst_nodes_id)

    return jsonify({"message": "Delete match nodes successfully!"})


@app.route("/up_compare_graphs", methods=["POST"])
def up_compare_graphs():
    ui_history.save_ui_state(ui)

    ui.up_direction_compare()

    return jsonify({"message": "Up compare graphs successfully!"})


@app.route("/down_compare_graphs", methods=["POST"])
def down_compare_graphs():
    ui_history.save_ui_state(ui)

    ui.down_direction_compare()

    return jsonify({"message": "Down compare graphs successfully!"})


@app.route("/delete_nodes", methods=["POST"])
def delete_nodes():
    ui_history.save_ui_state(ui)

    data = request.json
    src_delete_nodes_id = data["src_delete_nodes_id"]
    dst_delete_nodes_id = data["dst_delete_nodes_id"]

    ui.delete_src_nodes(src_delete_nodes_id)
    ui.delete_dst_nodes(dst_delete_nodes_id)

    return jsonify({"message": "Delete nodes successfully!"})


@app.route("/delete_edges", methods=["POST"])
def delete_edges():
    ui_history.save_ui_state(ui)

    data = request.json
    src_edges = data["src_edges"]
    dst_edges = data["dst_edges"]

    ui.delete_src_edges(src_edges)
    ui.delete_dst_edges(dst_edges)

    return jsonify({"message": "Delete edges successfully!"})


@app.route("/get_node_id_by_line_num", methods=["POST"])
def get_node_id_by_line_num():
    data = request.json
    node_line_num = data["node_line_num"]
    src_or_dst = data["src_or_dst"]

    node_id = INVALID_NODE_ID
    if src_or_dst == "src":
        node = ui.get_src_node_by_line_num(node_line_num)
        if node is not None:
            node_id = node.get_node_id()
    else:
        node = ui.get_dst_node_by_line_num(node_line_num)
        if node is not None:
            node_id = node.get_node_id()

    return jsonify({"node_id": node_id})


@app.route("/undo", methods=["POST"])
def undo():
    global ui
    last_state = ui_history.load_last_ui_state()
    if last_state is None:  # 没有上一步操作
        return jsonify({"message": "No previous operation!"})
    ui = last_state
    ui.save_src_html()
    ui.save_dst_html()
    return jsonify({"message": "Undo successfully!"})


@app.route("/get_node_info", methods=["POST"])
def get_node_info():
    data = request.json
    node_id = data["node_id"]
    src_or_dst = data["src_or_dst"]

    info_lines = []

    if src_or_dst == "src":
        node = ui.get_src_node_by_node_id(node_id)
        info_lines.extend(node.get_node_base_info(""))
        info_lines.extend(node.get_full_scope_info(""))
        info_lines.extend(node.get_stack_info(""))
        info_lines.extend(ui.get_src_node_mismatch_info(node, ""))
    else:
        node = ui.get_dst_node_by_node_id(node_id)
        info_lines.extend(node.get_node_base_info(""))
        info_lines.extend(node.get_full_scope_info(""))
        info_lines.extend(node.get_stack_info(""))
        info_lines.extend(ui.get_dst_node_mismatch_info(node, ""))

    info = "\n".join(info_lines)
    return jsonify({"info": info})


@app.route("/compare_isolated_nodes", methods=["POST"])
def compare_isolated_nodes():
    info_lines = ui.compare_isolated_nodes()
    diff_result = "\n".join(info_lines)
    return jsonify({"diff_result": diff_result})


def find_available_port(start_port=5000):
    port = start_port
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            port += 1


if __name__ == "__main__":
    available_port = find_available_port()
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':  # 解决程序运行时打开两次页面的问题
        webbrowser.open_new(f'http://127.0.0.1:{available_port}/')
    app.run(debug=True, use_reloader=False, port=available_port)
