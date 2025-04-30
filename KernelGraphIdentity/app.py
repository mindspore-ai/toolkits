import os
import webbrowser
import socket
from flask import Flask, jsonify, render_template, request
from ExecutingOrderPreCheckUI import ExecutingOrderPreCheckUI


ui = ExecutingOrderPreCheckUI()
app = Flask(__name__)


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
    ui.save_src_html()
    ui.save_dst_html()

    return render_template("index.html")


@app.route("/set_file", methods=["POST"])
def set_file():
    ui.set_show_all_layer(False)

    src_file = request.files.get("src_file")
    if src_file is not None:
        src_file.save(ui.get_src_executing_order_file_path())
        ui.update_src_executing_order()

    dst_file = request.files.get("dst_file")
    if dst_file is not None:
        dst_file.save(ui.get_dst_executing_order_file_path())
        ui.update_dst_executing_order()

    return jsonify({"message": "Set file successfully!"})


@app.route("/set_anchor_line_num", methods=["POST"])
def set_anchor_line_num():
    ui.set_show_all_layer(False)

    data = request.json
    anchor_line_num = int(data["anchor_line_num"])
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        new_anchor = ui.get_src_executing_order().get_node_manager().get_node_by_line_num(anchor_line_num)
        ui.update_src_first_level_anchor(new_anchor)
    else:
        new_anchor = ui.get_dst_executing_order().get_node_manager().get_node_by_line_num(anchor_line_num)
        ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Set anchor line number successfully!"})


@app.route("/set_anchor", methods=["POST"])
def set_anchor():
    ui.set_show_all_layer(False)

    data = request.json
    anchor_id = data["anchor_id"]
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        new_anchor = ui.get_src_executing_order().get_node_manager().get_node_by_node_id(anchor_id)
        ui.update_src_first_level_anchor(new_anchor)
    else:
        new_anchor = ui.get_dst_executing_order().get_node_manager().get_node_by_node_id(anchor_id)
        ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Set anchor successfully!"})


@app.route("/change_to_whole_graph", methods=["POST"])
def change_to_whole_graph():
    ui.set_show_all_layer(True)

    new_anchor = ui.get_src_executing_order().get_node_manager().get_virtual_node()
    ui.update_src_first_level_anchor(new_anchor)
    new_anchor = ui.get_dst_executing_order().get_node_manager().get_virtual_node()
    ui.update_dst_first_level_anchor(new_anchor)

    return jsonify({"message": "Change to whole graph successfully!"})


@app.route("/fuse_nodes", methods=["POST"])
def fuse_nodes():
    data = request.json
    fuse_nodes_id = data["fuse_nodes_id"]
    peer_node_id = data["peer_node_id"]
    fused_node_operator_type = data["fused_node_operator_type"]
    src_or_dst = data["src_or_dst"]

    if src_or_dst == "src":
        peer_node = ui.get_dst_executing_order().get_node_manager().get_node_by_node_id(peer_node_id)
        fused_node_id = ui.fuse_src_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)
    else:
        peer_node = ui.get_src_executing_order().get_node_manager().get_node_by_node_id(peer_node_id)
        fused_node_id = ui.fuse_dst_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

    return jsonify({"fused_node_id": fused_node_id})


@app.route("/set_layer_num", methods=["POST"])
def set_layer_num():
    data = request.json
    layer_num = int(data["layer_num"])

    ui.set_layer_num(layer_num)

    return jsonify({"message": "Set layer number successfully!"})


@app.route("/set_match_nodes", methods=["POST"])
def set_match_nodes():
    data = request.json
    src_node_id = data["src_node_id"]
    src_node = ui.get_src_executing_order().get_node_manager().get_node_by_node_id(src_node_id)
    dst_node_id = data["dst_node_id"]
    dst_node = ui.get_dst_executing_order().get_node_manager().get_node_by_node_id(dst_node_id)

    ui.add_second_level_anchor(src_node, dst_node)

    return jsonify({"message": "Set match nodes successfully!"})


@app.route("/compare_graphs", methods=["POST"])
def compare_graphs():
    ui.compare_graphs()

    return jsonify({"message": "Compare graphs successfully!"})


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
