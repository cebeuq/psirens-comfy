"""Convert API-format workflows to ComfyUI editor (UI) format.

Usage: python3 scripts/api_to_ui.py object_info.json workflows/x_api.json > workflows/x.json
object_info.json comes from GET http://<comfyui>/object_info.
"""
import json
import sys

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def is_widget(spec):
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES


def convert(info, api, layout=None):
    ids = {k: i + 1 for i, k in enumerate(sorted(api, key=int))}
    depth = {}

    def d(k):
        if k not in depth:
            ups = [v[0] for v in api[k]["inputs"].values()
                   if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]

    for k in api:
        d(k)
    nodes, links, col_y = [], [], {}
    out_links = {}
    link_id = 0
    for k in sorted(api, key=lambda k: (depth[k], int(k))):
        n = api[k]
        cls = n["class_type"]
        spec = info[cls]["input"]
        order = info[cls].get("input_order") or {s: list(spec.get(s, {})) for s in ("required", "optional")}
        inputs, widgets = [], []
        for section in ("required", "optional"):
            for name in order.get(section, []):
                s = spec[section][name]
                val = n["inputs"].get(name)
                linked = isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)
                if is_widget(s):
                    t = "COMBO" if isinstance(s[0], list) else s[0]
                    widgets.append("" if linked else val)
                    if name == "seed" or (len(s) > 1 and isinstance(s[1], dict) and s[1].get("control_after_generate")):
                        widgets.append("fixed")
                    if linked:
                        inputs.append({"name": name, "type": t, "widget": {"name": name}, "link": None, "_src": val})
                else:
                    if section == "optional" and not linked:
                        inputs.append({"name": name, "type": s[0], "link": None})
                        continue
                    inputs.append({"name": name, "type": s[0], "link": None, "_src": val})
        outs = [{"name": nm, "type": t, "links": [], "slot_index": i}
                for i, (t, nm) in enumerate(zip(info[cls]["output"], info[cls].get("output_name") or info[cls]["output"]))]
        x = 40 + depth[k] * 400
        y = col_y.get(depth[k], 140)
        h = 90 + 26 * (len(widgets) + max(len(inputs), len(outs)))
        if cls == "PsirensVoicePrompt":
            h += 220
        if cls in ("SaveImage", "PreviewImage"):
            h += 420
        col_y[depth[k]] = y + h + 40
        w = 340
        if layout and k in layout:
            x, y, w, h = layout[k]
        node = {"id": ids[k], "type": cls, "pos": [x, y], "size": [w, h], "flags": {},
                "order": len(nodes), "mode": 0, "inputs": inputs, "outputs": outs,
                "properties": {"Node name for S&R": cls}, "widgets_values": widgets}
        if n.get("_meta", {}).get("title"):
            node["title"] = n["_meta"]["title"]
        nodes.append(node)
        out_links[k] = node
    byid = {nd["id"]: nd for nd in nodes}
    for nd in nodes:
        for slot, inp in enumerate(nd["inputs"]):
            src = inp.pop("_src", None)
            if src is None:
                continue
            link_id += 1
            up = byid[ids[src[0]]]
            up["outputs"][src[1]]["links"].append(link_id)
            inp["link"] = link_id
            links.append([link_id, up["id"], src[1], nd["id"], slot, inp["type"]])
    return {"last_node_id": max(ids.values()), "last_link_id": link_id, "nodes": nodes,
            "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


if __name__ == "__main__":
    info = json.load(open(sys.argv[1]))
    layout = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    print(json.dumps(convert(info, json.load(open(sys.argv[2])), layout), indent=1))
