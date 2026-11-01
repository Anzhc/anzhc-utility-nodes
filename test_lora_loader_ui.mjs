import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

class BaseWidget {
    constructor(name) {
        this.name = typeof name === "string" ? name : "divider";
    }
    computeSize(width) {
        return [width, 20];
    }
}

class BaseNode {
    constructor() {
        this.properties = {};
        this.widgets = [];
        this.size = [500, 100];
        this.flags = {};
        this.id = -1;
    }
    configure(info) {
        Object.assign(this.properties, info.properties);
        this.flags = { ...info.flags };
        if (info.size) this.size = [...info.size];
        if (info.id != null) this.id = info.id;
    }
    serialize() {
        return structuredClone({
            id: this.id,
            properties: this.properties,
            flags: this.flags,
            size: this.size,
            widgets_values: this.widgets.map((widget, index) => widget.serializeValue?.(this, index) ?? widget.value),
        });
    }
    addCustomWidget(widget) {
        this.widgets.push(widget);
        return widget;
    }
    addWidget(type, name, value, callback, options) {
        return this.addCustomWidget({ type, name, value, callback, options });
    }
    removeWidget(index) {
        this.widgets.splice(index, 1);
    }
    computeSize() {
        return [this.size[0], 50 + this.widgets.reduce((height, widget) => height + (widget.computeSize?.(this.size[0])[1] ?? 20), 0)];
    }
    setDirtyCanvas() {}
}

function moveArrayItem(items, item, index) {
    items.splice(items.indexOf(item), 1);
    items.splice(index, 0, item);
}

const source = readFileSync(new URL("web/lora_loader_node.js", import.meta.url), "utf8").replace(/^import .*;\r?$/gm, "");
const { AnzhcPowerLoraLoader } = new Function(
    "app", "RgthreeBaseServerNode", "RgthreeBaseWidget", "RgthreeBetterButtonWidget", "RgthreeDividerWidget",
    "rgthree", "rgthreeApi", "LORA_INFO_SERVICE", "LiteGraph", "drawNumberWidgetPart", "moveArrayItem", "setTimeout",
    source + "\nreturn { AnzhcPowerLoraLoader };",
)(
    { registerExtension() {} }, BaseNode, BaseWidget, BaseWidget, BaseWidget,
    { newLogSession: () => ({}) }, { getLoras: () => Promise.resolve([]) }, { getInfo: () => Promise.resolve(null) },
    { NODE_WIDGET_HEIGHT: 20 }, { WIDTH_TOTAL: 80 }, moveArrayItem, () => {},
);

function createNode() {
    const node = new AnzhcPowerLoraLoader();
    node.onNodeCreated();
    return node;
}

function loras(node) {
    return node.widgets.filter((widget) => widget.name?.startsWith("lora_"));
}

test("copy/paste without an id preserves hidden weights, flags, strengths and SVC toggles", () => {
    const node = createNode();
    const widget = node.addNewLoraWidget("test.safetensors");
    Object.assign(widget.value, { on: false, merge: true, strength: 0.7, early_blocks: 0.4, mid_blocks: 1.3, text: 0.2 });
    node.toggleBlockWeights();
    node.setMergeMethod("svc");
    node.flags.collapsed = true;
    node.size = [740, 300];
    const saved = node.serialize();
    delete saved.id;
    const pasted = createNode();
    pasted.configure(saved);
    assert.equal(pasted.areBlockWeightsEnabled(), false);
    assert.equal(pasted.flags.collapsed, true);
    assert.equal(pasted.getMergeMethod(), "svc");
    assert.deepEqual(pasted.size, [740, 300]);
    assert.deepEqual(loras(pasted)[0].serializeValue(pasted), widget.serializeValue(node));
    pasted.toggleBlockWeights();
    assert.equal(loras(pasted)[0].value.early_blocks, 0.4);
    assert.equal(loras(pasted)[0].value.mid_blocks, 1.3);
    assert.equal(loras(pasted)[0].value.text, 0.2);
});

test("separate model and clip strengths survive saving before the first draw", () => {
    const node = createNode();
    node.properties["Show Strengths"] = "Separate Model & Clip";
    const widget = node.addNewLoraWidget("test.safetensors");
    widget.value = { ...widget.value, strength: 0.7, strengthTwo: 0.25 };
    const saved = node.serialize();
    delete saved.id;
    const pasted = createNode();
    pasted.configure(saved);
    assert.equal(pasted.properties["Show Strengths"], "Separate Model & Clip");
    assert.equal(loras(pasted)[0].serializeValue(pasted).strengthTwo, 0.25);
    assert.equal(loras(pasted)[0].value.strength, 0.7);
});

test("API values restore hidden weights and separate clip mode without workflow properties", () => {
    const node = createNode();
    node.configure({
        merge_method: "svc",
        widgets_values: [{ lora: "test.safetensors", on: true, merge: true, strength: 0.6, strengthTwo: 0.2,
            block_weights_enabled: false, late_blocks: 0.3 }],
    });
    assert.equal(node.areBlockWeightsEnabled(), false);
    assert.equal(node.properties["Show Strengths"], "Separate Model & Clip");
    assert.equal(node.properties.merge_method, "svc");
    assert.equal(loras(node)[0].serializeValue(node).late_blocks, 0.3);
    assert.equal(loras(node)[0].serializeValue(node).strengthTwo, 0.2);
});

test("repeated configure keeps row names stable and controls compact row height immediately", () => {
    const node = createNode();
    node.addNewLoraWidget("first.safetensors");
    node.addNewLoraWidget("second.safetensors");
    node.toggleBlockWeights();
    const saved = node.serialize();
    for (let iteration = 0; iteration < 3; iteration++) node.configure(saved);
    assert.deepEqual(loras(node).map((widget) => widget.name), ["lora_1", "lora_2"]);
    assert.equal(loras(node)[0].getLayoutMetrics().totalHeight, 32);
    assert.equal(node.addNewLoraWidget("third.safetensors").name, "lora_3");
    node.toggleBlockWeights();
    assert.ok(loras(node)[0].getLayoutMetrics().totalHeight > 32);
});
