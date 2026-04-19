import { app } from "../../scripts/app.js";

const NODE_NAMES = new Set(["Anzhc Resolution", "ANZC Resolution"]);
const SWAP_WIDGET_NAME = "Swap";
const PATCH_FLAG = "__anzhcResolutionSwapPatched";
const GRAPH_PATCH_FLAG = "__anzhcResolutionGraphPatched";

function getWidgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function ensureSwapWidget(node) {
    if (!NODE_NAMES.has(node.comfyClass) && !NODE_NAMES.has(node.type)) {
        return;
    }

    const widthWidget = getWidgetByName(node, "width");
    const heightWidget = getWidgetByName(node, "height");
    const existingSwapWidget = getWidgetByName(node, SWAP_WIDGET_NAME);
    if (!widthWidget || !heightWidget || existingSwapWidget) {
        return;
    }

    const swapWidget = node.addWidget("button", SWAP_WIDGET_NAME, SWAP_WIDGET_NAME, () => {
        const nextWidth = heightWidget.value;
        const nextHeight = widthWidget.value;

        widthWidget.value = nextWidth;
        heightWidget.value = nextHeight;
        widthWidget.callback?.(nextWidth, app.canvas, node, widthWidget);
        heightWidget.callback?.(nextHeight, app.canvas, node, heightWidget);

        node.setDirtyCanvas(true, true);
    });

    swapWidget.serialize = false;
    if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
        node.setSize(node.computeSize());
    }
}

function patchNodeType(nodeType) {
    if (!nodeType?.prototype || nodeType.prototype[PATCH_FLAG]) {
        return;
    }

    nodeType.prototype[PATCH_FLAG] = true;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        ensureSwapWidget(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        queueMicrotask(() => ensureSwapWidget(this));
        return result;
    };
}

function installGraphHooks() {
    if (!app.graph) {
        setTimeout(installGraphHooks, 0);
        return;
    }

    if (app.graph[GRAPH_PATCH_FLAG]) {
        return;
    }

    app.graph[GRAPH_PATCH_FLAG] = true;

    const onNodeAdded = app.graph.onNodeAdded;
    app.graph.onNodeAdded = function (node) {
        const result = onNodeAdded?.apply?.(this, arguments);
        queueMicrotask(() => ensureSwapWidget(node));
        return result;
    };

    queueMicrotask(() => {
        for (const node of app.graph?._nodes ?? []) {
            ensureSwapWidget(node);
        }
    });
}

app.registerExtension({
    name: "anzhc.resolution.swap",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }
        patchNodeType(nodeType);
    },

    nodeCreated(node) {
        queueMicrotask(() => ensureSwapWidget(node));
    },

    setup() {
        for (const nodeName of NODE_NAMES) {
            patchNodeType(LiteGraph.registered_node_types[nodeName]);
        }
        installGraphHooks();
    },
});
