import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAMES = new Set(["Anzhc LM Studio LLM"]);
const MODEL_WIDGET_NAME = "model";
const REFRESH_WIDGET_NAME = "Refresh Models";
const EMPTY_MODEL_OPTION = "<no active LM Studio models>";
const PATCH_FLAG = "__anzhcLmStudioPatched";
const GRAPH_PATCH_FLAG = "__anzhcLmStudioGraphPatched";
const REFRESH_WIDGET_PROP = "__anzhcLmStudioRefreshWidget";

function isTargetNode(node) {
    return NODE_NAMES.has(node?.comfyClass) || NODE_NAMES.has(node?.type);
}

function getWidgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

async function fetchModels() {
    const response = await api.fetchApi("/anzhc/lm-studio/models", { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(payload?.error ?? "Failed to fetch LM Studio models.");
    }

    const models = Array.isArray(payload?.models) ? payload.models.filter((value) => typeof value === "string") : [];
    return models.length ? models : [EMPTY_MODEL_OPTION];
}

function applyModelOptions(node, models) {
    const modelWidget = getWidgetByName(node, MODEL_WIDGET_NAME);
    if (!modelWidget) {
        return;
    }

    modelWidget.options.values = models.length ? models : [EMPTY_MODEL_OPTION];
    if (!modelWidget.options.values.includes(modelWidget.value)) {
        modelWidget.value = modelWidget.options.values[0];
        modelWidget.callback?.(modelWidget.value, app.canvas, node, modelWidget);
    }

    node.setDirtyCanvas(true, true);
}

async function refreshModels(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const refreshWidget = node[REFRESH_WIDGET_PROP] ?? getWidgetByName(node, REFRESH_WIDGET_NAME);
    if (refreshWidget) {
        refreshWidget.value = "Loading...";
        node.setDirtyCanvas(true, true);
    }

    try {
        const models = await fetchModels();
        applyModelOptions(node, models);
    } catch (error) {
        console.error("Anzhc LM Studio LLM model refresh failed.", error);

        const modelWidget = getWidgetByName(node, MODEL_WIDGET_NAME);
        const currentOptions = modelWidget?.options?.values;
        if (!Array.isArray(currentOptions) || currentOptions.length === 0) {
            applyModelOptions(node, [EMPTY_MODEL_OPTION]);
        }
    } finally {
        if (refreshWidget) {
            refreshWidget.value = REFRESH_WIDGET_NAME;
            node.setDirtyCanvas(true, true);
        }
    }
}

function ensureRefreshWidget(node) {
    if (!isTargetNode(node)) {
        return;
    }

    if (!getWidgetByName(node, MODEL_WIDGET_NAME)) {
        return;
    }

    if (!node[REFRESH_WIDGET_PROP]) {
        const existingWidget = getWidgetByName(node, REFRESH_WIDGET_NAME);
        if (existingWidget) {
            node[REFRESH_WIDGET_PROP] = existingWidget;
        }
    }

    if (!node[REFRESH_WIDGET_PROP]) {
        const refreshWidget = node.addWidget("button", REFRESH_WIDGET_NAME, REFRESH_WIDGET_NAME, () => {
            refreshModels(node);
        });
        refreshWidget.serialize = false;
        node[REFRESH_WIDGET_PROP] = refreshWidget;

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            node.setSize(node.computeSize());
        }
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
        queueMicrotask(() => {
            ensureRefreshWidget(this);
            refreshModels(this);
        });
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        queueMicrotask(() => {
            ensureRefreshWidget(this);
            refreshModels(this);
        });
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
        queueMicrotask(() => {
            ensureRefreshWidget(node);
            refreshModels(node);
        });
        return result;
    };

    queueMicrotask(() => {
        for (const node of app.graph?._nodes ?? []) {
            ensureRefreshWidget(node);
            refreshModels(node);
        }
    });
}

app.registerExtension({
    name: "anzhc.lm_studio_llm.models",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }
        patchNodeType(nodeType);
    },

    nodeCreated(node) {
        queueMicrotask(() => {
            ensureRefreshWidget(node);
            refreshModels(node);
        });
    },

    setup() {
        for (const nodeName of NODE_NAMES) {
            patchNodeType(LiteGraph.registered_node_types[nodeName]);
        }
        installGraphHooks();
    },
});
