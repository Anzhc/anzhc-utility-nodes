import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const NODE_NAMES = new Set(["Anzhc LM Studio LLM", "Anzhc LM Studio Pipeline"]);
const MODEL_WIDGET_NAME = "model";
const FAKE_CHAT_TURNS_WIDGET_NAME = "fake_chat_turns";
const REFRESH_WIDGET_NAME = "Refresh Models";
const ADD_CHAT_TURN_WIDGET_NAME = "Add Chat Turn";
const EMPTY_MODEL_OPTION = "<no active LM Studio models>";
const PATCH_FLAG = "__anzhcLmStudioPatched";
const GRAPH_PATCH_FLAG = "__anzhcLmStudioGraphPatched";
const REFRESH_WIDGET_PROP = "__anzhcLmStudioRefreshWidget";
const FAKE_CHAT_WIDGETS_PROP = "__anzhcLmStudioFakeChatWidgets";

function isTargetNode(node) {
    return NODE_NAMES.has(node?.comfyClass) || NODE_NAMES.has(node?.type);
}

function getWidgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function resizeNodeToWidgets(node) {
    if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
        node.setSize(node.computeSize());
    }
    node.setDirtyCanvas(true, true);
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

function hideFakeChatBackingWidget(node) {
    const widget = getWidgetByName(node, FAKE_CHAT_TURNS_WIDGET_NAME);
    if (!widget) {
        return null;
    }

    widget.hidden = true;
    widget.serialize = true;
    widget.computeSize = () => [0, -4];
    return widget;
}

function parseSerializedFakeChatTurns(node) {
    const widget = getWidgetByName(node, FAKE_CHAT_TURNS_WIDGET_NAME);
    if (!widget) {
        return [];
    }

    const rawValue = String(widget.value ?? "").trim();
    if (!rawValue) {
        return [];
    }

    try {
        const parsed = JSON.parse(rawValue);
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed
            .filter((turn) => turn && typeof turn === "object")
            .map((turn) => ({
                user: String(turn.user ?? ""),
                assistant: String(turn.assistant ?? ""),
            }));
    } catch (error) {
        console.warn("Anzhc LM Studio LLM fake chat turns JSON could not be parsed.", error);
        return [];
    }
}

function setSerializedFakeChatTurns(node, turns) {
    const widget = getWidgetByName(node, FAKE_CHAT_TURNS_WIDGET_NAME);
    if (!widget) {
        return;
    }

    widget.value = JSON.stringify(turns);
    widget.callback?.(widget.value, app.canvas, node, widget);
    node.setDirtyCanvas(true, true);
}

function removeFakeChatControlWidgets(node) {
    const widgets = node[FAKE_CHAT_WIDGETS_PROP] ?? [];
    for (const widget of widgets) {
        const index = node.widgets?.indexOf(widget) ?? -1;
        if (index !== -1) {
            node.widgets.splice(index, 1);
        }
        widget.onRemove?.();
    }
    node[FAKE_CHAT_WIDGETS_PROP] = [];
}

function readFakeChatTurnsFromControls(node) {
    return (node[FAKE_CHAT_WIDGETS_PROP] ?? [])
        .filter((entry) => entry?.__anzhcFakeChatTurn)
        .map((entry) => ({
            user: String(entry.userWidget?.value ?? ""),
            assistant: String(entry.assistantWidget?.value ?? ""),
        }));
}

function syncFakeChatTurnsFromControls(node) {
    setSerializedFakeChatTurns(node, readFakeChatTurnsFromControls(node));
}

function chainWidgetCallback(widget, callback) {
    const originalCallback = widget.callback;
    widget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        callback();
        return result;
    };
}

function addFakeChatTextWidget(node, name, value, onChanged) {
    const created = ComfyWidgets.STRING(
        node,
        name,
        ["STRING", { default: value, multiline: true }],
        app
    );
    const widget = created.widget;
    widget.value = value;
    widget.serialize = false;
    chainWidgetCallback(widget, onChanged);
    return widget;
}

function moveRefreshWidgetToEnd(node) {
    const refreshWidget = node[REFRESH_WIDGET_PROP] ?? getWidgetByName(node, REFRESH_WIDGET_NAME);
    const index = node.widgets?.indexOf(refreshWidget) ?? -1;
    if (index === -1 || index === node.widgets.length - 1) {
        return;
    }

    node.widgets.splice(index, 1);
    node.widgets.push(refreshWidget);
}

function renderFakeChatTurnWidgets(node, turns = parseSerializedFakeChatTurns(node)) {
    if (!isTargetNode(node) || !hideFakeChatBackingWidget(node)) {
        return;
    }

    removeFakeChatControlWidgets(node);

    const controls = [];
    turns.forEach((turn, index) => {
        const userWidget = addFakeChatTextWidget(
            node,
            `Example User ${index + 1}`,
            turn.user,
            () => syncFakeChatTurnsFromControls(node)
        );
        const assistantWidget = addFakeChatTextWidget(
            node,
            `Example Assistant ${index + 1}`,
            turn.assistant,
            () => syncFakeChatTurnsFromControls(node)
        );
        const deleteWidget = node.addWidget(
            "button",
            `Delete Chat Turn ${index + 1}`,
            "Delete",
            () => {
                const nextTurns = readFakeChatTurnsFromControls(node);
                nextTurns.splice(index, 1);
                setSerializedFakeChatTurns(node, nextTurns);
                renderFakeChatTurnWidgets(node, nextTurns);
            }
        );
        deleteWidget.serialize = false;

        const turnEntry = {
            __anzhcFakeChatTurn: true,
            userWidget,
            assistantWidget,
            deleteWidget,
        };
        controls.push(turnEntry, userWidget, assistantWidget, deleteWidget);
    });

    const addWidget = node.addWidget("button", ADD_CHAT_TURN_WIDGET_NAME, ADD_CHAT_TURN_WIDGET_NAME, () => {
        const nextTurns = readFakeChatTurnsFromControls(node);
        nextTurns.push({ user: "", assistant: "" });
        setSerializedFakeChatTurns(node, nextTurns);
        renderFakeChatTurnWidgets(node, nextTurns);
    });
    addWidget.serialize = false;
    controls.push(addWidget);

    node[FAKE_CHAT_WIDGETS_PROP] = controls;
    ensureRefreshWidget(node);
    moveRefreshWidgetToEnd(node);
    resizeNodeToWidgets(node);
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

        resizeNodeToWidgets(node);
    }
}

function ensureFakeChatWidgets(node) {
    if (!isTargetNode(node)) {
        return;
    }

    if (!getWidgetByName(node, FAKE_CHAT_TURNS_WIDGET_NAME)) {
        return;
    }

    if (!Array.isArray(node[FAKE_CHAT_WIDGETS_PROP])) {
        renderFakeChatTurnWidgets(node);
    } else {
        hideFakeChatBackingWidget(node);
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
            ensureFakeChatWidgets(this);
            ensureRefreshWidget(this);
            refreshModels(this);
        });
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        queueMicrotask(() => {
            renderFakeChatTurnWidgets(this);
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
            ensureFakeChatWidgets(node);
            ensureRefreshWidget(node);
            refreshModels(node);
        });
        return result;
    };

    queueMicrotask(() => {
        for (const node of app.graph?._nodes ?? []) {
            ensureFakeChatWidgets(node);
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
            ensureFakeChatWidgets(node);
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
