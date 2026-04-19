import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAMES = new Set(["Anzhc MCP Skills"]);
const SERVER_WIDGET_NAME = "mcp_server";
const SKILL_WIDGET_NAME = "skill";
const REFRESH_WIDGET_NAME = "Refresh Skills";
const EMPTY_SERVER_OPTION = "<no MCP skill packs>";
const EMPTY_SKILL_OPTION = "<no MCP skills>";
const PATCH_FLAG = "__anzhcMcpSkillsPatched";
const GRAPH_PATCH_FLAG = "__anzhcMcpSkillsGraphPatched";
const REFRESH_WIDGET_PROP = "__anzhcMcpSkillsRefreshWidget";
const REGISTRY_PROP = "__anzhcMcpSkillsRegistry";
const CALLBACK_PATCH_PROP = "__anzhcMcpSkillsCallbackPatched";

function isTargetNode(node) {
    return NODE_NAMES.has(node?.comfyClass) || NODE_NAMES.has(node?.type);
}

function getWidgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function normalizeRegistry(payload) {
    const servers = Array.isArray(payload?.servers) ? payload.servers : [];
    const byId = new Map();
    const options = [];

    for (const server of servers) {
        if (!server || typeof server.id !== "string" || !server.id) {
            continue;
        }

        const skills = Array.isArray(server.skills)
            ? server.skills.filter((skill) => skill && typeof skill.name === "string" && skill.name)
            : [];

        byId.set(server.id, {
            id: server.id,
            label: typeof server.label === "string" && server.label ? server.label : server.id,
            skills,
        });
        options.push(server.id);
    }

    return {
        byId,
        options: options.length ? options : [EMPTY_SERVER_OPTION],
    };
}

async function fetchRegistry() {
    const response = await api.fetchApi("/anzhc/mcp/skills", { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(payload?.error ?? "Failed to fetch MCP skills.");
    }

    return normalizeRegistry(payload);
}

function getSkillOptions(registry, serverId) {
    if (!registry?.byId?.has(serverId)) {
        return [EMPTY_SKILL_OPTION];
    }

    const server = registry.byId.get(serverId);
    const skills = Array.isArray(server?.skills) ? server.skills : [];
    const options = skills
        .map((skill) => (typeof skill?.name === "string" ? skill.name : ""))
        .filter((skillName) => skillName);

    return options.length ? options : [EMPTY_SKILL_OPTION];
}

function applySkillOptions(node, serverId) {
    const skillWidget = getWidgetByName(node, SKILL_WIDGET_NAME);
    if (!skillWidget) {
        return;
    }

    const registry = node[REGISTRY_PROP];
    const skillOptions = getSkillOptions(registry, serverId);
    skillWidget.options.values = skillOptions;
    if (!skillWidget.options.values.includes(skillWidget.value)) {
        skillWidget.value = skillWidget.options.values[0];
        skillWidget.callback?.(skillWidget.value, app.canvas, node, skillWidget);
    }

    node.setDirtyCanvas(true, true);
}

function ensureServerCallback(node) {
    const serverWidget = getWidgetByName(node, SERVER_WIDGET_NAME);
    if (!serverWidget || serverWidget[CALLBACK_PATCH_PROP]) {
        return;
    }

    serverWidget[CALLBACK_PATCH_PROP] = true;
    const originalCallback = serverWidget.callback;
    serverWidget.callback = function () {
        originalCallback?.apply(this, arguments);
        applySkillOptions(node, serverWidget.value);
    };
}

function applyRegistry(node, registry) {
    if (!isTargetNode(node)) {
        return;
    }

    const serverWidget = getWidgetByName(node, SERVER_WIDGET_NAME);
    const skillWidget = getWidgetByName(node, SKILL_WIDGET_NAME);
    if (!serverWidget || !skillWidget) {
        return;
    }

    node[REGISTRY_PROP] = registry;
    serverWidget.options.values = registry.options;
    if (!serverWidget.options.values.includes(serverWidget.value)) {
        serverWidget.value = serverWidget.options.values[0];
        serverWidget.callback?.(serverWidget.value, app.canvas, node, serverWidget);
    }

    ensureServerCallback(node);
    applySkillOptions(node, serverWidget.value);
    node.setDirtyCanvas(true, true);
}

async function refreshRegistry(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const refreshWidget = node[REFRESH_WIDGET_PROP] ?? getWidgetByName(node, REFRESH_WIDGET_NAME);
    if (refreshWidget) {
        refreshWidget.value = "Loading...";
        node.setDirtyCanvas(true, true);
    }

    try {
        const registry = await fetchRegistry();
        applyRegistry(node, registry);
    } catch (error) {
        console.error("Anzhc MCP Skills refresh failed.", error);

        const fallbackRegistry = normalizeRegistry({ servers: [] });
        if (!node[REGISTRY_PROP]) {
            applyRegistry(node, fallbackRegistry);
        }
    } finally {
        if (refreshWidget) {
            refreshWidget.value = REFRESH_WIDGET_NAME;
            node.setDirtyCanvas(true, true);
        }
    }
}

function ensureRefreshWidget(node) {
    if (!isTargetNode(node) || !getWidgetByName(node, SERVER_WIDGET_NAME)) {
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
            refreshRegistry(node);
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
            refreshRegistry(this);
        });
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        queueMicrotask(() => {
            ensureRefreshWidget(this);
            refreshRegistry(this);
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
            refreshRegistry(node);
        });
        return result;
    };

    queueMicrotask(() => {
        for (const node of app.graph?._nodes ?? []) {
            ensureRefreshWidget(node);
            refreshRegistry(node);
        }
    });
}

app.registerExtension({
    name: "anzhc.mcp_skills.registry",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }
        patchNodeType(nodeType);
    },

    nodeCreated(node) {
        queueMicrotask(() => {
            ensureRefreshWidget(node);
            refreshRegistry(node);
        });
    },

    setup() {
        for (const nodeName of NODE_NAMES) {
            patchNodeType(LiteGraph.registered_node_types[nodeName]);
        }
        installGraphHooks();
    },
});
