var _a;
import { app } from "/scripts/app.js";
import { RgthreeBaseServerNode } from "/extensions/rgthree-comfy/base_node.js";
import { rgthree } from "/extensions/rgthree-comfy/rgthree.js";
import { addConnectionLayoutSupport } from "/extensions/rgthree-comfy/utils.js";
import { drawInfoIcon, drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "/extensions/rgthree-comfy/utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, RgthreeDividerWidget, } from "/extensions/rgthree-comfy/utils_widgets.js";
import { rgthreeApi } from "/rgthree/common/rgthree_api.js";
import { showLoraChooser } from "/extensions/rgthree-comfy/utils_menu.js";
import { moveArrayItem, removeArrayItem } from "/rgthree/common/shared_utils.js";
import { RgthreeLoraInfoDialog } from "/extensions/rgthree-comfy/dialog_info.js";
import { LORA_INFO_SERVICE } from "/rgthree/common/model_info_service.js";
const PROP_LABEL_SHOW_STRENGTHS = "Show Strengths";
const PROP_LABEL_BLOCK_WEIGHTS = "Block Weights Enabled";
const PROP_LABEL_SHOW_STRENGTHS_STATIC = `@${PROP_LABEL_SHOW_STRENGTHS}`;
const PROP_VALUE_SHOW_STRENGTHS_SINGLE = "Single Strength";
const PROP_VALUE_SHOW_STRENGTHS_SEPARATE = "Separate Model & Clip";
const NODE_CLASS_TYPE = "Anzhc Lora Loader";
const LORA_WIDGET_MARGIN = 10;
const LORA_WIDGET_INNER_MARGIN = LORA_WIDGET_MARGIN * 0.33;
const LORA_INLINE_LABEL_GAP = 5;
const LORA_INLINE_CONTROL_GAP = 10;
function getPowerLoraMinWidth(showModelAndClip = false, showBlockWeights = true) {
    const toggleWidth = LiteGraph.NODE_WIDGET_HEIGHT * 1.5;
    const removeWidth = LiteGraph.NODE_WIDGET_HEIGHT * 0.7;
    const topStrengthWidth = showModelAndClip
        ? drawNumberWidgetPart.WIDTH_TOTAL * 2 + LORA_WIDGET_INNER_MARGIN
        : drawNumberWidgetPart.WIDTH_TOTAL;
    const topRowMinWidth = LORA_WIDGET_MARGIN * 2 +
        toggleWidth +
        LORA_WIDGET_INNER_MARGIN +
        120 +
        LORA_WIDGET_INNER_MARGIN +
        topStrengthWidth +
        LORA_WIDGET_INNER_MARGIN +
        removeWidth;
    const headerButtonWidth = 82;
    const headerMinWidth = LORA_WIDGET_MARGIN * 2 +
        toggleWidth +
        LORA_WIDGET_INNER_MARGIN +
        88 +
        LORA_WIDGET_INNER_MARGIN +
        headerButtonWidth +
        LORA_WIDGET_INNER_MARGIN +
        drawNumberWidgetPart.WIDTH_TOTAL;
    if (!showBlockWeights) {
        return Math.ceil(Math.max(topRowMinWidth, headerMinWidth));
    }
    const detailRowOneWidth = LORA_WIDGET_MARGIN * 2 + 22 +
        40 + LORA_INLINE_LABEL_GAP + drawNumberWidgetPart.WIDTH_TOTAL +
        LORA_INLINE_CONTROL_GAP +
        40 + LORA_INLINE_LABEL_GAP + drawNumberWidgetPart.WIDTH_TOTAL +
        LORA_INLINE_CONTROL_GAP +
        48 + LORA_INLINE_LABEL_GAP + drawNumberWidgetPart.WIDTH_TOTAL;
    const detailRowTwoWidth = LORA_WIDGET_MARGIN * 2 + 22 +
        44 + LORA_INLINE_LABEL_GAP + drawNumberWidgetPart.WIDTH_TOTAL +
        LORA_INLINE_CONTROL_GAP +
        72 + LORA_INLINE_LABEL_GAP + drawNumberWidgetPart.WIDTH_TOTAL;
    return Math.ceil(Math.max(topRowMinWidth, headerMinWidth, detailRowOneWidth, detailRowTwoWidth));
}
class AnzhcPowerLoraLoader extends RgthreeBaseServerNode {
    constructor(title = NODE_CLASS.title) {
        super(title);
        this.serialize_widgets = true;
        this.logger = rgthree.newLogSession(`[Power Lora Stack]`);
        this.loraWidgetsCounter = 0;
        this.widgetButtonSpacer = null;
        this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;
        this.properties[PROP_LABEL_BLOCK_WEIGHTS] = this.properties[PROP_LABEL_BLOCK_WEIGHTS] !== false;
        rgthreeApi.getLoras();
        if (rgthree.loadingApiJson) {
            const fullApiJson = rgthree.loadingApiJson;
            setTimeout(() => {
                this.configureFromApiJson(fullApiJson);
            }, 16);
        }
    }
    configureFromApiJson(fullApiJson) {
        var _b, _c;
        if (this.id == null) {
            const [n, v] = this.logger.errorParts("Cannot load from API JSON without node id.");
            (_b = console[n]) === null || _b === void 0 ? void 0 : _b.call(console, ...v);
            return;
        }
        const nodeData = fullApiJson[this.id] || fullApiJson[String(this.id)] || fullApiJson[Number(this.id)];
        if (nodeData == null) {
            const [n, v] = this.logger.errorParts(`No node found in API JSON for node id ${this.id}.`);
            (_c = console[n]) === null || _c === void 0 ? void 0 : _c.call(console, ...v);
            return;
        }
        this.configure({
            widgets_values: Object.values(nodeData.inputs).filter((input) => typeof (input === null || input === void 0 ? void 0 : input["lora"]) === "string"),
        });
    }
    configure(info) {
        var _b;
        while ((_b = this.widgets) === null || _b === void 0 ? void 0 : _b.length)
            this.removeWidget(0);
        this.widgetButtonSpacer = null;
        if (info.id != null) {
            super.configure(info);
        }
        this._tempWidth = this.size[0];
        this._tempHeight = this.size[1];
        for (const widgetValue of info.widgets_values || []) {
            if ((widgetValue === null || widgetValue === void 0 ? void 0 : widgetValue.lora) !== undefined) {
                const widget = this.addNewLoraWidget();
                widget.value = { ...widgetValue };
            }
        }
        this.addNonLoraWidgets();
        this.size[0] = this.computeSize()[0];
        this.size[1] = Math.max(this._tempHeight, this.computeSize()[1]);
    }
    onNodeCreated() {
        var _b;
        (_b = super.onNodeCreated) === null || _b === void 0 ? void 0 : _b.call(this);
        this.addNonLoraWidgets();
        const computed = this.computeSize();
        this.size = this.size || [0, 0];
        this.size[0] = Math.max(this.size[0], computed[0]);
        this.size[1] = Math.max(this.size[1], computed[1]);
        this.setDirtyCanvas(true, true);
    }
    areBlockWeightsEnabled() {
        return this.properties[PROP_LABEL_BLOCK_WEIGHTS] !== false;
    }
    getMinWidth() {
        return getPowerLoraMinWidth(this.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE, this.areBlockWeightsEnabled());
    }
    onResize(size) {
        size[0] = Math.max(size[0], this.getMinWidth());
        return super.onResize ? super.onResize(size) : undefined;
    }
    toggleBlockWeights() {
        this.properties[PROP_LABEL_BLOCK_WEIGHTS] = !this.areBlockWeightsEnabled();
        const computed = this.computeSize();
        this.size[0] = Math.max(this.size[0], this.getMinWidth());
        this.size[1] = Math.max(this._tempHeight || 15, computed[1]);
        this.setDirtyCanvas(true, true);
    }
    addNewLoraWidget(lora) {
        this.loraWidgetsCounter++;
        const widget = this.addCustomWidget(new PowerLoraLoaderWidget("lora_" + this.loraWidgetsCounter));
        widget.node = this;
        if (lora)
            widget.setLora(lora);
        if (this.widgetButtonSpacer) {
            moveArrayItem(this.widgets, widget, this.widgets.indexOf(this.widgetButtonSpacer));
        }
        return widget;
    }
    addNonLoraWidgets() {
        moveArrayItem(this.widgets, this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 })), 0);
        moveArrayItem(this.widgets, this.addCustomWidget(new PowerLoraLoaderHeaderWidget()), 1);
        this.widgetButtonSpacer = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));
        this.addCustomWidget(new RgthreeBetterButtonWidget("➕ Add Lora", (event, pos, node) => {
            rgthreeApi.getLoras().then((lorasDetails) => {
                const loras = lorasDetails.map((l) => l.file);
                showLoraChooser(event, (value) => {
                    var _b;
                    if (typeof value === "string") {
                        if (value.includes("Power Lora Chooser")) {
                        }
                        else if (value !== "NONE") {
                            this.addNewLoraWidget(value);
                            const computed = this.computeSize();
                            const tempHeight = (_b = this._tempHeight) !== null && _b !== void 0 ? _b : 15;
                            this.size[0] = Math.max(this.size[0], computed[0]);
                            this.size[1] = Math.max(tempHeight, computed[1]);
                            this.setDirtyCanvas(true, true);
                        }
                    }
                }, null, [...loras]);
            });
            return true;
        }));
    }
    getSlotInPosition(canvasX, canvasY) {
        var _b;
        const slot = super.getSlotInPosition(canvasX, canvasY);
        if (!slot) {
            let lastWidget = null;
            for (const widget of this.widgets) {
                if (!widget.last_y)
                    return;
                if (canvasY > this.pos[1] + widget.last_y) {
                    lastWidget = widget;
                    continue;
                }
                break;
            }
            if ((_b = lastWidget === null || lastWidget === void 0 ? void 0 : lastWidget.name) === null || _b === void 0 ? void 0 : _b.startsWith("lora_")) {
                return { widget: lastWidget, output: { type: "LORA WIDGET" } };
            }
        }
        return slot;
    }
    getSlotMenuOptions(slot) {
        var _b, _c, _d, _e, _f, _g;
        if ((_c = (_b = slot === null || slot === void 0 ? void 0 : slot.widget) === null || _b === void 0 ? void 0 : _b.name) === null || _c === void 0 ? void 0 : _c.startsWith("lora_")) {
            const widget = slot.widget;
            const index = this.widgets.indexOf(widget);
            const canMoveUp = !!((_e = (_d = this.widgets[index - 1]) === null || _d === void 0 ? void 0 : _d.name) === null || _e === void 0 ? void 0 : _e.startsWith("lora_"));
            const canMoveDown = !!((_g = (_f = this.widgets[index + 1]) === null || _f === void 0 ? void 0 : _f.name) === null || _g === void 0 ? void 0 : _g.startsWith("lora_"));
            const menuItems = [
                {
                    content: `ℹ️ Show Info`,
                    callback: () => {
                        widget.showLoraInfoDialog();
                    },
                },
                null,
                {
                    content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
                    callback: () => {
                        widget.value.on = !widget.value.on;
                    },
                },
                {
                    content: `⬆️ Move Up`,
                    disabled: !canMoveUp,
                    callback: () => {
                        moveArrayItem(this.widgets, widget, index - 1);
                    },
                },
                {
                    content: `⬇️ Move Down`,
                    disabled: !canMoveDown,
                    callback: () => {
                        moveArrayItem(this.widgets, widget, index + 1);
                    },
                },
                {
                    content: `🗑️ Remove`,
                    callback: () => {
                        removeArrayItem(this.widgets, widget);
                    },
                },
            ];
            new LiteGraph.ContextMenu(menuItems, {
                title: "LORA WIDGET",
                event: rgthree.lastCanvasMouseEvent,
            });
            return undefined;
        }
        return this.defaultGetSlotMenuOptions(slot);
    }
    refreshComboInNode(defs) {
        rgthreeApi.getLoras(true);
    }
    hasLoraWidgets() {
        var _b;
        return !!((_b = this.widgets) === null || _b === void 0 ? void 0 : _b.find((w) => { var _b; return (_b = w.name) === null || _b === void 0 ? void 0 : _b.startsWith("lora_"); }));
    }
    allLorasState() {
        var _b, _c, _d;
        let allOn = true;
        let allOff = true;
        for (const widget of this.widgets) {
            if ((_b = widget.name) === null || _b === void 0 ? void 0 : _b.startsWith("lora_")) {
                const on = (_c = widget.value) === null || _c === void 0 ? void 0 : _c.on;
                allOn = allOn && on === true;
                allOff = allOff && on === false;
                if (!allOn && !allOff) {
                    return null;
                }
            }
        }
        return allOn && ((_d = this.widgets) === null || _d === void 0 ? void 0 : _d.length) ? true : false;
    }
    toggleAllLoras() {
        var _b, _c;
        const allOn = this.allLorasState();
        const toggledTo = !allOn ? true : false;
        for (const widget of this.widgets) {
            if (((_b = widget.name) === null || _b === void 0 ? void 0 : _b.startsWith("lora_")) && ((_c = widget.value) === null || _c === void 0 ? void 0 : _c.on) != null) {
                widget.value.on = toggledTo;
            }
        }
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        addConnectionLayoutSupport(NODE_CLASS, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        setTimeout(() => {
            NODE_CLASS.category = comfyClass.category;
        });
    }
}
_a = PROP_LABEL_SHOW_STRENGTHS_STATIC;
AnzhcPowerLoraLoader.title = NODE_CLASS_TYPE;
AnzhcPowerLoraLoader.type = NODE_CLASS_TYPE;
AnzhcPowerLoraLoader.comfyClass = NODE_CLASS_TYPE;
AnzhcPowerLoraLoader[_a] = {
    type: "combo",
    values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
};
class PowerLoraLoaderHeaderWidget extends RgthreeBaseWidget {
    constructor(name = "PowerLoraLoaderHeaderWidget") {
        super(name);
        this.value = { type: "PowerLoraLoaderHeaderWidget" };
        this.type = "custom";
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown },
            blockWeights: { bounds: [0, 0, 0, 0], onClick: this.onBlockWeightsClick },
        };
        this.showModelAndClip = null;
    }
    draw(ctx, node, w, posY, height) {
        if (!node.hasLoraWidgets()) {
            return;
        }
        this.showModelAndClip =
            node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const allLoraState = node.allLorasState();
        const blockWeightsEnabled = node.areBlockWeightsEnabled();
        posY += 2;
        const midY = posY + height * 0.5;
        let posX = 10;
        ctx.save();
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: allLoraState });
        if (!lowQuality) {
            posX += this.hitAreas.toggle.bounds[1] + innerMargin;
            ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText("Toggle All", posX, midY);
            const buttonLabel = blockWeightsEnabled ? "Hide Weights" : "Show Weights";
            const buttonPaddingX = 8;
            const buttonWidth = Math.ceil(ctx.measureText(buttonLabel).width + buttonPaddingX * 2);
            const buttonHeight = Math.max(18, height - 8);
            const buttonX = posX + ctx.measureText("Toggle All").width + innerMargin * 3;
            const buttonY = posY + (height - buttonHeight) * 0.5;
            ctx.save();
            ctx.globalAlpha = app.canvas.editor_alpha * (blockWeightsEnabled ? 0.18 : 0.10);
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.beginPath();
            ctx.roundRect(buttonX, buttonY, buttonWidth, buttonHeight, [4]);
            ctx.fill();
            ctx.restore();
            ctx.globalAlpha = app.canvas.editor_alpha * 0.75;
            ctx.textAlign = "center";
            ctx.fillText(buttonLabel, buttonX + buttonWidth * 0.5, midY);
            this.hitAreas.blockWeights.bounds = [buttonX, buttonY, buttonWidth, buttonHeight];
            let rposX = node.size[0] - margin - innerMargin - innerMargin;
            ctx.textAlign = "center";
            ctx.fillText(this.showModelAndClip ? "Clip" : "Strength", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
            if (this.showModelAndClip) {
                rposX = rposX - drawNumberWidgetPart.WIDTH_TOTAL - innerMargin * 2;
                ctx.fillText("Model", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
            }
        }
        ctx.restore();
    }
    onToggleDown(event, pos, node) {
        node.toggleAllLoras();
        this.cancelMouseDown();
        return true;
    }
    onBlockWeightsClick(event, pos, node) {
        node.toggleBlockWeights();
        this.cancelMouseDown();
        return true;
    }
}
const DEFAULT_LORA_WIDGET_DATA = {
    on: true,
    lora: null,
    strength: 1,
    strengthTwo: null,
    early_blocks: 1,
    mid_blocks: 1,
    late_blocks: 1,
    text: 1,
    others: 1,
};
class PowerLoraLoaderWidget extends RgthreeBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
        this.haveMouseMovedStrength = false;
        this.loraInfoPromise = null;
        this.loraInfo = null;
        this.showModelAndClip = null;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown },
            lora: { bounds: [0, 0], onClick: this.onLoraClick },
            remove: { bounds: [0, 0, 0, 0], onClick: this.onRemoveClick },
            strengthDec: { bounds: [0, 0], onClick: this.onStrengthDecDown },
            strengthVal: { bounds: [0, 0], onClick: this.onStrengthValUp },
            strengthInc: { bounds: [0, 0], onClick: this.onStrengthIncDown },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove },
            strengthTwoDec: { bounds: [0, 0], onClick: this.onStrengthTwoDecDown },
            strengthTwoVal: { bounds: [0, 0], onClick: this.onStrengthTwoValUp },
            strengthTwoInc: { bounds: [0, 0], onClick: this.onStrengthTwoIncDown },
            strengthTwoAny: { bounds: [0, 0], onMove: this.onStrengthTwoAnyMove },
            earlyBlocksDec: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericDecDown("early_blocks") },
            earlyBlocksVal: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericValUp(event, "early_blocks") },
            earlyBlocksInc: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericIncDown("early_blocks") },
            earlyBlocksAny: { bounds: [0, 0], onMove: (event, pos, node) => this.onNumericAnyMove(event, "early_blocks") },
            midBlocksDec: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericDecDown("mid_blocks") },
            midBlocksVal: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericValUp(event, "mid_blocks") },
            midBlocksInc: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericIncDown("mid_blocks") },
            midBlocksAny: { bounds: [0, 0], onMove: (event, pos, node) => this.onNumericAnyMove(event, "mid_blocks") },
            lateBlocksDec: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericDecDown("late_blocks") },
            lateBlocksVal: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericValUp(event, "late_blocks") },
            lateBlocksInc: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericIncDown("late_blocks") },
            lateBlocksAny: { bounds: [0, 0], onMove: (event, pos, node) => this.onNumericAnyMove(event, "late_blocks") },
            textDec: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericDecDown("text") },
            textVal: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericValUp(event, "text") },
            textInc: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericIncDown("text") },
            textAny: { bounds: [0, 0], onMove: (event, pos, node) => this.onNumericAnyMove(event, "text") },
            othersDec: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericDecDown("others") },
            othersVal: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericValUp(event, "others") },
            othersInc: { bounds: [0, 0], onClick: (event, pos, node) => this.onNumericIncDown("others") },
            othersAny: { bounds: [0, 0], onMove: (event, pos, node) => this.onNumericAnyMove(event, "others") },
        };
        this._value = { ...DEFAULT_LORA_WIDGET_DATA };
    }
    set value(v) {
        this._value = this.normalizeValue(v);
        this.getLoraInfo();
    }
    get value() {
        return this._value;
    }
    normalizeValue(v) {
        let nextValue = typeof v === "object" && v != null ? { ...DEFAULT_LORA_WIDGET_DATA, ...v } : { ...DEFAULT_LORA_WIDGET_DATA };
        if (this.showModelAndClip && nextValue.strengthTwo == null) {
            nextValue.strengthTwo = nextValue.strength;
        }
        else if (!this.showModelAndClip) {
            nextValue.strengthTwo = nextValue.strengthTwo ?? null;
        }
        for (const prop of ["strength", "strengthTwo", "early_blocks", "mid_blocks", "late_blocks", "text", "others"]) {
            if (nextValue[prop] == null) {
                continue;
            }
            const numericValue = Number(nextValue[prop]);
            nextValue[prop] = Number.isFinite(numericValue) ? numericValue : DEFAULT_LORA_WIDGET_DATA[prop];
        }
        return nextValue;
    }
    setLora(lora) {
        this._value.lora = lora;
        this.getLoraInfo();
    }
    areBlockWeightsEnabled() {
        var _b;
        return ((_b = this.node) === null || _b === void 0 ? void 0 : _b.areBlockWeightsEnabled()) !== false;
    }
    computeSize(width) {
        const layout = this.getLayoutMetrics();
        const minWidth = getPowerLoraMinWidth(this.showModelAndClip === true, this.areBlockWeightsEnabled());
        return [Math.max(width || minWidth, minWidth), layout.totalHeight];
    }
    getLayoutMetrics() {
        const topRowHeight = LiteGraph.NODE_WIDGET_HEIGHT;
        const detailRowHeight = Math.max(18, Math.round(topRowHeight * 0.9));
        const paddingY = 6;
        const rowGap = 3;
        const blockWeightsEnabled = this.areBlockWeightsEnabled();
        return {
            topRowHeight,
            detailRowHeight,
            paddingY,
            rowGap,
            totalHeight: blockWeightsEnabled
                ? paddingY * 2 + topRowHeight + rowGap + detailRowHeight + rowGap + detailRowHeight
                : paddingY * 2 + topRowHeight,
        };
    }
    resetHiddenStrengthTwoHitAreas() {
        this.hitAreas.strengthTwoDec.bounds = [0, -1];
        this.hitAreas.strengthTwoVal.bounds = [0, -1];
        this.hitAreas.strengthTwoInc.bounds = [0, -1];
        this.hitAreas.strengthTwoAny.bounds = [0, -1];
    }
    resetBlockWeightHitAreas() {
        this.hitAreas.earlyBlocksDec.bounds = [0, -1];
        this.hitAreas.earlyBlocksVal.bounds = [0, -1];
        this.hitAreas.earlyBlocksInc.bounds = [0, -1];
        this.hitAreas.earlyBlocksAny.bounds = [0, -1];
        this.hitAreas.midBlocksDec.bounds = [0, -1];
        this.hitAreas.midBlocksVal.bounds = [0, -1];
        this.hitAreas.midBlocksInc.bounds = [0, -1];
        this.hitAreas.midBlocksAny.bounds = [0, -1];
        this.hitAreas.lateBlocksDec.bounds = [0, -1];
        this.hitAreas.lateBlocksVal.bounds = [0, -1];
        this.hitAreas.lateBlocksInc.bounds = [0, -1];
        this.hitAreas.lateBlocksAny.bounds = [0, -1];
        this.hitAreas.textDec.bounds = [0, -1];
        this.hitAreas.textVal.bounds = [0, -1];
        this.hitAreas.textInc.bounds = [0, -1];
        this.hitAreas.textAny.bounds = [0, -1];
        this.hitAreas.othersDec.bounds = [0, -1];
        this.hitAreas.othersVal.bounds = [0, -1];
        this.hitAreas.othersInc.bounds = [0, -1];
        this.hitAreas.othersAny.bounds = [0, -1];
    }
    syncStrengthMode(node) {
        const currentShowModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
        if (this.showModelAndClip !== currentShowModelAndClip) {
            const oldShowModelAndClip = this.showModelAndClip;
            this.showModelAndClip = currentShowModelAndClip;
            if (this.showModelAndClip) {
                if (oldShowModelAndClip != null) {
                    this.value.strengthTwo = this.value.strength ?? 1;
                }
            }
            else {
                this.value.strengthTwo = null;
                this.resetHiddenStrengthTwoHitAreas();
            }
        }
    }
    getStrengthTextColor(value) {
        var _b, _c, _d, _e;
        if (((_b = this.loraInfo) === null || _b === void 0 ? void 0 : _b.strengthMax) != null && value > ((_c = this.loraInfo) === null || _c === void 0 ? void 0 : _c.strengthMax)) {
            return "#c66";
        }
        if (((_d = this.loraInfo) === null || _d === void 0 ? void 0 : _d.strengthMin) != null && value < ((_e = this.loraInfo) === null || _e === void 0 ? void 0 : _e.strengthMin)) {
            return "#c66";
        }
        return undefined;
    }
    setNumberHitAreas(baseName, leftArrow, text, rightArrow, posY, height) {
        this.hitAreas[`${baseName}Dec`].bounds = [leftArrow[0], posY, leftArrow[1], height];
        this.hitAreas[`${baseName}Val`].bounds = [text[0], posY, text[1], height];
        this.hitAreas[`${baseName}Inc`].bounds = [rightArrow[0], posY, rightArrow[1], height];
        this.hitAreas[`${baseName}Any`].bounds = [leftArrow[0], posY, rightArrow[0] + rightArrow[1] - leftArrow[0], height];
    }
    drawInlineNumberControl(ctx, baseName, label, value, posX, posY, height, textColor) {
        const labelGap = LORA_INLINE_LABEL_GAP;
        const controlGap = LORA_INLINE_CONTROL_GAP;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, posX, posY + height * 0.5);
        posX += ctx.measureText(label).width + labelGap;
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
            posX,
            posY,
            height,
            value,
            textColor,
        });
        this.setNumberHitAreas(baseName, leftArrow, text, rightArrow, posY, height);
        return rightArrow[0] + rightArrow[1] + controlGap;
    }
    draw(ctx, node, w, posY, height) {
        var _b, _c;
        this.syncStrengthMode(node);
        const layout = this.getLayoutMetrics();
        const blockWeightsEnabled = this.areBlockWeightsEnabled();
        const widgetHeight = Math.max(height, layout.totalHeight);
        ctx.save();
        const margin = LORA_WIDGET_MARGIN;
        const innerMargin = LORA_WIDGET_INNER_MARGIN;
        const lowQuality = isLowQuality();
        const topRowY = posY + layout.paddingY;
        const middleRowY = topRowY + layout.topRowHeight + layout.rowGap;
        const bottomRowY = middleRowY + layout.detailRowHeight + layout.rowGap;
        const topMidY = topRowY + layout.topRowHeight * 0.5;
        const removeButtonSize = layout.topRowHeight * 0.7;
        const removeButtonX = node.size[0] - margin - removeButtonSize;
        const removeButtonY = posY + (widgetHeight - removeButtonSize) * 0.5;
        let posX = margin;
        ctx.save();
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.beginPath();
        ctx.roundRect(posX, posY, node.size[0] - margin * 2, widgetHeight, [0]);
        ctx.fill();
        if (!lowQuality) {
            ctx.stroke();
        }
        ctx.restore();
        const toggleBounds = drawTogglePart(ctx, { posX, posY: topRowY, height: layout.topRowHeight, value: this.value.on });
        this.hitAreas.toggle.bounds = [toggleBounds[0], topRowY, toggleBounds[1], layout.topRowHeight];
        posX += toggleBounds[1] + innerMargin;
        this.hitAreas.remove.bounds = [removeButtonX, removeButtonY, removeButtonSize, removeButtonSize];
        if (!this.value.on) {
            ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        let rposX = removeButtonX - innerMargin;
        const clipStrengthValue = this.showModelAndClip
            ? ((_b = this.value.strengthTwo) !== null && _b !== void 0 ? _b : 1)
            : ((_c = this.value.strength) !== null && _c !== void 0 ? _c : 1);
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
            posX: rposX,
            posY: topRowY,
            height: layout.topRowHeight,
            value: clipStrengthValue,
            direction: -1,
            textColor: this.getStrengthTextColor(clipStrengthValue),
        });
        if (this.showModelAndClip) {
            this.setNumberHitAreas("strengthTwo", leftArrow, text, rightArrow, topRowY, layout.topRowHeight);
        }
        else {
            this.setNumberHitAreas("strength", leftArrow, text, rightArrow, topRowY, layout.topRowHeight);
        }
        rposX = leftArrow[0] - innerMargin;
        if (this.showModelAndClip) {
            rposX -= innerMargin;
            const modelStrengthValue = this.value.strength ?? 1;
            const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
                posX: rposX,
                posY: topRowY,
                height: layout.topRowHeight,
                value: modelStrengthValue,
                direction: -1,
                textColor: this.getStrengthTextColor(modelStrengthValue),
            });
            this.setNumberHitAreas("strength", leftArrow, text, rightArrow, topRowY, layout.topRowHeight);
            rposX = leftArrow[0] - innerMargin;
        }
        else {
            this.resetHiddenStrengthTwoHitAreas();
        }
        const infoIconSize = layout.topRowHeight * 0.66;
        const infoWidth = infoIconSize + innerMargin + innerMargin;
        if (this.hitAreas["info"]) {
            rposX -= innerMargin;
            drawInfoIcon(ctx, rposX - infoIconSize, topRowY + (layout.topRowHeight - infoIconSize) / 2, infoIconSize);
            this.hitAreas.info.bounds = [rposX - infoIconSize, topRowY, infoWidth, layout.topRowHeight];
            rposX = rposX - infoIconSize - innerMargin;
        }
        const loraWidth = rposX - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const loraLabel = String((this.value === null || this.value === void 0 ? void 0 : this.value.lora) || "None");
        ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, topMidY);
        this.hitAreas.lora.bounds = [posX, topRowY, loraWidth, layout.topRowHeight];
        ctx.save();
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.fillStyle = "#b33";
        ctx.beginPath();
        ctx.roundRect(removeButtonX, removeButtonY, removeButtonSize, removeButtonSize, [Math.max(3, removeButtonSize * 0.18)]);
        ctx.fill();
        ctx.strokeStyle = "#611";
        ctx.lineWidth = 1;
        ctx.stroke();
        const removeInset = removeButtonSize * 0.3;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(removeButtonX + removeInset, removeButtonY + removeInset);
        ctx.lineTo(removeButtonX + removeButtonSize - removeInset, removeButtonY + removeButtonSize - removeInset);
        ctx.moveTo(removeButtonX + removeInset, removeButtonY + removeButtonSize - removeInset);
        ctx.lineTo(removeButtonX + removeButtonSize - removeInset, removeButtonY + removeInset);
        ctx.stroke();
        ctx.restore();
        if (!lowQuality && blockWeightsEnabled) {
            ctx.globalAlpha = app.canvas.editor_alpha * (this.value.on ? 0.8 : 0.45);
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
            let detailPosX = margin + 22;
            detailPosX = this.drawInlineNumberControl(ctx, "earlyBlocks", "Low", this.value.early_blocks ?? 1, detailPosX, middleRowY, layout.detailRowHeight);
            detailPosX = this.drawInlineNumberControl(ctx, "midBlocks", "Mid", this.value.mid_blocks ?? 1, detailPosX, middleRowY, layout.detailRowHeight);
            this.drawInlineNumberControl(ctx, "lateBlocks", "Late", this.value.late_blocks ?? 1, detailPosX, middleRowY, layout.detailRowHeight);
            detailPosX = margin + 22;
            detailPosX = this.drawInlineNumberControl(ctx, "text", "Text", this.value.text ?? 1, detailPosX, bottomRowY, layout.detailRowHeight);
            this.drawInlineNumberControl(ctx, "others", "Others", this.value.others ?? 1, detailPosX, bottomRowY, layout.detailRowHeight);
        }
        else {
            this.resetBlockWeightHitAreas();
        }
        ctx.restore();
    }
    serializeValue(node, index) {
        var _b;
        const v = { ...this.value };
        v.block_weights_enabled = this.areBlockWeightsEnabled();
        if (!this.showModelAndClip) {
            delete v.strengthTwo;
        }
        else {
            this.value.strengthTwo = (_b = this.value.strengthTwo) !== null && _b !== void 0 ? _b : 1;
            v.strengthTwo = this.value.strengthTwo;
        }
        if (!this.areBlockWeightsEnabled()) {
            delete v.early_blocks;
            delete v.mid_blocks;
            delete v.late_blocks;
            delete v.text;
            delete v.others;
        }
        return v;
    }
    onToggleDown(event, pos, node) {
        this.value.on = !this.value.on;
        this.cancelMouseDown();
        return true;
    }
    onInfoDown(event, pos, node) {
        this.showLoraInfoDialog();
    }
    onLoraClick(event, pos, node) {
        showLoraChooser(event, (value) => {
            if (typeof value === "string") {
                this.value.lora = value;
                this.loraInfo = null;
                this.getLoraInfo();
            }
            node.setDirtyCanvas(true, true);
        });
        this.cancelMouseDown();
    }
    onRemoveClick(event, pos, node) {
        removeArrayItem(node.widgets, this);
        const computed = node.computeSize();
        node.size[0] = Math.max(node.size[0], computed[0]);
        node.size[1] = Math.max((node._tempHeight !== null && node._tempHeight !== void 0 ? node._tempHeight : 15), computed[1]);
        node.setDirtyCanvas(true, true);
        this.cancelMouseDown();
        return true;
    }
    onStrengthDecDown(event, pos, node) {
        this.stepStrength(-1, false);
    }
    onStrengthIncDown(event, pos, node) {
        this.stepStrength(1, false);
    }
    onStrengthTwoDecDown(event, pos, node) {
        this.stepStrength(-1, true);
    }
    onStrengthTwoIncDown(event, pos, node) {
        this.stepStrength(1, true);
    }
    onStrengthAnyMove(event, pos, node) {
        this.doOnStrengthAnyMove(event, false);
    }
    onStrengthTwoAnyMove(event, pos, node) {
        this.doOnStrengthAnyMove(event, true);
    }
    onNumericDecDown(prop) {
        this.stepStrengthByProperty(prop, -1);
    }
    onNumericIncDown(prop) {
        this.stepStrengthByProperty(prop, 1);
    }
    onNumericAnyMove(event, prop) {
        if (event.deltaX) {
            this.haveMouseMovedStrength = true;
            this.value[prop] = (this.value[prop] ?? 1) + event.deltaX * 0.05;
        }
    }
    onNumericValUp(event, prop) {
        if (this.haveMouseMovedStrength)
            return;
        const canvas = app.canvas;
        canvas.prompt("Value", this.value[prop], (v) => (this.value[prop] = Number(v)), event);
    }
    doOnStrengthAnyMove(event, isTwo = false) {
        var _b;
        if (event.deltaX) {
            let prop = isTwo ? "strengthTwo" : "strength";
            this.haveMouseMovedStrength = true;
            this.value[prop] = ((_b = this.value[prop]) !== null && _b !== void 0 ? _b : 1) + event.deltaX * 0.05;
        }
    }
    onStrengthValUp(event, pos, node) {
        this.doOnStrengthValUp(event, false);
    }
    onStrengthTwoValUp(event, pos, node) {
        this.doOnStrengthValUp(event, true);
    }
    doOnStrengthValUp(event, isTwo = false) {
        if (this.haveMouseMovedStrength)
            return;
        let prop = isTwo ? "strengthTwo" : "strength";
        const canvas = app.canvas;
        canvas.prompt("Value", this.value[prop], (v) => (this.value[prop] = Number(v)), event);
    }
    onMouseUp(event, pos, node) {
        super.onMouseUp(event, pos, node);
        this.haveMouseMovedStrength = false;
    }
    showLoraInfoDialog() {
        if (!this.value.lora || this.value.lora === "None") {
            return;
        }
        const infoDialog = new RgthreeLoraInfoDialog(this.value.lora).show();
        infoDialog.addEventListener("close", ((e) => {
            if (e.detail.dirty) {
                this.getLoraInfo(true);
            }
        }));
    }
    stepStrength(direction, isTwo = false) {
        var _b;
        let step = 0.05;
        let prop = isTwo ? "strengthTwo" : "strength";
        let strength = ((_b = this.value[prop]) !== null && _b !== void 0 ? _b : 1) + step * direction;
        this.value[prop] = Math.round(strength * 100) / 100;
    }
    stepStrengthByProperty(prop, direction) {
        const step = 0.05;
        const strength = (this.value[prop] ?? 1) + step * direction;
        this.value[prop] = Math.round(strength * 100) / 100;
    }
    getLoraInfo(force = false) {
        if (!this.loraInfoPromise || force == true) {
            let promise;
            if (this.value.lora && this.value.lora != "None") {
                promise = LORA_INFO_SERVICE.getInfo(this.value.lora, force, true);
            }
            else {
                promise = Promise.resolve(null);
            }
            this.loraInfoPromise = promise.then((v) => (this.loraInfo = v));
        }
        return this.loraInfoPromise;
    }
}
const NODE_CLASS = AnzhcPowerLoraLoader;
app.registerExtension({
    name: "anzhc.PowerLoraLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_CLASS.type) {
            NODE_CLASS.setUp(nodeType, nodeData);
        }
    },
});
