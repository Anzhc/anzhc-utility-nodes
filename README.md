# Anzhc Utility Nodes

Just whatever nodes i need to make generation convenient and not suck so much.

Opened because Talan wanted lora loader node. Don't expect any of them to keep working, this is not actively maintained repo, and will not be guaranteed to work even on my machine.

## Nodes

| Node | Description |
| --- | --- |
| `Wildcards (Anzhc)` | Allows to do `__wildcard-name__` just like in good old A1111. |
| `Block Noise (Anzhc)` | Ignore |
| `Block Noise Debug (Anzhc)` | Ignore |
| `No-Noise KSampler (Anzhc)` | Ignore |
| `Trigger Tag Replace (Anzhc)` | Uses lora names (like ones from WAS lora loader, or mine) and searches for `(trigger)` in name, i.e.: `lora name(trigger) lycoris.safetensors`, if passed text has `TRIGGER` in it, that position will be replaced with all triggers that are found. |
| `Lora Loader (Anzhc)` | Mixture of rgthree and WAS loaders, with strongly extended functionality. Supports grouped parameter weighting (targetting Anima loras only) and on-the-fly merging (Only SVC method for now(it's just a bit better than generic stacking))|
| `Resolution (Anzhc)` | Are you tired of writing 832 and 1216 again and again to swap from portrait to landscape? |
| `LM Studio LLM (Anzhc)` | Allows for usage of locally hosted LLMs from LM studio. Minimal node, i use it just as prompt enhancer. |
| `LMStudio Pipeline (Anzhc)` | Runs tiered multi-agent workflows from subfolders in `LLM templates`, passing declared dependency outputs to lower-tier agents and returning the final tier-0 response. |
| `LM Studio Gacha (Anzhc)` | Expands image requests with adaptive model-planned visual decisions and seeded random option selection in one conversation. Editable system and user prompts default to Anima tags and description. Defaults to six decisions, 32 options per decision, eight overall concepts, and eight local LM Studio calls. |
| `MCP Skills (Anzhc)` | Haven't tested, do whatever you want. |
| `WDv3 Tagger (Anzhc)` | Haven't actually tested either. Basically plan is to use this for automatic prompting for adetailing, because im lazy. |
| `Autotagged Tile Conditioning (Anzhc)` | Patches a model for Tiled Diffusion, tags each tile with WDv3, and injects tile-specific positive prompt conditioning while keeping negative conditioning universal. Returns the patched model, global positive conditioning, and tile prompts for inspection. |

## Dependency Notes

- `rgthree-comfy` is needed for the custom frontend behavior of `Lora Loader (Anzhc)`.
- `LM Studio LLM (Anzhc)` expects LM Studio to be running locally at `http://127.0.0.1:1234`.
- `LM Studio Gacha (Anzhc)` uses the same local server and its JSON-schema output support. Its token limit applies to each call, including reasoning. The seed controls option selection; identical captions also depend on deterministic model sampling. Based on [Gacha Decoding](https://arxiv.org/abs/2610.01382), Appendix B.
- Gacha's `system_prompt` applies to every round; `user_prompt` supplies the initial request retained in the conversation. Both replace `<text_input>` with the source text and `<node_info>` with width, height, and aspect ratio. Each DECISION chooses a new question from the request and previous selections and returns `{name, ask, options}` JSON. Python selects an option, then the next round chooses what to elaborate next. All configured decision rounds run, followed by CONCEPTS and FINAL, for `decision_steps + 2` calls. Output formatting rules should apply to FINAL only. The default system prompt includes these stage instructions.
- Gacha's `response` output is the final prompt. Its `rounds` text output contains every user message, model response (including all options), and random selection in order, excluding the system prompt.
- Gacha's optional `image` input accepts a reference image or batch and attaches it to every round. Use a vision-capable LM Studio model. For image-to-video prompting, set the system and user prompts to your video instructions and output format; the defaults target Anima image prompts.
- `MCP Skills (Anzhc)` uses the `mcp` package listed in `requirements.txt`.
- `WDv3 Tagger (Anzhc)` requires `onnxruntime` and `huggingface_hub`. The WDv3 model files are downloaded from Hugging Face on first use.
- `Autotagged Tile Conditioning (Anzhc)` requires `ComfyUI-TiledDiffusion` plus the WDv3 Tagger dependencies above. `SpotDiffusion` is not supported because its shifted windows do not map cleanly to static tile tags.

## Acknowledgements

- https://github.com/WASasquatch/was-node-suite-comfyui
- https://github.com/rgthree/rgthree-comfy

Both repositories were used as references for the LoRA Loader.
Utils imported directly from rgthree node, you'll need to have it if you want lora loader to work.
