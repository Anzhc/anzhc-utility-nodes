import json
import random

from .lm_studio_llm_node import (
    DEFAULT_MODEL_OPTION,
    LM_STUDIO_MODELS_URL,
    call_lm_studio_chat,
    get_lm_studio_model_options,
    image_input_to_data_urls,
    inject_node_info,
    inject_text_input,
)


DEFAULT_SYSTEM_PROMPT = """You create image prompts for Anima. Use the user's input to design a complete image, including scene choices they leave open. For the final output, return a tag line and a plain description of that image. Every phrase must specify something to draw.

READ THE REQUEST
Input can contain scene content, instructions to you, or both. Apply instructions about creating, changing, expanding, or arranging the image. Do not turn those instructions into depicted features, tags, or commentary. Translate broad artistic requests into concrete visual choices. Copy wording into the image only when it is requested as on-image text.
Treat tags and prose as a draft with equal status. Explicit requested changes replace older details. Node context fills gaps. Keep the requested subject, count, defining features, central interaction, and explicit limits. Unspecified details are yours to choose; omission is not a request to leave them empty.
For a revision, carry the change through the whole scene and both output sections. Rework dependent pose, objects, and framing when needed. Resolve conflicts through unspecified choices first. If explicit requirements still conflict, retain the central request and drop the least central detail. Do not explain the correction.

BUILD THE SCENE
Actively choose the missing setting, viewpoint, crop, pose, arrangement, and lighting. Build a scene around the user's idea instead of only adding attributes to its existing nouns. Make these choices even when the user does not separately ask for composition or environment design. Commit to one result without offering alternatives.
Build the composition around a specific relationship between the subject and the surroundings. Choose the place, pose, and view together so the main interaction and important features fit. Use surfaces, spacing, overlap, and depth to arrange them. When activity is open, invent a suitable action or interaction and show its physical effects in this instant.
You may invent supporting objects, environmental conditions, and setting features. Each addition must establish the place, support the interaction, or shape the view through its position. Describe that relationship directly. Do not substitute lists of supplies, worn surfaces, vague clutter, or unnamed background shapes for scene design. Remove additions that do no useful work in the composition. Creative freedom concerns what is drawn; the wording stays literal.
Scale invention to the request. Sparse, open input invites substantial scene design. Detailed input calls for choices in the remaining gaps. Respect explicit requests for idleness, minimal or empty scenes, abstraction, and fixed backgrounds or framing. Do not replace distinctive requested features with familiar defaults or force novelty, activity, or depth layers into every image.

ONE IMAGE
All parts and actions must coexist in one frame. Give each trait and object a clear owner. Check occupied hands, support, contact, scale, and occlusion. A fixed crop excludes secondary details outside it; do not force excluded body parts into the view.
Describe motion through the current pose and physical effects. Do not narrate earlier or later events, hidden causes, or alternate interpretations. Keep intentional fantasy, unusual anatomy, and stylization. Anime and illustration are the default and need no added labels. Use other style or camera terms when requested or needed to specify the view.

PROTECTED TEXT
Copy text before ||| exactly unless the user asks to edit that prefix. Remove the gate. At the join, remove extra commas and spaces only from the start of the editable suffix; add a separator if needed. The prefix is exempt from spelling and ordering rules.
Keep supplied artist, trigger, quality, and style tags with their exact spelling unless explicitly changed. Do not invent metadata or put it in the description. Preserve supplied weights on the same concepts. Remove a weighted concept only when an explicit request supersedes it. Add or change weights only when explicitly asked.

TAGS
Rebuild the editable tags from the finished scene, including your important new choices. Replace stale tags, remove contradictions and duplicates, merge traits of the same item, and add missing concepts. The tags must develop along with the description.
Use compact names or descriptive phrases, not sentences. Use spaces instead of underscores in descriptive tags. Keep count-tag syntax, with the number joined directly to the subject type. Give one total per subject type; different types add together.
Order: supplied metadata; subject count/type and identity; traits, face, and hair; body and pose; clothing; interaction; objects; setting; lighting; viewpoint and composition. Skip unused categories.

DESCRIPTION
Start with the subject or scene. Use everyday names, direct verbs, and short statements of physical facts. Keep the same name for the same thing. Bind attributes to owners and state positions, contacts, orientations, and overlaps. Use the present tense. Give useful details enough room without reciting the tags.
Use adjectives for shape, size, color, material, or another visible property. Remove adverbs that judge or intensify actions, judgments, whimsy, metaphors, euphemisms, guesses, and explanations of what a detail adds to the image. Describe the choices you made as definite physical facts. Do not narrate attention, skill, effort, purpose, mood, or invented history. Show requested expression through the face and pose.
A light description must place a source, lit area, or shadow. For depth or blur, name the objects and their positions or sharpness. Do not describe an effect on the viewer, explain what light brings out, or add a generic lighting sentence to finish. Mention obstruction when relevant; otherwise visibility is assumed. Do not list absent things or announce compliance. Preserve requested on-image text exactly in double quotation marks; prose rules do not alter that text.

LENGTH AND OUTPUT
Follow any caption target or maximum with its stated units and scope. Without a scope, it covers both sections. No minimum length, fixed tag count, or padding to fill an API token allowance.
Check that the requested operation was carried out, open scene choices were made, and explicit limits remain intact. Check protected text, physical coherence, and agreement between tags and description. Remove instruction wording, nonvisual commentary, and unnecessary additions. Fine spatial details need not become tags.
For the final output, return only these two blocks, separated by one blank line. No headings, fences, or explanations:
<one comma-separated tag line>

<one paragraph of direct physical descriptions>

GACHA ROUNDS
The node runs several turns of one conversation. Use the original user input and node context throughout. Apply the scene-design and request-preservation rules to every round. The tag, description, and length conventions above apply to the final output.
DECISION: Review the original request, node context, and every selected answer. Choose for yourself the next useful visual detail or relationship to elaborate. Let previous selections guide what needs development next; do not follow a fixed checklist or plan all future questions. Leave explicit requirements and selected answers fixed. You may refine an earlier choice by adding new compatible information. Give this decision a short name and phrase it as a question. Enumerate distinct answers, each consistent with the request and all selected answers. Vary concrete scene choices, not synonyms, and keep each option concise. Return the requested decision JSON containing name, ask, and options. Do not choose an answer yourself; the node selects one randomly and supplies it in the next turn.
CONCEPTS: Enumerate coherent complete scene concepts using all selected answers. Vary concrete arrangements and interactions while retaining the request's fixed details. Return the requested options JSON. The node selects the overall concept.
FINAL: Realize the selected answers and overall concept as one image. Follow the original user instructions and final output conventions. If a selected choice conflicts with an explicit requirement, preserve the requirement. Unselected alternatives are not part of the image.
During DECISION and CONCEPTS, return only the requested JSON. Do not write the final tags or description until FINAL. Never include round labels, planning questions, option menus, or selection commentary in the final output."""

DEFAULT_USER_PROMPT = "Original user input:\n<text_input>\n\nAdditional node context:\n<node_info>"


DECISION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "gacha_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ask": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["name", "ask", "options"],
            "additionalProperties": False,
        },
    },
}

OPTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "gacha_options",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["options"],
            "additionalProperties": False,
        },
    },
}


def _sample_option(response: dict, rng: random.Random, stage: str) -> str:
    options = response["options"]
    if not options:
        raise RuntimeError(f"LM Studio Gacha returned no options for {stage}.")
    return rng.choice(options)


def _generate_gacha(
    *,
    model: str,
    temperature: float,
    seed: int,
    decision_steps: int,
    options_per_step: int,
    output_token_limit: int,
    text_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    width: int = 1024,
    height: int = 1024,
    image_data_urls: list[str] | None = None,
) -> tuple[str, str]:
    rng = random.Random(seed)
    system_prompt = inject_node_info(inject_text_input(system_prompt, text_input), width, height)
    user_request = inject_node_info(inject_text_input(user_prompt, text_input), width, height)
    if image_data_urls:
        user_request += (
            "\n\nReference images are attached to every round. Use them as the starting point "
            "for the user's request. Preserve visible subjects and scene details unless the user "
            "requests changes."
        )
    sampling = {
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": output_token_limit,
        "image_data_urls": image_data_urls,
    }
    turns = []
    rounds = []
    decisions = []
    chosen = ""
    for step in range(decision_steps):
        instruction = (
            f"Gacha round: DECISION {step + 1} of {decision_steps}\n{chosen}"
            "Review the original request, node context, and all selected choices. Choose the next "
            "useful visual detail or relationship you want to elaborate, based on what the scene "
            "currently needs. Add new information without replacing fixed requirements or selected "
            "choices. Give this decision a short name and a question. Do not plan future decisions.\n"
            f"Enumerate at least {options_per_step} meaningfully different, concise answers to your "
            "question, each compatible with the request and every choice already selected.\n"
            'Return JSON: {"name": "...", "ask": "...", "options": ["...", "..."]}.'
        )
        if not turns:
            instruction = f"{user_request}\n\n{instruction}"
        response = call_lm_studio_chat(
            **sampling,
            session_instruction=instruction,
            fake_chat_turns=turns,
            text_input=instruction,
            response_format=DECISION_FORMAT,
        )
        decision = json.loads(response)
        selection = _sample_option(decision, rng, decision["name"])
        turns.append({"user": instruction, "assistant": response})
        rounds.append(
            f"Round {step + 1}: DECISION\n\nUser:\n{instruction}\n\nAssistant:\n{response}\n\n"
            f"Selected {decision['name']}: {selection}"
        )
        decisions.append(f"{decision['name']}: {selection}")
        chosen = f"Selected {decision['name']}: {selection}\n\n"

    instruction = (
        f"Gacha round: CONCEPTS\n{chosen}Develop 8 distinct overall visual concepts that combine every selected choice. "
        "Make each concept concrete and different from the others.\n"
        "Keep every concept consistent with the original user request and node context.\n"
        'Return JSON: {"options": ["...", "..."]}.'
    )
    if not turns:
        instruction = f"{user_request}\n\n{instruction}"
    response = call_lm_studio_chat(
        **sampling,
        session_instruction=instruction,
        fake_chat_turns=turns,
        text_input=instruction,
        response_format=OPTION_FORMAT,
    )
    approach = _sample_option(json.loads(response), rng, "overall concept")
    turns.append({"user": instruction, "assistant": response})
    rounds.append(
        f"Round {decision_steps + 1}: CONCEPTS\n\nUser:\n{instruction}\n\nAssistant:\n{response}\n\n"
        f"Selected overall concept: {approach}"
    )
    decision_text = "\n".join(decisions)
    instruction = (
        f"{user_request}\n\nGacha round: FINAL\n"
        f"Selected visual decisions:\n{decision_text}\n\n"
        f"Overall concept: {approach}\n\n"
        "Produce the final prompt using these choices and the original user request and "
        "node context. Follow the system and user instructions, including their output conventions. "
        "Explicit requirements take precedence over selected choices. Return only the final result."
    )
    response = call_lm_studio_chat(
        **sampling,
        session_instruction=instruction,
        fake_chat_turns=turns,
        text_input=instruction,
    )
    rounds.append(f"Round {decision_steps + 2}: FINAL\n\nUser:\n{instruction}\n\nAssistant:\n{response}")
    return response, "\n\n---\n\n".join(rounds)


class AnzhcLMStudioGacha:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_lm_studio_model_options()
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "Seeds Python's option selection and LM Studio's token sampling."}),
                "decision_steps": ("INT", {"default": 6, "min": 0, "max": 64,
                                           "tooltip": "Number of adaptive visual decisions. The model chooses what to elaborate each round."}),
                "options_per_step": ("INT", {"default": 32, "min": 1, "max": 256,
                                             "tooltip": "Requested candidate count for each visual decision."}),
                "output_token_limit": ("INT", {"default": 8192, "min": 1, "max": 131072,
                                               "tooltip": "Token budget per call, including model reasoning."}),
                "text_input": ("STRING", {"forceInput": True,
                                          "tooltip": "Image request and any constraints to preserve."}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT,
                    "tooltip": "Shared scene-design, Gacha-round, and final-output instructions for every call. Supports <text_input> and <node_info>.",
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_USER_PROMPT,
                    "tooltip": "Initial user message retained throughout the conversation. Supports <text_input> and <node_info>.",
                }),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8,
                                  "tooltip": "Generation width inserted at <node_info>."}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8,
                                   "tooltip": "Generation height inserted at <node_info>."}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image or image batch used in every round. Requires a vision-capable LM Studio model.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "rounds")
    FUNCTION = "generate"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Expands an image request using adaptive Gacha Decoding: chooses a new visual decision "
        "from the request and previous selections each round, randomly "
        "selects model-enumerated options in sequence, samples an overall concept, then writes "
        "the final prompt in one conversation using the same system and user instructions. "
        "Six decision rounds use eight LM Studio calls. <text_input> inserts the source request; "
        "<node_info> inserts generation dimensions and aspect ratio. The rounds output contains "
        "each user message, model response, and random selection, excluding the system prompt. "
        "Optional reference images are attached to every round for image-edit or video prompts."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def generate(
        self, model, temperature, seed, decision_steps, options_per_step, output_token_limit, text_input,
        system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT, width=1024, height=1024,
        image=None,
    ):
        if model == DEFAULT_MODEL_OPTION:
            raise RuntimeError(
                f"No active LM Studio models were found at {LM_STUDIO_MODELS_URL}. "
                "Load a model in LM Studio, then refresh the node."
            )
        return _generate_gacha(
            model=model,
            temperature=float(temperature),
            seed=int(seed),
            decision_steps=int(decision_steps),
            options_per_step=int(options_per_step),
            output_token_limit=int(output_token_limit),
            text_input=str(text_input or ""),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            width=int(width),
            height=int(height),
            image_data_urls=image_input_to_data_urls(image),
        )


NODE_CLASS_MAPPINGS = {
    "Anzhc LM Studio Gacha": AnzhcLMStudioGacha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc LM Studio Gacha": "LM Studio Gacha (Anzhc)",
}
