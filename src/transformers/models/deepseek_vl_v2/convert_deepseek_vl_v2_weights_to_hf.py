import argparse
import gc
import json
import os
from typing import Optional

import regex as re
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    DeepseekVLV2Config,
    DeepseekVLV2ForCausalLM,
    DeepseekVLV2ImageProcessor,
    DeepseekVLV2Processor,
)
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Siglip Vision Encoder
    r"vision.pos_embed":                                  r"model.vision_model.vision_model.embeddings.position_embedding.weight",
    r"vision.patch_embed.proj.(weight|bias)":             r"model.vision_model.vision_model.embeddings.patch_embedding.\1",
    r"vision.blocks.(\d+).attn.qkv.(weight|bias)":        r"model.vision_model.vision_model.encoder.layers.\1.self_attn.(q|k|v)_proj.\2",
    r"vision.blocks.(\d+).attn.proj.(weight|bias)":       r"model.vision_model.vision_model.encoder.layers.\1.self_attn.out_proj.\2",
    r"vision.blocks.(\d+).norm(\d+).(weight|bias)":       r"model.vision_model.vision_model.encoder.layers.\1.layer_norm\2.\3",
    r"vision.blocks.(\d+).mlp.fc(\d+).(weight|bias)":     r"model.vision_model.vision_model.encoder.layers.\1.mlp.fc\2.\3",
    r"vision.norm.(weight|bias)":                         r"model.vision_model.vision_model.post_layernorm.\1",
    r"vision.attn_pool.latent":                           r"model.vision_model.vision_model.head.probe",
    r"vision.attn_pool.proj.(weight|bias)":               r"model.vision_model.vision_model.head.attention.out_proj.\1",
    r"vision.attn_pool.norm.(weight|bias)":               r"model.vision_model.vision_model.head.layernorm.\1",
    r"vision.attn_pool.mlp.fc(\d+).(weight|bias)":        r"model.vision_model.vision_model.head.mlp.fc\1.\2",

    # Projector
    r"projector.layers.0.(weight|bias)":               r"model.projector.layers.0.\1",
    r"projector.layers.2.(weight|bias)":               r"model.projector.layers.2.\1",

    # Deepseek V2 (Text Model)
    r"language.model.(\w+)":                   r"model.language_model.\1",
    r"language.lm_head.(weight|bias)":         r"model.lm_head.\1",
    r"image_newline":                          r"model.image_newline",
    r"view_seperator":                         r"model.view_seperator",
}
# fmt: on

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|sft▁begin|>\n"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "<image>"
    "{% elif content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}\n"
    "<|sft▁end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'][0]['text'] if message['content'] and message['content']|length > 0 else '' }}"
    "<|end_of_sentence|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|sft▁begin|>\n"
    "{% endif %}"
)


def convert_old_keys_to_new_keys(state_dict_keys: dict):
    output_dict = {}

    old_text = "\n".join(state_dict_keys)
    new_text = old_text
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        if replacement is None:
            new_text = re.sub(pattern, "", new_text)  # an empty line
            continue
        new_text = re.sub(pattern, replacement, new_text)
    output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))

    return output_dict


def get_qkv_state_dict(key, parameter):
    qkv_state_dict = {}
    placeholder = re.search(r"(\(.*?\))", key).group(1)  # finds   "(query|key|value)"
    replacements_keys = placeholder[1:-1].split(
        "|"
    )  # creates ['query', 'key', 'value']
    replacements_vals = torch.split(
        parameter,
        split_size_or_sections=parameter.size(0) // len(replacements_keys),
        dim=0,
    )
    for replacement_key, replacement_val in zip(replacements_keys, replacements_vals):
        qkv_state_dict[key.replace(placeholder, replacement_key)] = replacement_val
    return qkv_state_dict


def update_state_dict(old_state_dict):
    all_keys = list(old_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        current_parameter = old_state_dict.pop(key)

        if "qkv" in key and "vision_tower_high" not in key:
            qkv_state_dict = get_qkv_state_dict(new_key, current_parameter)
            state_dict.update(qkv_state_dict)
        elif "pos_embed" in key:
            if "vision_tower_high" not in key:
                # timm implementation of siglip creates this param of size [1, 576, 1024]
                # transformers implementation of siglip creates this param of size [576, 1024]
                state_dict[new_key] = current_parameter.squeeze(0)
            else:
                state_dict[new_key] = current_parameter
        else:
            state_dict[new_key] = current_parameter

    return state_dict


def load_model_state_dict(input_path: str) -> dict:
    """
    Load model state dict, handling both single and sharded files.
    """
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    single_file_path = os.path.join(input_path, "model.safetensors")

    # Check if we have a sharded model
    if os.path.exists(index_path):
        print("Loading sharded model...")
        state_dict = {}
        with open(index_path, "r") as f:
            index = json.load(f)

        # Get unique shard files and load each one only once
        unique_shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in unique_shard_files:
            print(f"Loading shard {shard_file}...")
            shard_path = os.path.join(input_path, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        return state_dict

    # Single file model
    elif os.path.exists(single_file_path):
        print("Loading single file model...")
        return load_file(single_file_path, device="cpu")

    else:
        raise ValueError(f"No model files found in {input_path}")


def convert_model(
    hf_repo_id: str,
    output_dir: Optional[str] = None,
    output_hub_path: Optional[str] = None,
    safe_serialization: bool = True,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        input_path = snapshot_download(hf_repo_id)
    except HFValidationError:
        # If the input path is not a HF repo ID, assume it's a local path
        input_path = hf_repo_id

    image_processor = DeepseekVLV2ImageProcessor(
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
    )

    tokenizer = AutoTokenizer.from_pretrained(input_path)

    special_tokens = ["<|User|>", "<|Assistant|>"]

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    processor = DeepseekVLV2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    if output_dir:
        print(f"Saving processor to {output_dir}...")
        processor.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing processor to hub at {output_hub_path}...")
        processor.push_to_hub(output_hub_path)

    # Load and convert state dict
    print("Loading state dict...")
    state_dict = load_model_state_dict(input_path)
    state_dict = update_state_dict(state_dict)

    vision_width = state_dict[
        "model.vision_model.vision_model.embeddings.patch_embedding.weight"
    ].shape[0]

    lang_hidden = state_dict["model.image_newline"].shape[0]  # 2048

    vocab_size = state_dict["model.language_model.embed_tokens.weight"].shape[0]
    projector_output = state_dict["model.projector.layers.0.weight"].shape[0]
    text_config = DeepseekV2Config(
        vocab_size=129280,
        hidden_size=1280,
        intermediate_size=6848,
        num_hidden_layers=12,
        # attention
        num_attention_heads=40,
        num_key_value_heads=4,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        kv_lora_rank=512,
        use_mla=False,
        n_shared_experts=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        moe_layer_freq=None,
        first_k_dense_replace=12,
    )
    config = DeepseekVLV2Config(
        candidate_resolutions=[
            [384, 384],
            [384, 768],
            [768, 384],
            [384, 1152],
            [1152, 384],
            [384, 1536],
            [1536, 384],
            [768, 768],
            [384, 1920],
            [1920, 384],
            [384, 2304],
            [2304, 384],
            [768, 1152],
            [1152, 768],
            [384, 2688],
            [2688, 384],
            [384, 3072],
            [3072, 384],
            [768, 1536],
            [1536, 768],
            [384, 3456],
            [3456, 384],
            [1152, 1152],
        ],
        global_view_pos="head",
        text_config=text_config,
        model_type="deepseek_vl_v2",
        projector_config={
            "model_type": "mlp_projector",
            "n_embed": projector_output,
        },
        tile_tag="2D",
        torch_dtype="bfloat16",
        vision_config={
            "layers": 27,
            "mlp_ratio": 3.7362,
            "model_name": "siglip_so400m_patch14_384",
            "model_type": "siglip_vision_model",
            "patch_size": 14,
            "width": 1152,
            "hidden_size": vision_width,
            "intermediate_size": int(vision_width * 3.7362),
            "image_size": 384,
        },
    )
    if output_dir:
        config.save_pretrained(output_dir)
        print("Model config saved successfully...")

    print("Creating empty model...")

    model = DeepseekVLV2ForCausalLM(config)
    # Load converted state dict
    print("Loading converted weights into model...")
    info = model.load_state_dict(state_dict, strict=False, assign=True)

    # Tie weights before any device mapping
    print("Tying weights...")
    model.tie_weights()

    # Save the model
    if output_dir:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    if output_hub_path:
        print(f"Pushing model to hub at {output_hub_path}...")
        model.push_to_hub(output_hub_path, safe_serialization=safe_serialization)

    del state_dict, model
    gc.collect()

    # Validate the saved model if saved locally
    if output_dir:
        print("Reloading the local model to check if it's saved correctly...")
        DeepseekVLV2ForCausalLM.from_pretrained(output_dir, device_map="auto")
        print("Local model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default="deepseek-ai/deepseek-vl2-tiny",
        help="Location of official weights from DeepseekAI on HF",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Location to write the converted model and processor",
    )
    parser.add_argument(
        "--output_hub_path",
        default=None,
        help="Repository ID to push model to hub (e.g. 'username/model-name')",
    )
    parser.add_argument(
        "--safe_serialization",
        default=True,
        type=bool,
        help="Whether or not to save using `safetensors`.",
    )
    args = parser.parse_args()

    convert_model(
        hf_repo_id=args.hf_repo_id,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
