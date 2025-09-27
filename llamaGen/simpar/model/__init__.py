AVAILABLE_MODELS = {
    "modeling_qwen2": "Qwen2ForCausalLM",
    "simpar_qwen2": "SimpARForCausalLM",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_classes} from llava.language_model.{model_name}. Error: {e}")
