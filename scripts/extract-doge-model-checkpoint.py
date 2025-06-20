from safetensors.torch import load_file, save_file
import os
from fire import Fire
from loguru import logger

def extract_doge_model(model_dir: str):
    if not os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        logger.info(f"No model.safetensors.index.json in {model_dir}, skipping")
        return
    
    # then load model-*-of-*.safetensors
    state_dict = {}
    os.makedirs(os.path.join(model_dir, 'backup'), exist_ok=True)
    
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors"):
            logger.info(f"Loading {file}...")
            local_state_dict = load_file(os.path.join(model_dir, file))
            for key in local_state_dict.keys():
                if "teacher_model" in key:
                    state_dict[key.split("teacher_model.")[1]] = local_state_dict[key]
            logger.info(f"Moving {file} to {os.path.join(model_dir, 'backup')}")
            os.rename(os.path.join(model_dir, file), os.path.join(model_dir, 'backup', file))
        elif file == "model.safetensors.index.json":
            logger.info(f"Moving {file} to {os.path.join(model_dir, 'backup')}")
            os.rename(os.path.join(model_dir, file), os.path.join(model_dir, 'backup', file))
    
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        os.remove(os.path.join(model_dir, "model.safetensors"))
    logger.info(f"Saving to {os.path.join(model_dir, 'model.safetensors')}")
    save_file(state_dict, os.path.join(model_dir, "model.safetensors"))
    


    
if __name__ == "__main__":
    Fire(extract_doge_model)
