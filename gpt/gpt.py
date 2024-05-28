import transformers
from typing import List, Dict, Optional, Iterable, Tuple

class GPTWrapper:

    def __init__(self):
        self.model = transformers.GPT2LMHeadModel.from_pretrained("/home/artyom/gpt_hw/checkpoint3000")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")

    def generate(self, input_text, generation_config: Dict):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs.update(generation_config)
        generated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(generated_tokens[0])
        

def construct_model():

    generation_config = {
        "max_new_tokens": 40,
        "num_beams": 2,
        "early_stopping": True,
        "no_repeat_ngram_size": 2
    }
    kwargs = {'generation_config': generation_config}
    model = GPTWrapper()
    return model, kwargs