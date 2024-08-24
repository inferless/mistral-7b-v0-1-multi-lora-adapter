from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import os

HF_TOKEN = os.environ.get("HF_TOKEN")

class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct").to("cuda")
        self.model = PeftModel.from_pretrained(base_model,"ushuradmin/usrs-extractions-adapteronly-16_bit_llama3.1_v1", adapter_name="extract", token=HF_TOKEN)
        self.model.load_adapter("ushuradmin/usrs-classification-adapteronly-16_bit_llama3.1_v1", adapter_name="classify", token=HF_TOKEN)

    def infer(self, inputs):
        prompt = inputs["prompt"]
        adapter_name = inputs.pop("adapter_name")
        temperature = inputs.get("temperature",0.7)
        repetition_penalty = float(inputs.get("repetition_penalty",1.18))
        max_new_tokens = inputs.get("max_new_tokens",1024)
        
        if (self.model.active_adapter) != adapter_name:
            self.model.set_adapter(adapter_name)

        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        result = self.tokenizer.decode(self.model.generate(**model_input,temperature=temperature, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)[0], skip_special_tokens=True)

        return {'generated_result': result}

    def finalize(self):
        self.model = None
