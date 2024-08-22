from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel


class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        base_model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-7b").to("cuda")
        self.model = PeftModel.from_pretrained(base_model,"philschmid/gemma-7b-dolly-chatml", adapter_name="french")
        self.model.load_adapter("ushuradmin/usrs_extractions_4b_adapter_only_v1", adapter_name="sql")
        self.model.load_adapter("ushuradmin/usrs_classification_adapter_only", adapter_name="dpo")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        adapter_name = inputs.pop("adapter_name")
        temperature = inputs.get("temperature",0.7)
        repetition_penalty = float(inputs.get("repetition_penalty",1.18))
        max_new_tokens = inputs.get("max_new_tokens",128)
        
        if (self.model.active_adapter) != adapter_name:
            self.model.set_adapter(adapter_name)

        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        result = self.tokenizer.decode(self.model.generate(**model_input,temperature=temperature, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)[0], skip_special_tokens=True)

        return {'generated_result': result}

    def finalize(self):
        self.model = None
