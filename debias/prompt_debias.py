from pydantic import BaseModel
from typing import Any

class PrompDebiasModelWrapper(BaseModel):
    model: Any
    model_name: str
    tokenizer: Any
    debias_prompt:str
    
    def debias_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_embeddings = self.model.get_input_embeddings()(inputs['input_ids'].to("cuda"))
        return inputs['attention_mask'], token_embeddings
    
    def forward(self, text, output_hidden_states: False, **kwargs):
        text = self.debias_prompt + text
        modified_embeddings = self.debias_embeddings(text)
        outputs = self.model(attention_mask=modified_embeddings[0],
                    inputs_embeds=modified_embeddings[1], output_hidden_states = output_hidden_states)
        return outputs
    
    def __call__(self, tokens, output_hidden_states: bool = False, **kwargs):
        return self.forward(tokens,output_hidden_states = output_hidden_states, **kwargs)