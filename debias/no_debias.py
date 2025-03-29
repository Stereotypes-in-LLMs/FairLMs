from pydantic import BaseModel
from typing import Any

class PlainModelWrapper(BaseModel):
    model: Any
    model_name: str
    tokenizer: Any

    def debias_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_embeddings = self.model.get_input_embeddings()(inputs['input_ids'].to("cuda"))
        return inputs['attention_mask'], token_embeddings

    
    def forward(self, text, output_hidden_states: False, number_eos = 0, **kwargs):
        eos_token = self.tokenizer.eos_token

        if eos_token is None:
            raise ValueError("This tokenizer does not have an eos_token defined.")

        text = text + eos_token * number_eos
        
        modified_embeddings = self.debias_embeddings(text)
        outputs = self.model(attention_mask=modified_embeddings[0],
                    inputs_embeds=modified_embeddings[1], output_hidden_states = output_hidden_states, disable_tqdm=True)
            
        return outputs
    
    def __call__(self, tokens, output_hidden_states: bool = False, number_eos = 0, **kwargs):
        return self.forward(tokens,output_hidden_states = output_hidden_states, number_eos = number_eos, **kwargs)