from transformers import ChemformerModel, ChemformerTokenizer

tokenizer = ChemformerTokenizer.from_pretrained("Chemformer")
model = ChemformerModel.from_pretrained("Chemformer")

# Example input
smiles = "CCO"
inputs = tokenizer(smiles, return_tensors="pt")
outputs = model(**inputs)

# Model outputs
print(outputs)
