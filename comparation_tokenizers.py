from transformers import AutoTokenizer,PreTrainedTokenizerFast

print("TOKENIZADOR BETO")

model_name = "dccuchile/bert-base-spanish-wwm-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


texto = "Me siento con mis preciosos brazos de niña envuelta tan bien alrededor de mi cuello"

inputs = tokenizer(texto, return_tensors="pt")

print(inputs)
print("🔢 IDs:", inputs["input_ids"][0])
print("🧩 Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("🗣️ Decodificado:", tokenizer.decode(inputs["input_ids"][0]))

print("TOKENIZADOR PROPIO")
own_tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer")

tokens= own_tokenizer(texto)
print("🔢 IDs:", tokens["input_ids"])
print("🧩 Tokens:", own_tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
print("🗣️ Decodificado:", own_tokenizer.decode(tokens["input_ids"]))


#Tras investigar se decide utilizar el tokenizador de Beto ya que tiene la misma estructura que BERT pero enfocada en español ademas
# de ser entrenada tambien por el corpus que se va a utilizar Wikipedia + OPUS + OpenSubtitles (corpus ~3B tokens en español)
#Volviendolo mas completa que el tokenizador propio