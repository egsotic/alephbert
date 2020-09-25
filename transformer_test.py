from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("models/bert-wordpiece-v1")
tokenizer = AutoTokenizer.from_pretrained("models/bert-wordpiece-v1")
fill_mask = pipeline(
    "fill-mask",
    # model=f"./models/bert-wordpiece-v1",
    model=model,
    # tokenizer=f"./models/bert-wordpiece-v1"
    tokenizer=tokenizer
)

result = fill_mask("קוראים לי [MASK].")
print(result)
