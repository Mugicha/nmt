from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                          use_auth_token=True,
                                          src_lang="jpn_Jpan",
                                          )
# print(tokenizer.lang_code_to_id.keys())

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",
                                              use_auth_token=True,
                                              )

article = ("傷")
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(**inputs,
                                  forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],  # 翻訳ターゲットのコード
                                  max_length=30,
                                  )
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
