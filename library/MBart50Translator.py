import os
import shutil
from pathlib import Path
import torch
import huggingface_hub as hub
from abc import ABC, abstractmethod

from library import Translator
from transformers import (
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
)


class MBart50Translator(Translator.Translator):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model, self.tokenizer, self.device = self.load_model()
        self.batch_size = 16
        self.max_new_tokens = 25
        self.num_beams = 5
        self.early_stopping = True

    def get_model_tokenizer_and_device(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        model  # .to(device)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        tokenizer.src_lang = "en_XX"
        return model, tokenizer, device

    def translate(self, text):
        model_inputs = self.tokenizer(text, return_tensors="pt")
        # Translate from English to Arabic
        generated_tokens = self.model.generate(
            **model_inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id["ar_AR"]
        )
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return translation

    def batch_translate(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_new_tokens=self.max_new_tokens,
        ).to(self.device)
        translated = self.model.generate(
            input_ids=inputs["input_ids"],
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
        )
        translations = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translations

    def test_translation(self, text):
        translation = self.translate(text)
        print(translation)

    def subtitles_processing(self, subs):
        model_inputs = []
        results = []
        for i, sub in enumerate(subs):
            print(f"{i}/{len(subs)}")
            text = sub["text"]

            model_inputs.append(text)

            if len(model_inputs) == self.batch_size or i == len(subs) - 1:
                translations = self.batch_translate(model_inputs)

                for text in model_inputs:
                    results.append(translations.pop(0))

                model_inputs = []

        for i, sub in enumerate(subs):
            sub["text"] = results[i]

        return subs


if __name__ == "__main__":
    print("Mbart50 Translator Class")
