import os
import shutil
from pathlib import Path
import torch
import huggingface_hub as hub
from abc import ABC, abstractmethod

from library import Translator
from transformers import MarianTokenizer, MarianMTModel


class OpusTranslator(Translator.Translator):
    def __init__(self, model_name):
        # Call parent __init__ with all required args
        super().__init__(model_name)
        # self.model_name = model_name
        self.model, self.tokenizer, self.device = self.load_model()
        self.batch_size = 16
        self.max_new_tokens = 25

    def get_model_tokenizer_and_device(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MarianMTModel.from_pretrained(model_path)
        model.to(device)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        return model, tokenizer, device

    def translate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        translated = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            num_beams=5,
            num_return_sequences=5,
            length_penalty=2.0,
        )

        # translated = self.model.generate(**inputs)
        # print(f"{translated=}")

        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)

        return translation

    def batch_translate(self, inputs):
        input_ids = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        inputs = {"input_ids": input_ids}
        translated = self.model.generate(**inputs)
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

            # Append text to batch
            model_inputs.append(text)

            # Translate batch if reached size
            if len(model_inputs) == self.batch_size or i == len(subs) - 1:
                translations = self.batch_translate(model_inputs)

                # Map translations back to original order
                for text in model_inputs:
                    results.append(translations.pop(0))

                model_inputs = []

        # Map results back to subs
        for i, sub in enumerate(subs):
            sub["text"] = results[i]

        return subs


if __name__ == "__main__":
    print("Opus Translator Class")
