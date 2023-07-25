from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from library import Translator
import torch


class M2M100Translator(Translator.Translator):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model, self.tokenizer, self.device = self.load_model()

    def get_model_tokenizer_and_device(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = "ar"
        return model, tokenizer, device

    def translate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generated_ids = self.model.generate(
            input_ids, max_length=128, num_beams=5, early_stopping=True
        )
        translation = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return translation

    def subtitles_processing(self, subs):
        return super().subtitles_processing(subs)

    def test_translation(self, text):
        return super().test_translation(text)
