import os
import shutil
import huggingface_hub as hub
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from library import utils


class Translator(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    def download_model(self):
        # Create model folder path
        model_name = self.model_name.split("/")[-1]
        model_path = Path("models") / model_name

        # Download model if folder doesn't exist
        if not os.path.exists(model_path):
            dirname = hub.snapshot_download(self.model_name)
            shutil.copytree(dirname, model_path)
        return model_path

    def load_model(self):
        model_path = self.download_model()
        model, tokenizer, device = self.get_model_tokenizer_and_device(model_path)
        return model, tokenizer, device

    def srt_translation(self, subtitle_path, subtitle_output_path):
        subs = utils.extract_subtitles(subtitle_path)
        translated_subs = self.subtitles_processing_new(subs)
        utils.write_srt_file(translated_subs, subtitle_output_path)

    # Add function to inspect model outputs
    def inspect_model_outputs(self, text):
        input_ids = self.tokenizer(
            text, return_tensors="pt"
        )  # .input_ids.to(self.device)

        # Add dummy decoder input IDs
        decoder_input_ids = torch.tensor(
            [[self.tokenizer.eos_token_id]], device=self.device
        )

        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        logits = outputs.logits

        print(torch.softmax(logits, dim=-1).shape)
        print(torch.topk(logits, k=5, dim=-1))

    @abstractmethod
    def translate(self, text):
        pass

    @abstractmethod
    def test_translation(self, text):
        pass

    @abstractmethod
    def subtitles_processing(self, subs):
        pass

    @abstractmethod
    def get_model_tokenizer_and_device(self, model_path):
        pass


if __name__ == "__main__":
    print("Translator Abstract Class")
