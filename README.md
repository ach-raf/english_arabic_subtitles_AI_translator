# Neural Machine Translation Model

This repo contains code to work with neural machine translation models from Hugging Face Transformers.

## Models

The following models are supported:

- Helsinki-NLP/opus-mt-en-ar
- Helsinki-NLP/opus-mt-tc-big-en-ar

Models are downloaded automatically on first use.

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Install PyTorch

Install PyTorch with or without CUDA following the instructions here: https://pytorch.org/get-started/locally/

For example:

```bash
# CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU only
pip install torch torchvision torchaudio
```

### Translate

```python
from library import OpusTranslator

model_name = "Helsinki-NLP/opus-mt-en-ar"
translator = OpusTranslator(model_name)

text = "Hello world"
translation = translator.translate(text)
print(translation)
```

### Translate Subtitles

```python
from library import OpusTranslator

model_name = "Helsinki-NLP/opus-mt-en-ar"
translator = OpusTranslator(model_name)

subtitle_path = "example.srt"
output_path = "example_translated.srt"

translator.srt_translation(subtitle_path, output_path)
```

## Inspecting Models

The `inspect_model_outputs` method can be used to inspect the raw model outputs:

```python
translator.inspect_model_outputs(text)
```

This prints the softmax probabilities and top 5 tokens.

## Contributing

Contributions are welcome!

## License

MIT
