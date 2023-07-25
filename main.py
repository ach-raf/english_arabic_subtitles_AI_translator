from library import OpusTranslator, MBart50Translator, M2M100Translator


if __name__ == "__main__":
    model_name = "Helsinki-NLP/opus-mt-en-ar"
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    model_name = "facebook/m2m100_418M"
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"

    # translator = MBart50Translator.MBart50Translator(model_name)
    # translator = M2M100Translator.M2M100Translator(model_name)
    translator = OpusTranslator.OpusTranslator(model_name)

    textt = "hello world"

    # translator.translate(textt)
    # Example usage:

    # translator.inspect_model_outputs(textt)
    translation = translator.translate(textt)
    print(f"{translation=}")

    # subtitle_path = "Revolver.2005.Bluray-1080p.srt"
    # subtitle_output_path = f"{subtitle_path[:-4]}_2023_big.srt"
    # translator.new_srt_translation(subtitle_path, subtitle_output_path)
