import re

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "ar"


def extract_subtitles(file_path):
    pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n"
    subtitles = []

    with open(file_path, "r") as file:
        srt_content = file.read()

        for match in re.finditer(pattern, srt_content, re.DOTALL):
            subtitle = {
                "index": int(match.group(1)),
                "timestamp": match.group(2),
                "text": match.group(3).strip(),
            }
            subtitles.append(subtitle)

    return subtitles


def subtitles_processing(
    translation_pipeline,
    subs,
    source_language=SOURCE_LANGUAGE,
    target_language=TARGET_LANGUAGE,
):
    for sub in subs:
        print(f'{sub["index"]}/{subs[-1]["index"]}')
        text = sub["text"]
        result_to_ar = translation_pipeline(text, max_length=250)[0]["translation_text"]
        sub["text"] = result_to_ar

        print(f"{sub['text']=}")
    return subs


def write_srt_file(subs, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for sub in subs:
            file.write(str(sub["index"]) + "\n")
            file.write(sub["timestamp"] + "\n")
            file.write(str(sub["text"]) + "\n\n")
