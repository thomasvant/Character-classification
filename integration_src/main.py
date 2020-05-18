from integration_src.download import download_episodes
from integration_src.parse import parse_episodes
import integration_src.file_manager as fm
from integration_src.embed import embed_transcripts

# Download
# fm.write_transcripts(download_episodes(845, 31373, 31600))

# Parse
# parsed = parse_episodes(fm.get_transcripts())
# print(parsed["parsed"]["character"])
# fm.write_df(parse_episodes(fm.get_transcripts()), "0_parsed")
# parsed = fm.get_df("0_parsed")
# print(parsed["parsed"]["line"])
# Embed
# parsed = fm.get_df("0_parsed")
# embedded = embed_transcripts(parsed, "fasttext")
# fm.write_df(embedded, "1_embedded")
embedded = fm.get_df("1_embedded")
    # Classify
    # Benchmark
#
#
# if __name__ == "__main__":
#     main()
