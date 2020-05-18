from integration_src.download import download_episodes
from integration_src.parse import parse_episodes
import integration_src.file_manager as fm
from integration_src.embed import embed_transcripts
from integration_src.classify import classify_characters

classify_characters(C=1, max_iter=500)

