import requests
from typing import List, Optional
import integration_src.file_manager as fm


def download_episodes():
    episodes = download_episodes(845, 31373, 31600)
    fm.write_transcripts(episodes)


def download_episodes(forum_id: int, ep_id_min: int, ep_id_max: int):
    episode_content_array = [download_episode(forum_id, cur) for cur in range(ep_id_min, ep_id_max)]

    return episode_content_array


def download_episode(forum_id, ep_id):
    print("Downloading episode " + str(ep_id))
    url = 'https://transcripts.foreverdreaming.org/viewtopic.php?f=' + str(forum_id) + '&t=' + str(
        ep_id) + '&view=print'
    r = requests.get(url)
    if r.status_code == 200:
        return r.content
    else:
        return None
