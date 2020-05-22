import requests
import src.file_manager as fm

__all__ = ["download_episodes"]

def download_episodes(forum_id =845, ep_id_min=31373, ep_id_max=31600):
    episode_content_array = [download_episode(forum_id, cur) for cur in range(ep_id_min, ep_id_max)]
    fm.write_transcripts(episode_content_array)
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
