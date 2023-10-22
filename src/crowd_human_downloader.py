
from arguments import CrowdHumanDownloaderArgs
from services.crowd_human.crowd_human_downloader import CrowdHumanDownloader


if __name__ == '__main__':
    arguments = CrowdHumanDownloaderArgs().parse_args()
    crowd_human_downloader = CrowdHumanDownloader()
    crowd_human_downloader.init(arguments.download, arguments.extract)
