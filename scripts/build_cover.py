"""Create the looping README cover from the original icon.mp4 video."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(ffmpeg=None):
    executable = ffmpeg or shutil.which('ffmpeg')
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise SystemExit('Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg.') from error
    source = ROOT / 'assets/brand/icon.mp4'
    output = ROOT / 'assets/brand/moonlit-video-cover.webp'
    subprocess.run([str(executable), '-hide_banner', '-loglevel', 'error', '-y',
                    '-threads', '2', '-i', str(source), '-vf', 'fps=12', '-an',
                    '-c:v', 'libwebp_anim', '-quality', '88',
                    '-compression_level', '5', '-loop', '0', str(output)], check=True)
    print(f'Built {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg')
    args = parser.parse_args()
    build(args.ffmpeg)
