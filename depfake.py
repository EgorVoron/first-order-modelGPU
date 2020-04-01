from time import time
import os
import warnings
import imageio
import numpy as np
import skvideo.io
from skimage.transform import resize
from moviepy.editor import *
from PIL import Image
from demo import load_checkpoints, make_animation
import argparse

warnings.filterwarnings("ignore")


class Printer:
    def __init__(self, on):
        self.on = on

    def write(self, string):
        if self.on:
            print(string)

    def log(self, string, value):
        if self.on:
            print(f'{string}: {value}')


parser = argparse.ArgumentParser()
parser.add_argument('video_path')
parser.add_argument('photo_path')
parser.add_argument('checkpoint_path')
parser.add_argument('output_path')
parser.add_argument('max_duration')
parser.add_argument('watermark')
parser.add_argument('debug')
args = parser.parse_args()

video_path = args.video_path
photo_path = args.photo_path
checkpoint_path = args.checkpoint_path
output_path = args.output_path
max_duration = int(args.max_duration)
watermark = bool(int(args.watermark))
debug = bool(int(args.debug))

printer = Printer(debug)


def prepare_input(photo_path, video_path, fps, max_duration):
    t = time()
    source_image = imageio.imread(photo_path)
    driving_video = skvideo.io.vread(video_path)

    source_image = resize(source_image, (256, 256))[..., :3]

    short_driving_video = []
    duration = min(int(len(driving_video) / int(fps)), max_duration)
    for n in range(int(fps) * duration):
        short_driving_video.append(resize(driving_video[n], (256, 256))[..., :3])

    printer.log('SHORTENED VIDEO DURATION', len(short_driving_video) / fps)
    printer.log('PREPROCESSING TIME', time() - t)
    return source_image, short_driving_video


def get_predictions(photo, video):
    t = time()
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                              checkpoint_path=checkpoint_path)
    # '/content/gdrive/My Drive/DepFak/vox-adv-cpk.pth.tar'
    predictions = make_animation(photo, video, generator, kp_detector, relative=True)
    printer.log('PREDICTION TIME', time() - t)
    return predictions


def add_watermark(img):
    return img


def run_core(video_path, photo_path, frames_dir, output_path, max_duration):
    original_video = VideoFileClip(video_path)

    fps = original_video.fps

    printer.write('Preparing input...')
    source_image, driving_video = prepare_input(photo_path, video_path, fps, max_duration)
    printer.write('Input is prepared!')

    printer.write('Making predictions...')
    predictions = get_predictions(video=driving_video, photo=source_image)
    printer.write('Predictions are done!')

    printer.write('Saving frames...')
    for n, i in enumerate(predictions):
        input_size = 256
        img = Image.fromarray((i * 255).astype(np.uint8)).resize((input_size, input_size)).convert('RGB')
        img.save(os.path.join(frames_dir, f'{n}.png'))
    printer.write('Frames are saved')

    printer.write('Saving video...')

    clips = []
    for i in range(len(os.listdir(frames_dir))):
        filename = str(i) + '.png'
        img = np.array(Image.open(os.path.join(frames_dir, filename)))
        img = add_watermark(img)
        clips.append(ImageClip(img).set_duration(1 / fps))

    duration = min(int(len(driving_video) / int(fps)), max_duration)
    audio_clip = original_video.subclip(0, duration).audio
    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio_clip)
    final_video.write_videofile(output_path, fps=fps, audio=True)
    printer.write('Video is saved!')


if not os.path.isdir('frames'):
    os.mkdir('frames')

t = time()
run_core(video_path=video_path,
         photo_path=photo_path,
         frames_dir='frames',
         output_path=output_path,
         max_duration=max_duration)
printer.log('FULL TIME', time() - t)
printer.write('FINISHED')
