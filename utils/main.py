import torch
import uuid
import os
import shutil
import time
import random
import string
from pydub import AudioSegment
from .utils.preprocess import CropAndExtract
from .test_audio2coeff import Audio2Coeff
from .facerender.animate import AnimateFromCoeff
from .generate_batch import get_data
from .generate_facerender_batch import get_facerender_data
from .utils.init_path import init_path


class SadTalker:
    def __init__(
        self, checkpoint_path="checkpoints", config_path="config", lazy_load=False
    ):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

        os.environ["TORCH_HOME"] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    @staticmethod
    def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
        mp3_file = AudioSegment.from_file(file=mp3_filename)
        mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")

    def test(
        self,
        source_image,
        driven_audio,
        preprocess="crop",
        still_mode=False,
        use_enhancer=False,
        batch_size=1,
        size=256,
        pose_style=0,
        exp_scale=1.0,
        use_ref_video=False,
        ref_video=None,
        ref_info=None,
        use_idle_mode=False,
        length_of_audio=0,
        use_blink=True,
        result_dir="./results/",
    ):
        self.sadtalker_paths = init_path(
            self.checkpoint_path, self.config_path, size, False, preprocess
        )
        print(self.sadtalker_paths)

        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        # time_tag = str(uuid.uuid4())
        # save_dir = os.path.join(result_dir, time_tag)
        # os.makedirs(save_dir, exist_ok=True)
        # 短いランダムなフォルダ名を生成
        time_tag = time.strftime("%Y%m%d%H%M") + "".join(
            random.choices(string.ascii_lowercase + string.digits, k=3)
        )
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        # Gradioが作成した一時ファイルを移動(move)するのではなく、コピー(copy)して使用します。
        # これにより、Gradioのファイル管理との競合を防ぎます。
        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        shutil.copy(source_image, pic_path)

        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))

            #### mp3 to wav
            if ".mp3" in audio_path:
                self.mp3_to_wav(driven_audio, audio_path.replace(".mp3", ".wav"), 16000)
                audio_path = audio_path.replace(".mp3", ".wav")
            else:
                shutil.copy(driven_audio, audio_path)

        elif use_idle_mode:
            audio_path = os.path.join(
                input_dir, "idlemode_" + str(length_of_audio) + ".wav"
            )  ## generate audio from this new audio_path

            one_sec_segment = AudioSegment.silent(
                duration=1000 * length_of_audio
            )  # duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == "all"

        if use_ref_video and ref_info == "all":
            # ビデオ名から拡張子を除去し、オーディオファイル名を作成
            ref_video_videoname = os.path.splitext(os.path.basename(ref_video))[0]
            audio_filename = ref_video_videoname + "_audio.wav"
            audio_path = os.path.join(save_dir, audio_filename)
            print("new audiopath:", audio_path)

            # FFmpeg コマンドを更新（入力と出力が異なることを確認）
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s" % (
                ref_video,
                audio_path,
            )
            os.system(cmd)

        os.makedirs(save_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            pic_path, first_frame_dir, preprocess, True, size
        )

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print("using ref video for genreation")
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_video_coeff_path, _, _ = self.preprocess_model.generate(
                ref_video, ref_video_frame_dir, preprocess, source_image_flag=False
            )
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == "pose":
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == "blink":
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == "pose+blink":
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == "all":
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise ("error in refinfo")
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        # audio2ceoff
        if use_ref_video and ref_info == "all":
            coeff_path = ref_video_coeff_path  # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(
                first_coeff_path,
                audio_path,
                self.device,
                ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
                still=still_mode,
                idlemode=use_idle_mode,
                length_of_audio=length_of_audio,
                use_blink=use_blink,
            )  # longer audio?
            coeff_path = self.audio_to_coeff.generate(
                batch, save_dir, pose_style, ref_pose_coeff_path
            )

        # coeff2video
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            expression_scale=exp_scale,
        )
        return_path = self.animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer="gfpgan" if use_enhancer else None,
            preprocess=preprocess,
            img_size=size,
        )
        video_name = data["video_name"]
        print(f"The generated video is named {video_name} in {save_dir}")

        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc

        gc.collect()

        return return_path
