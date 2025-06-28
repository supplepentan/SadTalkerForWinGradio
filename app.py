import gradio as gr
import os
import glob
import time
import shutil
import subprocess
from utils.main import SadTalker

current_directory = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_directory, "checkpoints")
config_path = os.path.join(current_directory, "config")
sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

# FFmpegが利用可能かチェック
if not shutil.which("ffmpeg"):
    print(
        "#################################################################################"
    )
    print("WARNING: ffmpeg is not installed or not in the system's PATH.")
    print("The video might not play in the browser when the enhancer is off.")
    print("Please install ffmpeg and add it to your PATH for full functionality.")
    print(
        "#################################################################################"
    )


def sadtalker_test_wrapper(
    source_image,
    driven_audio,
    preprocess_type,
    is_still_mode,
    enhancer,
    batch_size,
    size_of_image,
    pose_style,
):
    # Gradio 4.x以降のアップデートに対応。
    # アップロードされたファイルはオブジェクトとして渡されるため、.name属性からファイルパスを取得します。
    # これにより、古いバージョンでパスが直接渡された場合でも互換性を保ちます。
    if hasattr(source_image, "name"):
        source_image = source_image.name
    if hasattr(driven_audio, "name"):
        driven_audio = driven_audio.name

    # sad_talker.test は内部で結果をファイルに書き出すが、正しいパスを返さない、
    # または返された動画がブラウザで再生できないコーデックである可能性があるため、このラッパーで処理します。
    sad_talker.test(
        source_image,
        driven_audio,
        preprocess_type,
        is_still_mode,
        enhancer,
        batch_size,
        size_of_image,
        pose_style,
    )

    # 結果が保存される 'results' ディレクトリから最新の生成物を探します。
    results_dir = "results"
    # サブディレクトリのリストを取得し、最新のものを探します。
    sub_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    if not sub_dirs:
        print("No result directory found.")
        return None

    latest_dir = max(
        [os.path.join(results_dir, d) for d in sub_dirs], key=os.path.getmtime
    )
    print(f"Latest result directory: {latest_dir}")

    # enhancerが有効な場合、GFPGANによってH.264でエンコードされた `_enhanced.mp4` ファイルを探します。
    if enhancer:
        # Enhancerがオンの場合、最終成果物は `_enhanced.mp4` (これは既にブラウザ互換のはずです)
        video_files = glob.glob(os.path.join(latest_dir, "*_enhanced.mp4"))
        if video_files:
            video_path = video_files[0]
            print(f"Found enhanced video (already compatible): {video_path}")
            return video_path
        else:
            print(
                "Enhancer was on, but no enhanced video was found. Looking for other videos to convert..."
            )

    # Enhancerがオフ、またはenhanced.mp4が見つからない場合、他の動画を探します。
    # 優先順位: _full.mp4 > .mp4 (enhanced.mp4は既にチェック済みなので除外)
    video_files = glob.glob(os.path.join(latest_dir, "*_full.mp4"))
    if not video_files:
        video_files = glob.glob(os.path.join(latest_dir, "*.mp4"))
        # `_enhanced.mp4` は enhancer on の場合の最終成果物なので、ここでは変換対象から除外
        video_files = [f for f in video_files if not f.endswith("_enhanced.mp4")]

    if video_files:
        original_video_path = video_files[0]
        print(f"Found video to process: {original_video_path}")

        # ffmpegが利用可能か再チェック
        if shutil.which("ffmpeg"):
            # ファイル名にタイムスタンプを追加し、ブラウザのキャッシュ問題を確実に回避します
            timestamp = str(int(time.time()))
            reencoded_video_path = os.path.join(
                latest_dir, f"video_for_browser_{timestamp}.mp4"
            )
            print(f"Re-encoding for browser compatibility to: {reencoded_video_path}")

            # Web互換性を最大限に高めるための、シンプルで堅牢なffmpegコマンド
            command = [
                "ffmpeg",
                "-y",
                "-i",
                original_video_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",  # ピクセルフォーマットを互換性の高いものに
                "-c:a",
                "aac",
                "-ar",
                "44100",  # 音声サンプルレートを標準的な44.1kHzに変換
                "-movflags",
                "+faststart",  # Web再生最適化
                reencoded_video_path,
            ]

            try:
                # subprocess.runの出力をキャプチャして詳細なデバッグ情報を得る
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                # ffmpegは進捗や情報をstderrに出力することが多い
                print("--- FFmpeg Log ---")
                print(result.stderr)
                print("------------------")

                # 変換後のファイルが存在し、サイズが0より大きいか確認
                if (
                    os.path.exists(reencoded_video_path)
                    and os.path.getsize(reencoded_video_path) > 0
                ):
                    abs_path = os.path.abspath(reencoded_video_path)
                    print(
                        f"Re-encoding successful. Returning absolute path: {abs_path}"
                    )
                    return abs_path
                else:
                    print(
                        "Re-encoding reported success, but the output file is missing or empty."
                    )
                    print("Returning original video as a fallback.")
                    return original_video_path

            except subprocess.CalledProcessError as e:
                # ffmpegがエラーコードを返した場合
                print("--- FFmpeg Re-encoding FAILED ---")
                print("Stderr:", e.stderr)
                print("---------------------------------")
                print("Returning original video as a fallback.")
                return original_video_path
            except Exception as e:
                print(f"An unexpected error occurred during ffmpeg execution: {e}")
                return original_video_path

        else:
            # ffmpegがない場合は警告を出し、変換前のファイルを返す
            print(
                "ffmpeg not found. Returning original video, which may not play in the browser."
            )
            return original_video_path

    print("No suitable video file found in the result directory.")
    return None  # 動画ファイルが見つからなかった場合


app = gr.Blocks(analytics_enabled=False)
with app:
    with gr.Row():
        with gr.Column(variant="panel"):
            source_image = gr.Image(
                label="Source image",
                type="filepath",
                elem_id="img2img_image",
                width=512,
            )
            driven_audio = gr.Audio(label="Input audio", type="filepath")
        with gr.Column(variant="panel"):
            # gr.TabsがGradioのバージョンと競合してエラーを引き起こしているため、UI構造を単純化します。
            gen_video = gr.Video(
                label="Generated video", format="mp4", elem_id="sadtalker_genearted"
            )

            gr.Markdown("### Settings")
            gr.Markdown(
                "need help? please visit our best practice page for more detials"
            )
            with gr.Column(variant="panel"):
                pose_style = gr.Slider(
                    minimum=0, maximum=46, step=1, label="Pose style", value=0
                )
                size_of_image = gr.Radio(
                    [256, 512],
                    value=256,
                    label="face model resolution",
                    info="use 256/512 model?",
                )
                preprocess_type = gr.Radio(
                    ["crop", "resize", "full", "extcrop", "extfull"],
                    value="crop",
                    label="preprocess",
                    info="How to handle input image?",
                )
                is_still_mode = gr.Checkbox(
                    label="Still Mode (fewer head motion, works with preprocess `full`)"
                )
                batch_size = gr.Slider(
                    label="batch size in generation", step=1, maximum=10, value=2
                )
                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                submit = gr.Button(
                    "Generate", elem_id="sadtalker_generate", variant="primary"
                )

        submit.click(
            fn=sadtalker_test_wrapper,
            inputs=[
                source_image,
                driven_audio,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,
                size_of_image,
                pose_style,
            ],
            outputs=[gen_video],
        )


if __name__ == "__main__":
    app.queue()
    app.launch()
