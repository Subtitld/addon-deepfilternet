# DeepFilterNet add-on for Subtitld

Separates the voice of a video from background noise, for
[Subtitld](https://subtitld.org), with
[DeepFilterNet2](https://github.com/Rikorose/DeepFilterNet). Offline, on the
CPU, in about a tenth of the time the video plays, and cleaner than RNNoise.
Subtitld then plays the voice and the background on their own; the cleaned
voice also gives transcription a better start.

It serves the `audio.separate` task, like Subtitld's built-in ffmpeg
separator. DeepFilterNet is a noise suppressor, not a music separator: under
music, part of the music stays with the voice. For a voice over music, use the
HTDemucs add-on.

## What it needs

- **Disk:** about 30 MB. The model, 9 MB, is part of the add-on: nothing
  downloads. While it separates, the add-on keeps the decoded sound in
  Subtitld's temporary folder: about 11 MB a minute of video.
- **Memory:** about 200 MB, whatever the length of the video: the sound goes
  through the model in 30-second pieces.
- **Time:** a minute of video takes about 5 seconds on an Intel Core
  i7-13700T (8 performance cores).

Nothing leaves the machine.

## What it gives back

Two 48 kHz mono FLAC files, as the built-in separator does: the voice (what
DeepFilterNet2 keeps), and the background, which is the rest of the mix (the
video's sound minus the voice).

## Settings

| Setting | Default | |
|---|---|---|
| CPU threads | 0 | the performance cores on a CPU that has efficiency cores too, else one per core |

## How it runs the model

The model is run by [audio.cpp](https://github.com/0xShug0/audio.cpp)
(Apache-2.0), a C++ runtime on ggml, which the add-on calls through its C API
with ctypes. `tools/build_native.py` builds it from source at a pinned commit
with only its built-in audio utilities compiled in, and copies DeepFilterNet2's
weights from the same source tree, checked by sha256.

ffmpeg (Subtitld's own) reads the video's first audio track, at 48 kHz mono.
The model takes it in 30-second pieces, each overlapping the one before by a
second, and the overlaps are crossfaded; a cancel stops it between two
pieces. ffmpeg then writes the two files. The add-on itself needs nothing but
Python's standard library.

## Building

Linux, in the AlmaLinux 8 container, so the archive runs on glibc 2.28 and
newer (needs Docker):

```bash
tools/build_linux.sh                      # audio.cpp, tests, the bundle, the archive
tools/test_archive.sh almalinux:8         # the archive, installed as Subtitld does, on AlmaLinux 8
```

The model's weights are in the archive, so its self-test runs the model on
AlmaLinux 8 too. From source:

```bash
python tools/build_native.py              # Linux: tools/build_native_linux.sh
pip install -e '.[build,dev]'
pytest                                    # the weights come with the build: every test runs
pyinstaller pyinstaller.spec --noconfirm
dist/deepfilternet-addon/deepfilternet-addon --self-test
```

The tests need ffmpeg on `PATH`. The release workflow builds, proves and
publishes every platform on a tag `v<version>` matching `manifest.json`; a
manual run can build only some platforms (`targets`).
`tools/assemble_licences.py` fails the build when a native file in the bundle
has no notice.

## Licence

The add-on is Apache-2.0. The DeepFilterNet2 weights are MIT or Apache-2.0, at
your option; audio.cpp is Apache-2.0, with ggml, cJSON and libyaml (MIT)
compiled in. `THIRD_PARTY_LICENSES/` in each archive lists every file and its
notice.
