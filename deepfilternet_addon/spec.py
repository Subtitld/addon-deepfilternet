"""What this add-on runs: DeepFilterNet2, through audio.cpp.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Everything that differs between Subtitld's audio.cpp separators is here; the
rest of the package is the same in each of them.
"""

ENV_PREFIX = 'DEEPFILTERNET'         # options arrive as DEEPFILTERNET_<KEY>
MODEL_NAME = 'DeepFilterNet2'

# audio.cpp: the family, the task its session runs, and the load options.
# DeepFilterNet2 is one of audio.cpp's built-in audio utilities.
FAMILY = 'builtin_audio_utils'
CMAKE_MODEL = 'builtin_audio_utils'  # its target in audio.cpp's CMakeLists (AUDIOCPP_MODELS)
TASK = 's2s'
LOAD_OPTIONS = {'utility': 'deepfilternet2'}

# What the model takes and gives back. DeepFilterNet2 cleans speech of noise
# at 48 kHz mono; the add-on takes what it keeps as the voice, and the rest of
# the mix as the background.
RATE = 48000
CHANNELS = 1
OUTPUT = None                        # the named audio output; None for the session's single output

# The mix goes through in pieces of CHUNK_SEC, each overlapping the one before
# by OVERLAP_SEC, crossfaded: progress, a cancel between pieces, and memory
# that stays the same for a two-hour film.
CHUNK_SEC = 30.0
OVERLAP_SEC = 1.0

# How many frames the model's output lags its input; the add-on shifts it back
# so that the mix minus the voice leaves no echo of it in the background.
DELAY_FRAMES = 0

# The shortest piece the model takes; a shorter one (the end of a video, or a
# short clip) gets silence after it, which is cut off again. audio.cpp 0.9.5 runs
# DeepFilterNet2 on input of up to two seconds in one graph, which overflows
# its node budget from just over one second (GGML_ASSERT n_nodes < size);
# longer input it splits into one-second segments. Pieces of 2.1 s or more
# take that path.
MIN_PIECE_SEC = 2.1

# Where the weights come from: in the archive. tools/build_native.py copies
# them from the audio.cpp tree at its pinned commit, checked by sha256.
MODEL_REPO = ''
MODEL_REVISION = ''
MODEL_DIRNAME = ''
MODEL_FILES = ()
MODEL_FILE = ''
BUNDLED_WEIGHTS = ('deepfilternet2', 'deepfilternet2.safetensors')  # under assets/framework/audio_utilities
BUNDLED_SHA256 = '544740171cd81e55dcc7c5d1d889ec79df5808a75eb840a669a681f0d34b0626'

# Memory and speed, for the messages.
RAM_GB = 0.4

# The weights' licence, in THIRD_PARTY_LICENSES/models/ (MIT, or Apache-2.0 beside it).
MODEL_LICENCE = 'LICENSE-MIT.txt'
