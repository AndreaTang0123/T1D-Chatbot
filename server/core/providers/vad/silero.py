import time
import os
import numpy as np
import onnxruntime
import opuslib_next
from config.logger import setup_logging
from core.providers.vad.base import VADProviderBase

TAG = __name__
logger = setup_logging()


class VADProvider(VADProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD", config)

        model_path = os.path.join(
            config["model_dir"], "src", "silero_vad", "data", "silero_vad.onnx"
        )
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )

        self.decoder = opuslib_next.Decoder(16000, 1)

        # Handle empty string case
        threshold = config.get("threshold", "0.5")
        threshold_low = config.get("threshold_low", "0.2")
        min_silence_duration_ms = config.get("min_silence_duration_ms", "1000")

        self.vad_threshold = float(threshold) if threshold else 0.5
        self.vad_threshold_low = float(threshold_low) if threshold_low else 0.2

        self.silence_threshold_ms = (
            int(min_silence_duration_ms) if min_silence_duration_ms else 1000
        )

        # Minimum frames to count as voice
        self.frame_window_threshold = 3

    def _init_connection_state(self, conn):
        """为连接初始化独立的 VAD 状态"""
        if not hasattr(conn, "_vad_state"):
            conn._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        if not hasattr(conn, "_vad_context"):
            conn._vad_context = np.zeros((1, 64), dtype=np.float32)

    def is_vad(self, conn, opus_packet):
        # Manual mode: return True directly, no real-time VAD detection, buffer all audio
        if conn.client_listen_mode == "manual":
            return True
            
        try:
            pcm_frame = self.decoder.decode(opus_packet, 960)
            conn.client_audio_buffer.extend(pcm_frame)  # Add new data to buffer

            # Process complete frames in buffer (512 samples each)
            client_have_voice = False
            while len(conn.client_audio_buffer) >= 512 * 2:
                # Extract first 512 samples (1024 bytes)
                chunk = conn.client_audio_buffer[: 512 * 2]
                conn.client_audio_buffer = conn.client_audio_buffer[512 * 2 :]

                # Convert to tensor format required by model
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_input = np.concatenate(
                    [conn._vad_context, audio_float32.reshape(1, -1)], axis=1
                ).astype(np.float32)

                # Detect voice activity
                with torch.no_grad():
                    speech_prob = self.model(audio_tensor, 16000).item()

                # Dual threshold judgment
                if speech_prob >= self.vad_threshold:
                    is_voice = True
                elif speech_prob <= self.vad_threshold_low:
                    is_voice = False
                else:
                    is_voice = conn.last_is_voice

                # If sound not below min threshold, continue previous state, judge as voice present
                conn.last_is_voice = is_voice

                # Update sliding window
                conn.client_voice_window.append(is_voice)
                client_have_voice = (
                    conn.client_voice_window.count(True) >= self.frame_window_threshold
                )

                # If previously voice present, but now silent, and time since last voice exceeds silence threshold, sentence is finished
                if conn.client_have_voice and not client_have_voice:
                    stop_duration = time.time() * 1000 - conn.vad_last_voice_time
                    if stop_duration >= self.silence_threshold_ms:
                        conn.client_voice_stop = True
                if client_have_voice:
                    conn.client_have_voice = True
                    conn.vad_last_voice_time = time.time() * 1000

            return client_have_voice
        except opuslib_next.OpusError as e:
            logger.bind(tag=TAG).info(f"Decoding error: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")
