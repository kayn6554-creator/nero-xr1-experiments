# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import copy
import glob
import json
import os
import random
import re
from functools import lru_cache

import numpy as np
import torch
try:
    from decord import VideoReader as _DecordVideoReader
except ImportError:
    _DecordVideoReader = None


class _NumpyBatch:
    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


class VideoReader:
    """Decord-compatible reader with a runtime FFmpeg/OpenCV fallback.

    Decord can import successfully while its bundled FFmpeg still lacks the
    codec used by a particular file (the real Piper recordings use AV1). In
    that case construction or get_batch fails, so an ImportError-only fallback
    is insufficient.
    """

    def __init__(self, path, num_threads=2):
        self.path = str(path)
        self._reader = None
        self._decord_error = None
        if _DecordVideoReader is not None:
            try:
                self._reader = _DecordVideoReader(self.path, num_threads=num_threads)
            except Exception as error:
                self._decord_error = error

    def seek(self, position):
        if self._reader is not None:
            try:
                self._reader.seek(position)
            except Exception as error:
                self._decord_error = error
                self._reader = None

    def get_batch(self, indices):
        indices = [int(index) for index in indices]
        if self._reader is not None:
            try:
                return self._reader.get_batch(indices)
            except Exception as error:
                self._decord_error = error
                self._reader = None

        imageio_error = None
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(self.path, "ffmpeg")
            try:
                return _NumpyBatch(np.stack([reader.get_data(index) for index in indices]))
            finally:
                reader.close()
        except Exception as error:
            imageio_error = error

        try:
            import cv2

            cap = cv2.VideoCapture(self.path)
            frames = []
            try:
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(f"failed to decode frame {index}")
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            return _NumpyBatch(np.stack(frames))
        except Exception as opencv_error:
            raise RuntimeError(
                f"no video backend could decode {self.path}; "
                f"decord={self._decord_error!r}, imageio={imageio_error!r}, "
                f"opencv={opencv_error!r}"
            ) from opencv_error

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
from transformers.utils import logging
from tqdm import tqdm

from mibot.utils.io import (
    build_action_mask,
    compose_action,
    compose_piper_left_state,
    compose_piper_joint_action,
    compose_state,
    get_action_parts,
    get_value,
    normalize_action_parts,
    normalize_quantile,
    resize_image,
    rotm2aa_batch,
    validate_quantiles,
    validate_stats,
)

from mibot.utils.nero_steps import step_text_for_frame
from mibot.utils.nero_kin import fk as nero_fk, rot_log_batch as nero_rot_log_batch
from mibot.utils.ef_align import (
    PIPER_GRIPPER_TO_NERO_M,
    compose_ee_state,
    ee_relative_action,
    is_piper_traj,
    nero_flange_pose,
    nero_flange_poses,
    piper_flange_pose,
    piper_flange_poses,
)

logger = logging.get_logger(__name__)
PROMPT_RE = re.compile(r"(<image>|<video>)")


class JsonDataset(Dataset):
    def __init__(self, params):
        data = params["train_datasets"]
        self.action_length = int(data.get("action_length", params.get("action_length", 30)))
        self.action_layout = str(data.get("action_layout", params.get("action_layout", "ee")))
        self.batch_size = int(data.get("batch_size", 16))
        self.max_samples = int(params.get("max_steps", 1000)) * self.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        self.mean, self.std = validate_stats(data["mean"], data["std"], self.action_length)
        self.q01, self.q99 = validate_quantiles(data["q01"], data["q99"])
        self.files = self._json_files(data["paths"])
        self.finite_epoch = bool(params.get("finite_epoch", False))
        # M1: append "Step k/4: ..." for the current frame (off = original behaviour).
        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))
        self.step_merge23 = bool(data.get("step_merge23", False))  # M4: steps 2+3 share one text
        # M2: gripper target = recorded command (absolute), not command - current opening.
        self.gripper_absolute = bool(data.get("gripper_absolute", False))
        # M2: thin long static runs as training anchors (None = off).
        dwell = data.get("dwell_keep")
        self.dwell_keep = {str(k): int(v) for k, v in dict(dwell).items()} if dwell else None
        self.dwell_tail = int(data.get("dwell_tail", 10))
        self.dwell_speed_deg = float(data.get("dwell_speed_deg", 2.0))
        self.dwell_gripper_m = float(data.get("dwell_gripper_m", 0.001))
        self.dwell_removed = {}
        self.samples = self._samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        traj = self._read_json(sample["file"])
        frame = sample["frame_index"]
        steps = min(self.action_length, int(traj["num_frames"]) - frame)
        if steps <= 0:
            raise IndexError(f"invalid sample at index={index}: frame_index={frame}")

        prompt = self._prompt(traj, frame)
        if self.action_layout == "ee":
            left = self._arm_action(traj, "left", frame, steps)
            right = self._arm_action(traj, "right", frame, steps)
            waist = self._delta(traj, "proprios.waist_pos", "actions.waist_pos", frame, steps)
            base = self._pad(self._future(traj, "actions.base_vel", frame, steps), steps)
            action_parts = normalize_action_parts(
                (*left, *right, waist, base), self.mean, self.std, self.action_layout
            )
            action = compose_action(*action_parts, action_length=self.action_length)
        elif self.action_layout == "piper_joint":
            left = self._joint_action(traj, "left", frame, steps)
            right = self._joint_action(traj, "right", frame, steps)
            action_parts = normalize_action_parts(
                (*left, *right), self.mean, self.std, self.action_layout
            )
            action = compose_piper_joint_action(
                *action_parts, action_length=self.action_length
            )
        elif self.action_layout == "piper_left_joint":
            joint, gripper = normalize_action_parts(
                self._joint_action(traj, "left", frame, steps), self.mean, self.std, self.action_layout
            )
            action = np.zeros_like(self.mean, dtype=np.float32)
            action[:, :6], action[:, 6:7] = joint, gripper
        elif self.action_layout in ("nero_joint", "nero_joint_sb"):
            # Single Nero: seven joints + gripper, with no second arm or view.
            values = (
                self._delta(traj, "proprios.arm_joint", "actions.arm_joint", frame, steps),
                self._pad(self._future(traj, "actions.gripper_pos", frame, steps), steps)
                if self.gripper_absolute
                else self._delta(traj, "proprios.gripper_pos", "actions.gripper_pos", frame, steps),
            )
            joint, gripper = normalize_action_parts(values, self.mean, self.std, self.action_layout)
            if joint.shape != (self.action_length, 7) or gripper.shape != (self.action_length, 1):
                raise ValueError("Nero requires seven joints and one gripper")
            action = np.zeros((self.action_length, 60), dtype=np.float32)
            action[:, :7], action[:, 7:8] = joint, gripper
            if self.action_layout == "nero_joint_sb":
                # M5: slot 8 = 1 where a future frame already belongs to the next step.
                idx = (traj.get("source") or {}).get("step_index")
                flag = np.zeros((steps, 1), dtype=np.float32)
                if idx is not None:
                    flag[:, 0] = [float(idx[frame + k] != idx[frame]) for k in range(steps)]
                flag = self._pad(flag, steps)
                action[:, 8:9] = (flag - self.mean[:, 8:9]) / (self.std[:, 8:9] + 1e-6)
        elif self.action_layout == "nero_ee":
            # EF: flange pose of the commanded joints relative to the follower's
            # flange at the chunk start, + gripper, + j3 delta (elbow).
            q0 = self._frame(traj, "proprios.arm_joint", frame).astype(np.float64)
            qa = self._future(traj, "actions.arm_joint", frame, steps).astype(np.float64)
            T0 = nero_fk(q0)
            Ts = np.stack([nero_fk(q) for q in qa])
            R0t = T0[:3, :3].T
            pos = (R0t @ (Ts[:, :3, 3] - T0[:3, 3]).T).T
            aa = nero_rot_log_batch(np.einsum("ij,njk->nik", R0t, Ts[:, :3, :3]))
            elbow = qa[:, 2:3] - q0[2]
            grip = self._future(traj, "actions.gripper_pos", frame, steps)
            if not self.gripper_absolute:
                grip = grip - self._frame(traj, "proprios.gripper_pos", frame)
            values = tuple(self._pad(v, steps) for v in (pos, aa, grip, elbow))
            parts = normalize_action_parts(values, self.mean, self.std, self.action_layout)
            action = np.zeros((self.action_length, 60), dtype=np.float32)
            for (_, slc), value in zip(get_action_parts(self.action_layout), parts):
                if value.shape != (self.action_length, slc.stop - slc.start):
                    raise ValueError(f"nero_ee part {slc} has shape {value.shape}")
                action[:, slc] = value
        elif self.action_layout == "nero_ee_sb":
            if is_piper_traj(traj):
                q0 = self._frame(traj, "proprios.left_arm_joint", frame).astype(np.float64)
                qa = self._future(traj, "actions.left_arm_joint", frame, steps).astype(np.float64)
                T0 = piper_flange_pose(q0)
                Ts = piper_flange_poses(qa)
                pos, aa = ee_relative_action(T0, Ts)
                grip = self._future(traj, "actions.left_gripper_pos", frame, steps).astype(np.float64).reshape(steps, 1) * PIPER_GRIPPER_TO_NERO_M
                if not self.gripper_absolute:
                    g0 = float(self._frame(traj, "proprios.left_gripper_pos", frame).reshape(-1)[0]) * PIPER_GRIPPER_TO_NERO_M
                    grip = grip - g0
                elbow = np.zeros((steps, 1), dtype=np.float32)
                has_elbow = False
            else:
                q0 = self._frame(traj, "proprios.arm_joint", frame).astype(np.float64)
                qa = self._future(traj, "actions.arm_joint", frame, steps).astype(np.float64)
                T0 = nero_flange_pose(q0)
                Ts = nero_flange_poses(qa)
                pos, aa = ee_relative_action(T0, Ts)
                grip = self._future(traj, "actions.gripper_pos", frame, steps).astype(np.float64).reshape(steps, 1)
                if not self.gripper_absolute:
                    g0 = float(self._frame(traj, "proprios.gripper_pos", frame).reshape(-1)[0])
                    grip = grip - g0
                elbow = qa[:, 2:3] - q0[2]
                has_elbow = True
            idx = (traj.get("source") or {}).get("step_index")
            has_step_flag = idx is not None
            flag = np.zeros((steps, 1), dtype=np.float32)
            if has_step_flag:
                flag[:, 0] = [float(idx[frame + k] != idx[frame]) for k in range(steps)]
            values = tuple(self._pad(v, steps) for v in (pos, aa, grip, elbow, flag))
            parts = normalize_action_parts(values, self.mean, self.std, self.action_layout)
            action = np.zeros((self.action_length, 60), dtype=np.float32)
            for (_, slc), value in zip(get_action_parts(self.action_layout), parts):
                if value.shape != (self.action_length, slc.stop - slc.start):
                    raise ValueError(f"nero_ee_sb part {slc} has shape {value.shape}")
                action[:, slc] = value
        else:
            raise ValueError(f"unsupported action_layout={self.action_layout!r}")
        state = normalize_quantile(self._state(traj, frame), self.q01, self.q99)
        action_mask = build_action_mask(
            self.action_length, self._mask(traj, steps), self.action_layout
        )
        if self.action_layout == "nero_ee_sb":
            if not has_elbow:
                action_mask[:, 7:8] = 0
            if not has_step_flag:
                action_mask[:, 8:9] = 0
        prompt["conversations"] += [
            {"from": "human", "value": "Robot state: <state>"},
            {"from": "gpt", "value": "".join(f"<a_{index}>" for index in range(steps)) + "<score>"},
        ]

        return {
            "_diagnostic_sample": {
                "dataset_index": int(index),
                "file": sample["file"],
                "frame_index": int(frame),
                "num_frames": int(traj["num_frames"]),
                "steps": int(steps),
            },
            "messages": self._messages(
                prompt["conversations"],
                self._augment(self._images(traj, prompt["images"], frame)),
            ),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "state": torch.from_numpy(state),
            "vlm_action_target": torch.from_numpy(action[:steps]),
            "vlm_action_mask": torch.from_numpy(action_mask[:steps]),
            "vlm_action_actual_length": steps,
        }

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_json(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def _json_files(paths):
        files = set()
        for path in paths:
            if os.path.isfile(path) and path.endswith(".json"):
                files.add(path)
            elif os.path.isdir(path):
                files.update(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
            else:
                raise ValueError(f"training path is not a JSON file or directory: {path}")
        if not files:
            raise ValueError("training paths contain no JSON files")
        return sorted(files)

    def _samples(self):
        source_samples = []
        logger.info(f"train set has {len(self.files)} files.")
        for path in tqdm(self.files):
            count = int(self._read_json(path)["num_frames"])
            if count <= 0:
                logger.warning(f"File {path} has no frames, skipping.")
                continue
            keep = self._dwell_keep_mask(path, count) if self.dwell_keep else None
            source_samples.extend(
                {"file": path, "frame_index": i} for i in range(count) if keep is None or keep[i]
            )

        if not source_samples:
            raise ValueError("training data has no frames")
        # Choose from the complete corpus before applying the step-bounded
        # budget. A sorted prefix silently starves later roots in mixed-data
        # runs that are shorter than one full pass over the data.
        random.Random(42).shuffle(source_samples)
        if self.finite_epoch:
            logger.info(f"Finite epoch train samples: {len(source_samples)}")
            return source_samples
        q, r = divmod(self.max_samples, len(source_samples))
        samples = source_samples * q + source_samples[:r]
        logger.info(f"Raw train samples: {len(source_samples)}")
        logger.info(f"Total train samples: {len(samples)}")
        return samples

    def _dwell_keep_mask(self, path, count):
        """M2: True = use this frame as a training anchor.

        A frame is static when the fastest joint moves slower than
        dwell_speed_deg (deg/s) and the commanded gripper changes less than
        dwell_gripper_m since the previous frame.  In each static run inside
        one step keep the first dwell_keep[step] (or dwell_keep["default"])
        frames and the last dwell_tail frames; drop the middle.
        """
        traj = self._read_json(path)
        if is_piper_traj(traj):
            q = np.asarray(get_value(traj, "proprios.left_arm_joint"), dtype=np.float64)
            g = np.asarray(get_value(traj, "actions.left_gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0] * PIPER_GRIPPER_TO_NERO_M
        else:
            q = np.asarray(get_value(traj, "proprios.arm_joint"), dtype=np.float64)
            g = np.asarray(get_value(traj, "actions.gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0]
        source = traj.get("source") or {}
        steps = list(source.get("step_index") or [0] * count)
        fps = float(traj.get("fps", 10))
        speed = np.r_[np.inf, np.abs(np.diff(q, axis=0)).max(1) * (180.0 / np.pi) * fps]
        grip = np.r_[np.inf, np.abs(np.diff(g))]
        static = (speed < self.dwell_speed_deg) & (grip < self.dwell_gripper_m)
        keep = np.ones(count, dtype=bool)
        i = 0
        while i < count:
            if not static[i]:
                i += 1
                continue
            j = i
            while j < count and static[j] and steps[j] == steps[i]:
                j += 1
            head = int(self.dwell_keep.get(str(steps[i]), self.dwell_keep.get("default", 10**9)))
            if j - i > head + self.dwell_tail:
                keep[i + head:j - self.dwell_tail] = False
                key = f"{source.get('dataset', '?')}/step{steps[i]}"
                self.dwell_removed[key] = self.dwell_removed.get(key, 0) + (j - i - head - self.dwell_tail)
            i = j
        return keep

    def _prompt(self, traj, frame=None):
        prompts = traj.get("instruction", {}).get("general") or []
        if not prompts:
            raise ValueError(f"trajectory {traj.get('time', '<unknown>')} has no instruction.general")
        prompt = copy.deepcopy(random.choice(prompts))
        conversations = prompt.get("conversations", [])
        if self.action_layout in ("piper_left_joint", "nero_ee_sb") and "observations.wrist_right" in prompt["images"]:
            if prompt["images"] != ["observations.ego", "observations.wrist_left", "observations.wrist_right"]:
                raise ValueError("unexpected Piper camera order")
            marker = "\n# Right-Wrist View\n<image>"
            if conversations[0]["value"].count(marker) != 1:
                raise ValueError("missing right-wrist prompt marker")
            prompt["images"].pop()
            conversations[0]["value"] = conversations[0]["value"].replace(marker, "")
        if len(conversations) >= 2:
            if getattr(self, "step_prompt", False):
                src = traj.get("source") or {}
                if src.get("step_index") is not None and str(src.get("dataset", "")).startswith("nero_"):
                    conversations[0]["value"] += " " + step_text_for_frame(
                        traj, frame, getattr(self, "step_merge23", False))
            conversations[0]["value"] += " /no_cot"
            conversations[1]["value"] = "<cot></cot>"
        return prompt

    @staticmethod
    def _images(traj, keys, frame):
        images = []
        frame = [max(frame, 0)]
        for key in keys:
            infos = get_value(traj, key)
            if infos is None:
                raise ValueError(f"{key} is missing in trajectory {traj.get('time', '<unknown>')}")
            for info in infos:
                video = VideoReader(info["path"], num_threads=2)
                video.seek(0)
                images.extend(
                    Image.fromarray(image)
                    for image in JsonDataset._crop_bbox(
                        video.get_batch([frame[0] + int(info.get("start", 0))]).asnumpy(),
                        info.get("crop_bbox"),
                    )
                )
        return images

    @staticmethod
    def _crop_bbox(frames, bbox):
        if bbox is None:
            return frames
        height, width = frames.shape[1:3]
        top, left, crop_height, crop_width = (int(value) for value in bbox)
        output = np.zeros((frames.shape[0], crop_height, crop_width, frames.shape[3]), dtype=frames.dtype)
        source_top, source_left = max(0, top), max(0, left)
        source_bottom, source_right = min(height, top + crop_height), min(width, left + crop_width)
        if source_bottom > source_top and source_right > source_left:
            target_top, target_left = source_top - top, source_left - left
            output[
                :, target_top : target_top + source_bottom - source_top,
                target_left : target_left + source_right - source_left,
            ] = frames[:, source_top:source_bottom, source_left:source_right]
        return output

    def _augment(self, images):
        ops = (
            (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (adjust_contrast, random.uniform(0.5, 1.5)),
            (adjust_saturation, random.uniform(0.5, 1.5)),
            (adjust_hue, random.uniform(0.0, 0.0)),
        )
        flags = [random.randint(0, 1) == 0 for _ in ops]

        out = []
        for image in images:
            image = resize_image(image, factor=32, max_pixels=160000)
            for use, (op, value) in zip(flags, ops):
                if use:
                    image = op(image, value)
            out.append(image)
        return out

    @staticmethod
    def _messages(conversations, images):
        pool = [{"type": "image", "image": image} for image in images]
        messages = []

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                messages.append({"role": role, "content": [{"type": "text", "text": turn["value"]}]})
                continue

            content = []
            for part in PROMPT_RE.split(turn["value"]):
                if part == "<image>":
                    if not pool:
                        raise ValueError("number of <image> placeholders exceeds the number of provided images")
                    content.append(pool.pop(0))
                elif part == "<video>":
                    raise ValueError("video placeholders are not supported in the JSON dataset")
                elif part:
                    content.append({"type": "text", "text": part})
            messages.append({"role": role, "content": content})

        if pool:
            raise ValueError(f"{len(pool)} image(s) remain unused")
        return messages

    def _mask(self, traj, steps):
        kind = traj.get("trajectory_type", "success")
        if kind not in {"success", "ongoing", "invalid"}:
            raise ValueError(f"Unsupported trajectory_type: {kind}")
        mask = np.zeros(self.action_length, dtype=np.int32)
        mask[:steps] = 1
        return mask

    def _arm_action(self, traj, arm, frame, steps):
        rotm = self._frame(traj, f"proprios.{arm}_ee_rotm", frame).reshape(3, 3)
        pos = self._frame(traj, f"proprios.{arm}_ee_pos", frame)
        target_pos = self._future(traj, f"actions.{arm}_ee_pos", frame, steps)
        target_rotm = self._future(traj, f"actions.{arm}_ee_rotm", frame, steps).reshape(-1, 3, 3)
        return (
            self._pad((rotm.T @ (target_pos - pos).T).T, steps),
            self._pad(rotm2aa_batch(rotm.T @ target_rotm), steps),
            self._delta(traj, f"proprios.{arm}_gripper_pos", f"actions.{arm}_gripper_pos", frame, steps),
        )

    def _joint_action(self, traj, arm, frame, steps):
        return (
            self._delta(
                traj,
                f"proprios.{arm}_arm_joint",
                f"actions.{arm}_arm_joint",
                frame,
                steps,
            ),
            self._delta(
                traj,
                f"proprios.{arm}_gripper_pos",
                f"actions.{arm}_gripper_pos",
                frame,
                steps,
            ),
        )

    def _delta(self, traj, current_key, target_key, frame, steps):
        return self._pad(self._future(traj, target_key, frame, steps) - self._frame(traj, current_key, frame), steps)

    @staticmethod
    def _frame(traj, key, frame):
        return np.asarray(get_value(traj, key)[frame], dtype=np.float32)

    @staticmethod
    def _future(traj, key, frame, steps):
        return np.asarray(get_value(traj, key)[frame : frame + steps], dtype=np.float32)

    def _pad(self, value, steps):
        value = np.asarray(value)
        if steps == self.action_length:
            return value
        return np.concatenate([value, np.repeat(value[-1:], self.action_length - steps, axis=0)], axis=0)

    def _state(self, traj, frame):
        if self.action_layout == "piper_left_joint":
            return compose_piper_left_state(self._frame(traj, "proprios.left_arm_joint", frame),
                                           self._frame(traj, "proprios.left_gripper_pos", frame))
        if self.action_layout == "nero_ee_sb":
            if is_piper_traj(traj):
                T0 = piper_flange_pose(self._frame(traj, "proprios.left_arm_joint", frame))
                g0 = float(self._frame(traj, "proprios.left_gripper_pos", frame).reshape(-1)[0]) * PIPER_GRIPPER_TO_NERO_M
                return compose_ee_state(T0, g0, is_piper=True)
            T0 = nero_flange_pose(self._frame(traj, "proprios.arm_joint", frame))
            g0 = float(self._frame(traj, "proprios.gripper_pos", frame).reshape(-1)[0])
            return compose_ee_state(T0, g0, is_piper=False)
        if self.action_layout in ("nero_joint", "nero_ee", "nero_joint_sb"):
            joint = self._frame(traj, "proprios.arm_joint", frame)
            gripper = self._frame(traj, "proprios.gripper_pos", frame)
            if joint.shape != (7,) or gripper.shape != (1,):
                raise ValueError("Nero requires seven joints and one gripper")
            state = np.zeros((1, 60), dtype=np.float32)
            state[:, :7], state[:, 7:8] = joint, gripper
            return state
        return compose_state(
            left_gripper=np.asarray(get_value(traj, "proprios.left_gripper_pos")[frame], dtype=np.float32),
            left_joint=np.asarray(get_value(traj, "proprios.left_arm_joint")[frame], dtype=np.float32),
            right_gripper=np.asarray(get_value(traj, "proprios.right_gripper_pos")[frame], dtype=np.float32),
            right_joint=np.asarray(get_value(traj, "proprios.right_arm_joint")[frame], dtype=np.float32),
        )
