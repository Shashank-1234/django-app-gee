import os
import numpy as np
import tensorflow as tf
from django.conf import settings

class LandCoverModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Update the path to your SavedModel directory.
        # (Make sure this model was saved using tf.saved_model.save if you intend to use its signatures.)
        model_dir = os.path.join(settings.BASE_DIR, '/Users/shashankdutt/Downloads/saved_model/my_model')
        try:
            # Load the SavedModel; returns a trackable object
            loaded = tf.saved_model.load(model_dir)
            # Grab the default serving signature
            self.infer = loaded.signatures["serving_default"]

            # —–––– Debug: list out the keys so you see what your model expects/returns
            print(">>> SavedModel INPUT keys:", list(self.infer.structured_input_signature[1].keys()))
            print(">>> SavedModel OUTPUT keys:", list(self.infer.structured_outputs.keys()))

            # Automatically pick the sole output key (or change index if you have multiple)
            out_keys = list(self.infer.structured_outputs.keys())
            if not out_keys:
                raise RuntimeError("No outputs found in the SavedModel signature!")
            self._output_key = out_keys[0]

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_dir}: {e}")

        # your color map
        self.classifications = {
            0: {"name": "No data",                  "color": "000000"},
            1: {"name": "Water",                    "color": "419BDF"},
            2: {"name": "Crops & Trees",            "color": "397D49"},
            3: {"name": "Grass & Scrub",            "color": "88B053"},
            4: {"name": "Flooded vegetation",       "color": "7A87C6"},
            5: {"name": "Built Area & Bare Ground", "color": "C4281B"},
            6: {"name": "Snow/Ice",                 "color": "B39FE1"},
            7: {"name": "Cloud",                    "color": "FFFFFF"},
        }

    def predict(self, model_input: np.ndarray):
        """
        model_input: shape (1, H, W, C)
        Splits channels apart to match signature inputs, runs infer, then
        returns (class_map, prob_map).
        """
        # Build feed dict by splitting on channel axis
        input_keys = list(self.infer.structured_input_signature[1].keys())
        if model_input.shape[-1] != len(input_keys):
            raise ValueError(f"Model expects {len(input_keys)} channels, got {model_input.shape[-1]}")

        # tf.split returns list of (1,H,W,1) tensors
        split_tensors = tf.split(model_input, num_or_size_splits=len(input_keys), axis=-1)
        feed = dict(zip(input_keys, split_tensors))

        # Run inference
        outputs = self.infer(**feed)

        # Grab the logits/probs tensor
        tensor = outputs[self._output_key]
        # If it's logits, apply softmax; if it's already softmaxed, this is a no-op
        probs = tf.nn.softmax(tensor, axis=-1)[0].numpy()
        classes = np.argmax(probs, axis=-1)

        return classes, probs














''' # First Basic stuff - WORKS 


from django.db import models

class SatelliteImage(models.Model):
    image_id = models.CharField(max_length=100, unique=True)
    date = models.DateTimeField()
    metadata = models.JSONField()

    def __str__(self):
        return self.image_id
'''