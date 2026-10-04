"""Frozen C-RADIOv4 feature extraction: one feature vector per 16x16 patch."""
import torch
from einops import rearrange
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor


class FeatureExtractor:
    """
    C-RADIOv4 used as a frozen feature extractor (inference mode, no training).

    The preprocessor only rescales pixels to [0, 1] (do_normalize: false); the per-channel
    normalization is done inside the model (input conditioner), so it is never applied twice.
    """

    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.image_processor = CLIPImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.model.eval().to(device)
        self.patch_size = getattr(self.model, "patch_size", 16)

    def input_size(self, width, height, resolution=None):
        """
        (width, height) the image is resized to before entering the model.

        resolution=None keeps the image size (default, behaviour of v1.3).
        An integer sets the longer side to that value, keeping the aspect ratio.
        In both cases the size is then rounded by the model to a supported resolution
        (a multiple of the patch size).
        """
        if resolution is not None:
            scale = resolution / max(width, height)
            width, height = round(width * scale), round(height * scale)
        height, width = self.model.get_nearest_supported_resolution(height=height, width=width)
        return width, height

    def preprocess(self, pil_image, resolution=None):
        """Resize to a supported resolution and return the model input tensor and its (width, height)."""
        width, height = self.input_size(pil_image.width, pil_image.height, resolution)
        resized = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        inputs = self.image_processor(images=resized, return_tensors="pt", do_resize=False)
        pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
        return pixel_values, (width, height)

    def extract(self, pil_image, resolution=None):
        """
        Run C-RADIOv4 on one RGB image and return one feature vector per patch.

        resolution : None (native size, default) or the longer side in pixels (e.g. 512).

        Returns
        -------
        features : np.ndarray of shape (h_patches, w_patches, D), float32
        resized_size : (width, height) the image was resized to before entering the model
        """
        pixel_values, (width, height) = self.preprocess(pil_image, resolution)
        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype):
            summary, patch_tokens = self.model(pixel_values)  # the global summary vector is not used

        h_patches, w_patches = height // self.patch_size, width // self.patch_size
        features = rearrange(patch_tokens, "1 (h w) d -> h w d", h=h_patches, w=w_patches)
        return features.float().cpu().numpy(), (width, height)
