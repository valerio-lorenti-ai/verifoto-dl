"""
Advanced augmentations specifiche per AI-generated image detection.
Focus su artefatti realistici che il modello deve imparare a gestire.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from typing import Tuple


class RandomJPEGCompression:
    """
    Ricompressione JPEG a qualità variabile.
    Simula compressione WhatsApp, social media, etc.
    """
    def __init__(self, quality_min=50, quality_max=95, p=0.6):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        q = random.randint(self.quality_min, self.quality_max)
        arr = np.array(img.convert("RGB"))
        bgr = arr[:, :, ::-1]

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        ok, enc = cv2.imencode(".jpg", bgr, encode_param)
        if not ok:
            return img

        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        rgb = dec[:, :, ::-1]
        return Image.fromarray(rgb)


class RandomResizeDownUp:
    """
    Resize down poi up con interpolazione.
    Simula screenshot, resize, re-upload.
    Introduce blur e artefatti di interpolazione.
    """
    def __init__(self, scale_min=0.5, scale_max=0.9, p=0.4):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.scale_min, self.scale_max)
        
        # Resize down
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_small = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Resize up (back to original size)
        img_up = img_small.resize((w, h), Image.BILINEAR)
        
        return img_up


class RandomSharpening:
    """
    Sharpening casuale.
    Simula post-processing, filtri Instagram, etc.
    """
    def __init__(self, factor_min=1.0, factor_max=2.0, p=0.3):
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        factor = random.uniform(self.factor_min, self.factor_max)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)


class RandomGaussianNoise:
    """
    Rumore gaussiano.
    Simula sensor noise, low-light, compressione aggressiva.
    """
    def __init__(self, sigma_min=0.0, sigma_max=0.02, p=0.35):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        
        arr = np.asarray(img).astype(np.float32) / 255.0
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomScreenshotArtifacts:
    """
    Simula artefatti da screenshot:
    - Crop con margini
    - Bordi neri/bianchi
    - Resize non uniforme
    - UI-like margins
    
    IMPORTANTE: Non altera il contenuto semantico, solo aggiunge margini/crop.
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        w, h = img.size
        
        # Tipo di artefatto
        artifact_type = random.choice(['border', 'crop', 'ui_margin'])
        
        if artifact_type == 'border':
            # Aggiungi bordo nero/bianco
            border_size = random.randint(2, 10)
            border_color = random.choice([(0, 0, 0), (255, 255, 255)])
            
            new_img = Image.new('RGB', (w + 2*border_size, h + 2*border_size), border_color)
            new_img.paste(img, (border_size, border_size))
            
            # Resize back to original size
            return new_img.resize((w, h), Image.BILINEAR)
        
        elif artifact_type == 'crop':
            # Crop casuale poi resize
            crop_pct = random.uniform(0.85, 0.95)
            new_w = int(w * crop_pct)
            new_h = int(h * crop_pct)
            
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            
            cropped = img.crop((left, top, left + new_w, top + new_h))
            return cropped.resize((w, h), Image.BILINEAR)
        
        else:  # ui_margin
            # Simula UI margin (es: status bar, navigation bar)
            margin_size = random.randint(5, 20)
            margin_color = random.choice([(0, 0, 0), (255, 255, 255), (240, 240, 240)])
            
            # Margin top o bottom
            if random.random() < 0.5:
                # Top margin
                new_img = Image.new('RGB', (w, h), margin_color)
                new_img.paste(img.resize((w, h - margin_size), Image.BILINEAR), (0, margin_size))
            else:
                # Bottom margin
                new_img = Image.new('RGB', (w, h), margin_color)
                new_img.paste(img.resize((w, h - margin_size), Image.BILINEAR), (0, 0))
            
            return new_img


class RandomBlur:
    """
    Blur gaussiano casuale.
    Simula motion blur, out-of-focus, compressione.
    """
    def __init__(self, radius_min=0.5, radius_max=2.0, p=0.25):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class StrongAugmentationForReal:
    """
    Augmentation pipeline FORTE per immagini reali.
    Obiettivo: ridurre overfitting su dettagli fini delle immagini reali.
    
    Applica più trasformazioni in sequenza con probabilità più alte.
    """
    def __init__(self):
        self.jpeg = RandomJPEGCompression(quality_min=40, quality_max=95, p=0.8)
        self.resize_down_up = RandomResizeDownUp(scale_min=0.4, scale_max=0.85, p=0.6)
        self.sharpening = RandomSharpening(factor_min=0.8, factor_max=2.5, p=0.5)
        self.noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.03, p=0.5)
        self.screenshot = RandomScreenshotArtifacts(p=0.4)
        self.blur = RandomBlur(radius_min=0.3, radius_max=2.5, p=0.4)

    def __call__(self, img: Image.Image):
        # Applica trasformazioni in ordine casuale
        transforms = [
            self.resize_down_up,
            self.jpeg,
            self.blur,
            self.noise,
            self.screenshot,
            self.sharpening,
        ]
        
        random.shuffle(transforms)
        
        for tf in transforms:
            img = tf(img)
        
        return img


class NormalAugmentationForReal:
    """
    Augmentation pipeline NORMALE per immagini reali.
    Meno aggressiva di StrongAugmentationForReal.
    """
    def __init__(self):
        self.jpeg = RandomJPEGCompression(quality_min=55, quality_max=95, p=0.55)
        self.resize_down_up = RandomResizeDownUp(scale_min=0.6, scale_max=0.9, p=0.3)
        self.sharpening = RandomSharpening(factor_min=1.0, factor_max=2.0, p=0.25)
        self.noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=0.35)
        self.screenshot = RandomScreenshotArtifacts(p=0.2)
        self.blur = RandomBlur(radius_min=0.5, radius_max=1.5, p=0.15)

    def __call__(self, img: Image.Image):
        # Applica alcune trasformazioni
        img = self.jpeg(img)
        img = self.resize_down_up(img)
        img = self.blur(img)
        img = self.noise(img)
        img = self.screenshot(img)
        img = self.sharpening(img)
        return img


class LightAugmentationForGenerated:
    """
    Augmentation LEGGERA per immagini generate.
    Le immagini generate hanno già artefatti propri, non serve augmentation pesante.
    """
    def __init__(self):
        self.jpeg = RandomJPEGCompression(quality_min=70, quality_max=95, p=0.4)
        self.noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.01, p=0.2)

    def __call__(self, img: Image.Image):
        img = self.jpeg(img)
        img = self.noise(img)
        return img
