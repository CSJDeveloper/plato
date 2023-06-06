"""
Prompts of Templates for different datasets.

Three datasets are supports. They are:
- CIFAR-10
- CIFAR-100
- ImageNet

These templates are directly copied from
https://github.com/openai/CLIP/blob/main/data/prompts.md

"""

prompts_template = {
    "CIFAR1O": {
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "templates": [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a black and white photo of a {}.",
            "a low contrast photo of a {}.",
            "a high contrast photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of a small {}.",
            "a photo of a big {}.",
            "a photo of the {}.",
            "a blurry photo of the {}.",
            "a black and white photo of the {}.",
            "a low contrast photo of the {}.",
            "a high contrast photo of the {}.",
            "a bad photo of the {}.",
            "a good photo of the {}.",
            "a photo of the small {}.",
            "a photo of the big {}.",
        ],
    }
}
