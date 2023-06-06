""" 
Implementation of hard prompts for different datasets.

Prompts of different datasets can be accessed on the website,
https://github.com/openai/CLIP/blob/main/data/prompts.md
"""


from typing import List


def create_classification_prompts(classes_name: List[str], prompts_template: List[str]):
    """Creating prompts for classification tasks."""

    text_prompts = []

    for name in classes_name:
        texts = [template.format(name) for template in prompts_template]
        text_prompts.append(texts)

    return text_prompts
