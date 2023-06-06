"""
The zero-shot classification by adopting CLIP model.
"""
import torch
import torchvision.transforms as transforms

from plato.datasources import registry as datasources_registry
from plato.config import Config

from pflbases.models import clip
from pflbases.models import hard_prompts, prompts_template


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


device = Config().device()
clip_model = clip.CLIP()


dataset_name = Config().data.datasource
datasource = datasources_registry.get(
    train_transform=None, test_transform=transforms.ToTensor()
)
testset = datasource.get_test_set()

dataset_prompt_tmpl = prompts_template.templates[dataset_name]
data_classes_prompts = hard_prompts.create_classification_prompts(
    classes_name=dataset_prompt_tmpl["classes"],
    prompts_template=dataset_prompt_tmpl["templates"],
)

test_loader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=3)

class_prompts_embeddings = clip_model.get_text_prompts_embeddings(
    text_prompts=data_classes_prompts
)

whole_lables = []
whole_predictions = []

indx = 0

with torch.no_grad():
    for examples, labels in test_loader:
        outputs = clip_model.zeroshot_classification_forward(
            images=examples, text_prompts=data_classes_prompts
        )
        logits = outputs["logits"]
        whole_predictions.append(logits)
        whole_lables.append(labels)

        indx += 1

        if indx > 10:
            break

logits = torch.cat(whole_predictions, dim=0)
labels = torch.cat(whole_lables, dim=0)

acc_results = accuracy(logits, labels, (1, 5))

print(f"top-1 accuracy for {dataset_name} dataset: {acc_results[0]:.3f}")
print(f"top-5 accuracy for {dataset_name} dataset: {acc_results[1]:.3f}")
