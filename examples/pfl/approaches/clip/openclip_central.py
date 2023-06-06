"""
The zero-shot classification by adopting CLIP model.
"""
import torch

from plato.datasources import registry as datasources_registry
from plato.config import Config

from pflbases.models import open_clip
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
clip_model = open_clip.CLIP()


dataset_name = Config().data.datasource
datasource = datasources_registry.get(
    train_transform=None, test_transform=clip_model.preprocesser
)
testset = datasource.get_test_set()

text_descriptions = [f"A photo of a {label}" for label in testset.classes]

test_loader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=3)


whole_lables = []
whole_predictions = []

indx = 0

with torch.no_grad():
    for examples, labels in test_loader:
        outputs = clip_model.zeroshot_classification_forward(
            images=examples, text_prompts=text_descriptions
        )

        whole_predictions.append(outputs["probs"])
        whole_lables.append(labels)

        indx += 1

        if indx > 10:
            break
        print(indx)

logits = torch.cat(whole_predictions, dim=0)
labels = torch.cat(whole_lables, dim=0)

acc_results = accuracy(logits, labels, (1, 5))

print(f"top-1 accuracy for {dataset_name} dataset: {acc_results[0]:.3f}")
print(f"top-5 accuracy for {dataset_name} dataset: {acc_results[1]:.3f}")
