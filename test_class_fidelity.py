"""Classifier normalization, confusion aggregation and selection/seed checks."""

import copy

import pytest
import torch

from class_fidelity import ClassAgreement
from dit_followup import select, statistics_from_rows, FAMILIES


class InspectClassifier(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.inputs = None

    def forward(self, images):
        self.inputs = images.clone()
        return torch.nn.functional.one_hot(torch.tensor(self.predictions, device=images.device), 10).float()


def test_agreement_normalization_counts_and_frozen_parameters():
    model = InspectClassifier([1, 2, 1])
    metric = ClassAgreement(model, torch.device("cpu"))
    images = torch.zeros(3, 3, 32, 32)
    metric.update(images, torch.tensor([1, 2, 2]), "real")
    assert not model.training and not model.anchor.requires_grad
    torch.testing.assert_close(model.inputs[0, :, 0, 0], -metric.mean.flatten() / metric.std.flatten())
    metric.update(images, torch.tensor([1, 0, 1]), "generated")
    result = metric.compute()
    assert result['real']['examples'] == result['generated']['examples'] == 3
    assert result['real']['agreement'] == pytest.approx(2/3)
    assert result['generated']['per_class_agreement'][1] == 1
    assert result['generated']['predicted_class_counts'][1] == 2
    assert result['real']['confusion_rows_requested_columns_predicted'][2][1] == 1


@pytest.mark.parametrize("bad", [float('nan'), -1., 2.])
def test_metric_rejects_bad_image_range(bad):
    metric = ClassAgreement(InspectClassifier([0]), torch.device('cpu'))
    with pytest.raises(ValueError):
        metric.update(torch.full((1,3,32,32), bad), torch.tensor([0]), 'generated')


def row(key, seed, fid, lr=2e-4, mse=.1):
    variant, width = FAMILIES[key]
    return {'config': {'variant': variant, 'width': width, 'seed': seed, 'lr': lr, 'adapter_lr_multiplier': 1},
            'generative_evaluation': {'fid':fid,'class_agreement':{'generated':{'agreement':.25}}},
            'ema_validation': {'noise_mse':mse}, 'parameters':100, 'forward_matmul_flops_per_image_estimate':200,
            'mean_train_images_per_second_excluding_warmup':300}


def test_selection_uses_fid_before_denoising_and_frozen_lr_tie_break():
    rows=[row('dplr',42,100,4e-4,.08),row('dplr',42,99,2e-4,.09),row('dplr',42,99,1e-4,.09)]
    before=copy.deepcopy(rows)
    assert select(rows)['dplr']['config']['lr']==1e-4
    assert rows==before


def test_seed_statistics_and_paired_differences():
    rows=[row(key,seed,seed-40+(1 if key=='dplr' else 0)) for key in FAMILIES for seed in (43,44,45)]
    stats=statistics_from_rows(rows)
    assert stats['adaln_d192']['fid']['mean']==4 and stats['adaln_d192']['fid']['sample_sd']==1
    assert stats['dplr']['paired_fid_change_vs_adaln_d192']==[1,1,1]
    assert stats['dplr']['paired_fid_change_vs_width_controls']['adaln_d240']==[1,1,1]
    with pytest.raises(ValueError): statistics_from_rows(rows[:-1])


def test_generative_pipeline_streams_balanced_images_to_both_evaluators(monkeypatch, tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from train_dit_cifar10 import Config, generative_evaluation, make_noise_scheduler
    import torchmetrics.image.fid

    calls = []
    class FakeFID:
        def __init__(self, **kwargs): pass
        def to(self, device): return self
        def update(self, images, real):
            assert images.dtype == torch.uint8 and images.shape[1:] == (3,32,32)
            calls.append((real, len(images)))
        def compute(self): return torch.tensor(1.)
    class ConstantClassifier(torch.nn.Module):
        def forward(self, images): return torch.zeros(len(images),10)
    class Predictor(torch.nn.Module):
        def forward(self, images, timesteps, labels): return images*.1
    monkeypatch.setattr(torchmetrics.image.fid, 'FrechetInceptionDistance', FakeFID)
    monkeypatch.setattr(ClassAgreement, 'from_run', classmethod(lambda cls, directory, device, indices:
                        cls(ConstantClassifier(),device)))
    data=TensorDataset(torch.zeros(10,3,32,32),torch.arange(10))
    data.indices=list(range(10))
    cfg=Config(device='cpu',precision='fp32',classifier_run='mock',fid_images=10,fid_batch_size=5,
               sample_images=10,num_train_timesteps=10,sampling_steps=3)
    result=generative_evaluation(Predictor(),make_noise_scheduler(cfg),DataLoader(data,batch_size=5),
                                 cfg,torch.device('cpu'),tmp_path)
    assert calls==[(True,5),(True,5),(False,5),(False,5)]
    assert result['class_agreement']['generated']['class_counts']==[1]*10
    assert result['class_agreement']['real']['class_counts']==[1]*10
    assert result['class_agreement']['generated']['agreement']==pytest.approx(.1)
    assert result['sample_grid_labels']==list(range(10)) and (tmp_path/'samples_ema.png').exists()
