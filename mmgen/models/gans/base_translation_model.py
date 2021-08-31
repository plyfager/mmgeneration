from abc import ABCMeta, abstractmethod
from copy import deepcopy

import torch.nn as nn

from ..builder import MODELS


@MODELS.register_module()
class BaseTranslationModel(nn.Module, metaclass=ABCMeta):
    """Base Translation Model.

    Translation models can transfer images from one domain to
    another. Domain information like `default_domain`,
    `reachable_domains` are needed to initialize the class.
    And we also provide query functions like `is_domain_reachable`,
    `_get_other_domains`.

    You can get a specific generator based on the domain,
    and by specifying `target_domain` in the forward function,
    you can decide the domain of generated images.
    Considering the difference of different image translation models,
    we only provide the external interfaces mentioned above.
    When you implement image translation with a specific method,
    you can inherit both `BaseTranslationModel`
    and the method (e.g BaseGAN) and implement abstract methods.

    Args:
        default_domain (str): Default output domain.
        reachable_domains (list[str]): Domains that can be generated by
            the model.
        related_domains (list[str]): Domains involved in training and
            testing.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 default_domain,
                 reachable_domains,
                 related_domains,
                 train_cfg=None,
                 test_cfg=None):
        self._default_domain = default_domain
        self._reachable_domains = reachable_domains
        self._related_domains = related_domains
        assert self._default_domain in self._reachable_domains
        assert set(self._reachable_domains) <= set(self._related_domains)

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    @abstractmethod
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""

    @abstractmethod
    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""

    def forward(self, img, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img (tensor): Input image tensor.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if not test_mode:
            return self.forward_train(img, **kwargs)

        return self.forward_test(img, **kwargs)

    def forward_train(self, img, target_domain, **kwargs):
        """Forward function for training.

        Args:
            img (tensor): Input image tensor.
            target_domain (str): Target domain of output image.
            kwargs (dict): Other arguments.

        Returns:
            Dict: Forward results.
        """
        target = self.translation(img, target_domain=target_domain, **kwargs)
        results = dict(source=img, target=target)
        return results

    def forward_test(self, img, target_domain, **kwargs):
        """Forward function for testing.

        Args:
            img (tensor): Input image tensor.
            target_domain (str): Target domain of output image.
            kwargs (dict): Other arguments.

        Returns:
            Dict: Forward results.
        """
        target = self.translation(img, target_domain=target_domain, **kwargs)
        results = dict(source=img.cpu(), target=target.cpu())
        return results

    def is_domain_reachable(self, domain):
        """Whether image of this domain can be generated."""
        return domain in self._reachable_domains

    def _get_other_domains(self, domain):
        """get other domains."""
        return list(set(self._related_domains) - set([domain]))

    @abstractmethod
    def _get_domain_generator(self, domain):
        """get domain generator."""

    def translation(self, image, target_domain=None, **kwargs):
        """Translation Image to target style.

        Args:
            image (tensor): Image tensor with a shape of (N, C, H, W).
            target_domain (str, optional): Target domain of output image.
                Default to None.

        Returns:
            Dict: Image tensor of target style.
        """
        if target_domain is None:
            target_domain = self._default_domain
        _model = self._get_domain_generator(target_domain)
        outputs = _model(image, **kwargs)
        return outputs
