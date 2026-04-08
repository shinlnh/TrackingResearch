from .lasot import Lasot


class LasotPerson(Lasot):
    """LaSOT restricted to the person class."""

    def __init__(self, root=None, image_loader=None, split='train', data_fraction=None):
        super().__init__(root=root, image_loader=image_loader, split=split, data_fraction=data_fraction)
        self.sequence_list = [seq for seq in self.sequence_list if seq.startswith('person-')]
        self.seq_per_class = self._build_class_list()

