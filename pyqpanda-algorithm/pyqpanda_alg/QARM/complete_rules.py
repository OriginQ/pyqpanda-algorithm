from itertools import combinations

from .qarm import QuantumAssociationRulesMining as _QuantumAssociationRulesMining


class QuantumAssociationRulesMining(_QuantumAssociationRulesMining):
    """QARM implementation with complete association-rule enumeration."""

    def _get_all_conf(self, qlist, clist, position, show, file_name, machine_type):
        fn, fn_dict = self._fk_result(
            qlist, clist, position, show, file_name, machine_type
        )
        if len(fn) < 2:
            return None

        support_by_itemset = {
            frozenset(itemset): data[1]
            for itemset, data in fn_dict.items()
        }
        conf_dict = {}
        for level in fn[1:]:
            for itemset in level:
                itemset_set = frozenset(itemset)
                supp_xy = support_by_itemset[itemset_set]
                ordered_items = sorted(itemset_set)
                for cause_size in range(1, len(ordered_items)):
                    for cause_items in combinations(ordered_items, cause_size):
                        cause = frozenset(cause_items)
                        supp_x = support_by_itemset[cause]
                        conf = self._conf_x_y(supp_xy, supp_x)
                        if conf >= self.min_conf:
                            effect = itemset_set - cause
                            key = self._get_conf_key(cause, effect)
                            conf_dict[key] = conf
        return conf_dict

    def _get_conf_key(self, cause, effect):
        cause_str = ','.join(
            self.items_dict[item] for item in sorted(cause)
        )
        effect_str = ','.join(
            self.items_dict[item] for item in sorted(effect)
        )
        return cause_str + '->' + effect_str
