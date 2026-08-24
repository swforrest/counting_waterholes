"""
Central class registry for the waterhole detection pipeline.

Replaces hardcoded assumptions of exactly 5 waterhole classes (Dry_WH, WH_swamp,
WH_wet, WH_sink, U) with a registry built from whatever `names:` /
`class_distance_cutoff_px:` a given config file declares. See the config files'
"Class definitions" comment block for the required YAML schema.
"""
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassRegistry:
    id_to_name: dict
    name_to_id: dict
    id_to_threshold: dict

    @property
    def ids(self):
        """Class ids in ascending order."""
        return sorted(self.id_to_name)

    @property
    def names(self):
        """Class names, in id order."""
        return [self.id_to_name[i] for i in self.ids]


def load_class_registry(config: dict, require_thresholds: bool = True) -> ClassRegistry:
    """
    Build a ClassRegistry from an already-loaded config dict.

    Args:
        config: parsed config dict, must contain "names" (dict[int, str], the
            standard YOLOv5 dataset-yaml id->name mapping).
        require_thresholds: if True, also requires "class_distance_cutoff_px"
            (dict[str, number], keyed by the names in "names") and populates
            id_to_threshold. If False, id_to_threshold is left empty (used by
            call sites that only need id<->name, e.g. while building training
            labels, where clustering thresholds are irrelevant).

    Raises:
        ValueError: if "names" is missing, its ids aren't contiguous from 0,
            it has duplicate names, or (require_thresholds=True) a class in
            "names" has no matching entry in "class_distance_cutoff_px".
    """
    if "names" not in config:
        raise ValueError(
            "config is missing required 'names:' block (id -> class name mapping). "
            "See the config file's 'Class definitions' comment for the required format."
        )

    raw_names = config["names"]
    id_to_name = {int(cid): str(name) for cid, name in raw_names.items()}

    ids = sorted(id_to_name)
    if ids != list(range(len(ids))):
        raise ValueError(
            f"class ids in 'names:' must be contiguous integers starting at 0; got {ids}"
        )

    names = list(id_to_name.values())
    dupes = {name for name in names if names.count(name) > 1}
    if dupes:
        raise ValueError(f"duplicate class name(s) in 'names:': {sorted(dupes)}")

    name_to_id = {name: cid for cid, name in id_to_name.items()}

    id_to_threshold = {}
    if require_thresholds:
        if "class_distance_cutoff_px" not in config:
            raise ValueError(
                "config is missing required 'class_distance_cutoff_px:' block "
                "(class name -> pixel distance mapping). See the config file's "
                "'Class definitions' comment for the required format."
            )
        cutoffs = config["class_distance_cutoff_px"]

        missing = [name for name in name_to_id if name not in cutoffs]
        if missing:
            raise ValueError(
                "the following classes are defined in 'names:' but have no matching "
                f"entry in 'class_distance_cutoff_px:': {missing}. Add "
                f"'class_distance_cutoff_px.{missing[0]}: <pixels>' (and the others) "
                "to the config."
            )

        extra = [name for name in cutoffs if name not in name_to_id]
        if extra:
            warnings.warn(
                "'class_distance_cutoff_px' has entries not present in 'names:': "
                f"{extra}. They will be ignored."
            )

        id_to_threshold = {name_to_id[name]: cutoffs[name] for name in name_to_id}

    return ClassRegistry(
        id_to_name=id_to_name,
        name_to_id=name_to_id,
        id_to_threshold=id_to_threshold,
    )
